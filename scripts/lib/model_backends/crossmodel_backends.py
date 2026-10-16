"""Backends for the further models: GLM-4.6V-Flash, Gemma 4 E4B and MiMo-VL-7B.

GLM-4.6V-Flash and Gemma 4 are non-Qwen in both the language tower and the vision encoder, and
both accept two videos in one chat turn through HF transformers, so they can run the main
`combined` format (window clip + tight event clip) with the prompt byte-identical. MiMo-VL-7B
(MimoVLBackend, at the end of this module) runs through the Qwen3-VL path. Frames come from shared_frames.load_frames, never the model's
own sampler -- see that module for why.

Thinking is off for both, to match the main run's direct greedy answer: Gemma 4 only thinks when
the template is given enable_thinking=True, and GLM-4.6V's template appends `/nothink` when
enable_thinking=False. A reasoning block that leaks anyway is NOT stripped: it reaches the same
extract_json_object/validate_output as every Qwen run and counts against schema validity.

`last_call_info` records the frames actually fed per video, so 07b can log them per output.
"""

from .base import VLMBackend
from .shared_frames import build_content

DEFAULT_VIDEO_LABEL = "The following clip is the TARGET analysis window:"
DEFAULT_CONTEXT_LABEL = "The following clip is PRECEDING DISCOURSE CONTEXT only, not the target window:"


class _FramesFedBackend(VLMBackend):
    #: kwargs passed to apply_chat_template to keep thinking off
    template_kwargs = {}

    def __init__(self, model_id, device="cuda", fps=16, max_pixels=360 * 420):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.fps = fps
        self.max_pixels = max_pixels
        self.model_version = model_id
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Pinned here: offloading a module to CPU (GLM) changes what `model.device` reports.
        self.input_device = self.model.device
        self.last_call_info = None

    def generate(self, video_path, prompt_text, context_video_path=None,
                 context_text=None, temperature=0.0, do_sample=False,
                 max_new_tokens=1500, video_label=DEFAULT_VIDEO_LABEL,
                 context_video_label=DEFAULT_CONTEXT_LABEL):
        content, videos, metadata, frame_counts = build_content(
            video_path, prompt_text, context_video_path, context_text,
            video_label, context_video_label, self.fps, self.max_pixels)
        messages = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs)

        proc_kwargs = {}
        if videos:
            proc_kwargs = dict(videos=videos, video_metadata=metadata, do_sample_frames=False)
        inputs = self.processor(text=[chat_text], return_tensors="pt", **proc_kwargs)
        inputs = inputs.to(self.input_device)

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
        else:
            # Neither checkpoint's generation_config is greedy by default; the main run is.
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)

        with self._torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        n_in = inputs["input_ids"].shape[1]
        self.last_call_info = {
            "frames_per_video": frame_counts,
            "sampled_fps_per_video": [round(m.fps, 4) for m in metadata],
            "input_tokens": int(n_in),
            "output_tokens": int(out.shape[1] - n_in),
        }
        return self.processor.batch_decode(
            out[:, n_in:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class Gemma4Backend(_FramesFedBackend):
    template_kwargs = {"enable_thinking": False}

    def __init__(self, model_id="google/gemma-4-E4B-it", **kwargs):
        self.model_name = "Gemma-4"
        super().__init__(model_id, **kwargs)
        self._allow_unequal_video_lengths()

    def _allow_unequal_video_lengths(self):
        """Let one call carry a 128-frame window and a 22-frame target clip.

        The stock video processor stacks all videos of a call into one (num_videos, num_frames,
        ...) tensor, so two clips of different lengths raise. The model immediately flattens
        (num_videos, num_frames) into a single frame axis and encodes every frame independently
        (Gemma4Model.get_video_features), so concatenating the videos along the frame axis is the
        same computation. Per-video soft-token counts and timestamps are kept per video, so the
        prompt still introduces two separate clips.
        """
        import torch
        from transformers.feature_extraction_utils import BatchFeature

        vp = self.processor.video_processor
        original = vp._preprocess

        def per_video(videos, **kwargs):
            if len(videos) <= 1:
                return original(videos, **kwargs)
            parts = [original([v], **kwargs) for v in videos]
            return BatchFeature(data={
                "pixel_values_videos": torch.cat([p["pixel_values_videos"] for p in parts], dim=1),
                "video_position_ids": torch.cat([p["video_position_ids"] for p in parts], dim=1),
                "num_soft_tokens_per_video": [int(n) for p in parts
                                              for n in torch.as_tensor(p["num_soft_tokens_per_video"]).flatten()],
            }, tensor_type=kwargs.get("return_tensors"))

        vp._preprocess = per_video


class Glm46VBackend(_FramesFedBackend):
    template_kwargs = {"enable_thinking": False}
    #: temporal units (2 frames each) per vision-encoder pass
    vision_chunk = 16

    def __init__(self, model_id="zai-org/GLM-4.6V-Flash", **kwargs):
        self.model_name = "GLM-4.6V"
        super().__init__(model_id, **kwargs)
        self._chunk_vision_encoder()
        self._embeddings_and_head_on_cpu()

    def _embeddings_and_head_on_cpu(self):
        """Move the two vocab-sized matrices (2 x 1.24 GB) off the GPU so the 23k-token prefill of
        the longest item fits beside the remaining weights.

        Neither move changes what is computed. The embedding is a table lookup. The head is
        evaluated in fp32 on the CPU and rounded to bf16, which is what the GPU bf16 matmul does
        (fp32 accumulate, bf16 output). It runs once per generated token, so the cost is small.
        """
        import torch

        gpu = self.input_device
        embed = self.model.model.language_model.embed_tokens.to("cpu")
        embed.register_forward_pre_hook(lambda mod, args: (args[0].to("cpu"),) + tuple(args[1:]))
        embed.register_forward_hook(lambda mod, args, out: out.to(gpu))

        class CpuHead(torch.nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.out_dtype = linear.weight.dtype
                self.weight = torch.nn.Parameter(linear.weight.detach().to("cpu", torch.float32),
                                                 requires_grad=False)

            def forward(self, hidden):
                logits = hidden.to("cpu", torch.float32) @ self.weight.T
                return logits.to(self.out_dtype).to(gpu)

        self.model.lm_head = CpuHead(self.model.lm_head)
        torch.cuda.empty_cache()

    def _chunk_vision_encoder(self):
        """Encode video in chunks of temporal units so the 11.9 s window fits beside 20.6 GB of weights.

        Glm46VModel.get_video_features already splits each video into independent [1, h, w] grids,
        one per temporal unit, and the vision attention never crosses a grid (cu_seqlens), so
        running the units through the encoder a chunk at a time gives the same embeddings while
        holding only one chunk's activations in memory.
        """
        import torch
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        inner = self.model.model
        chunk = self.vision_chunk

        def get_video_features(pixel_values_videos, video_grid_thw=None, **kwargs):
            kwargs.pop("return_dict", None)
            pixel_values_videos = pixel_values_videos.type(inner.visual.dtype)
            hw = torch.repeat_interleave(video_grid_thw[:, 1:], video_grid_thw[:, 0], dim=0)
            grids = torch.cat([hw.new_ones(hw.shape[0], 1), hw], dim=1)
            rows = grids.prod(-1).tolist()
            pooled, start = [], 0
            for i in range(0, len(rows), chunk):
                n = sum(rows[i:i + chunk])
                out = inner.visual(pixel_values_videos[start:start + n], grid_thw=grids[i:i + chunk],
                                   return_dict=True, **kwargs)
                pooled.append(out.pooler_output)
                start += n
            split_sizes = (video_grid_thw.prod(-1) // inner.visual.spatial_merge_size ** 2).tolist()
            return BaseModelOutputWithPooling(pooler_output=torch.split(torch.cat(pooled), split_sizes))

        self._unchunked_get_video_features = inner.get_video_features
        inner.get_video_features = get_video_features


class MimoVLBackend(VLMBackend):
    """MiMo-VL-7B-RL-2508: MiMo-7B language tower on a Qwen2.5-VL-compatible stack.

    Runs through Qwen3VLBackend's own path (qwen_vl_utils sampling + VideoMetadata), which is the
    main run's frame pipeline verbatim. The one input difference is MiMo's documented thinking
    switch: `/no_think` as the final text of the user turn. It is appended after the unchanged
    prompt, and recorded in every output via model_version.
    """

    NO_THINK = "/no_think"

    def __init__(self, model_id="XiaomiMiMo/MiMo-VL-7B-RL-2508", device="cuda", fps=16,
                 max_pixels=360 * 420):
        from .qwen3_vl_backend import Qwen3VLBackend
        self._inner = Qwen3VLBackend(model_id=model_id, device=device, fps=fps,
                                     max_pixels=max_pixels)
        self.model_name = "MiMo-VL"
        self.model_version = f"{model_id} [+{self.NO_THINK}]"
        self.last_call_info = None

    def generate(self, video_path, prompt_text, **kwargs):
        return self._inner.generate(video_path, f"{prompt_text}\n{self.NO_THINK}", **kwargs)
