"""Qwen3-VL backend (open-source weights on Hugging Face).

Primary backend. Qwen3-VL-8B-Instruct runs unquantized (bf16) on a single 24GB GPU, which is
how every reported run was made; 4-bit loading is available but no reported run uses it.

Two things this backend does deliberately:
  1. 4-bit NF4 loading via BitsAndBytesConfig (opt-in through `quantization`).
  2. It forwards the video sampling metadata returned by qwen_vl_utils to the processor
     (`return_video_kwargs=True` -> `**video_kwargs`), so the configured `fps` is actually
     honoured. Without it the Qwen3-VL processor warns "no video metadata ... Defaulting to
     fps=24" and silently samples at 24fps instead of the configured rate.

Requires: pip install -r requirements-model.txt  (torch, transformers>=5.x,
qwen-vl-utils; plus `bitsandbytes` when quantization is enabled).
"""

from .base import VLMBackend


class Qwen3VLBackend(VLMBackend):
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct", device="cuda", fps=8,
                 max_pixels=360 * 420, quantization=None):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL backend requires torch, transformers>=5.x and qwen-vl-utils. "
                "Install with: pip install -r requirements-model.txt"
            ) from exc

        self.model_name = "Qwen3-VL"
        self.fps = fps
        self.max_pixels = max_pixels
        self.quantization = quantization
        self._torch = torch

        load_kwargs = {}
        if quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            if quantization == "4bit":
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                bnb = BitsAndBytesConfig(load_in_8bit=True)
            # Pin everything to the single GPU (no CPU offload) so an over-budget model raises
            # a clean CUDA OOM we can fall back on, rather than silently offloading and crawling.
            device_map = {"": 0} if str(device).startswith("cuda") else device
            load_kwargs = dict(quantization_config=bnb, torch_dtype=torch.bfloat16, device_map=device_map)
            self.model_version = f"{model_id} [{quantization}-nf4]" if quantization == "4bit" else f"{model_id} [{quantization}]"
        else:
            load_kwargs = dict(torch_dtype="auto", device_map=device)
            self.model_version = model_id

        # Resolve the model class from the checkpoint instead of hardcoding one: the dense and
        # MoE Qwen3-VL variants use different classes, and loading a checkpoint into the wrong
        # class does not fail loudly.
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _video_block(self, path):
        return {
            "type": "video",
            "video": f"file://{path}",
            "fps": self.fps,
            "max_pixels": self.max_pixels,
        }

    def _process_vision(self, messages):
        """Call qwen_vl_utils.process_vision_info, returning (images, videos, video_kwargs).

        Newer qwen-vl-utils returns per-video sampling metadata (fps, frame counts) when asked;
        forwarding it to the processor is what makes the configured fps stick. Falls back
        gracefully on older versions that lack `return_video_kwargs`.
        """
        from qwen_vl_utils import process_vision_info
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            return image_inputs, video_inputs, video_kwargs
        except TypeError:
            image_inputs, video_inputs = process_vision_info(messages)
            return image_inputs, video_inputs, {}

    @staticmethod
    def _n_frames(video):
        shape = getattr(video, "shape", None)
        return int(shape[0]) if shape is not None else len(video)

    def _video_metadata(self, video_inputs, video_kwargs):
        """Build one VideoMetadata per pre-sampled video so frame timestamps are correct.

        qwen_vl_utils has already sampled the frames by the time we get here, and it reports
        the sampled rate as a per-video LIST (`{'fps': [8.0]}`). Handing that straight to the
        processor is wrong twice over on transformers 5.x: the `fps` field is validated as a
        scalar and rejects the list, and even when a scalar gets through, the processor cannot
        derive per-frame timestamps from `fps` alone -- it warns that `video_metadata` is
        missing and silently DEFAULTS TO fps=24. That default is not cosmetic: Qwen3-VL writes
        a "<N.N s>" marker before each frame, so at a configured fps of 8 the model would be
        told an 8-second clip is 2.7 seconds long, corrupting every timestamp it reads or
        reports. Describing the sampled sequence in its own terms (frame i sits at i/fps
        seconds) reproduces true wall-clock time without needing the source frame indices.
        """
        try:
            from transformers.video_utils import VideoMetadata
        except ImportError:  # older transformers: no metadata plumbing to fill
            return None

        fps_field = video_kwargs.get("fps")
        fps_list = list(fps_field) if isinstance(fps_field, (list, tuple)) else None
        if not fps_list:  # absent, or an empty list -- fall back to the configured rate
            fps_list = None
        metadata = []
        for i, video in enumerate(video_inputs):
            if fps_list is not None:
                fps = float(fps_list[i]) if i < len(fps_list) else float(fps_list[-1])
            else:
                fps = float(fps_field if fps_field is not None else self.fps)
            n = self._n_frames(video)
            metadata.append(VideoMetadata(
                total_num_frames=n, fps=fps, frames_indices=list(range(n)), duration=n / fps,
            ))
        return metadata

    def generate(self, video_path, prompt_text, context_video_path=None,
                 context_text=None, temperature=0.0, do_sample=False,
                 max_new_tokens=1500,
                 video_label="The following clip is the TARGET analysis window:",
                 context_video_label="The following clip is PRECEDING DISCOURSE CONTEXT only, not the target window:"):
        """`video_label` / `context_video_label` are the sentences introducing the main video and
        the optional context video. The defaults are the Prompt C wording for the older local/
        context condition and must stay as-is so those runs are unaffected; 07b_run_inference_events.py's `combined` clip kind overrides
        both (context video is the same-passage window clip, not preceding discourse).

        `video_path=None` sends the prompt with NO video attached, for the text-only visual-null
        baseline (19_run_visual_null_baselines.py). The prompt text is passed through unaltered --
        it still refers to "this clip" -- because editing it would change two things at once and
        stop the condition being an ablation of the video alone. The introducing `video_label` is
        dropped, since there is nothing for it to introduce."""
        content = []
        if context_video_path is not None:
            content.append({"type": "text", "text": context_video_label})
            content.append(self._video_block(context_video_path))
        if context_text:
            content.append({"type": "text", "text": f"Context note: {context_text}"})
        if video_path is not None:
            content.append({"type": "text", "text": video_label})
            content.append(self._video_block(video_path))
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = self._process_vision(messages)
        processor_kwargs = dict(video_kwargs)
        if not video_inputs:
            # Text-only call (video_path=None, no context video): qwen_vl_utils still returns a
            # per-video `fps` key, as an EMPTY LIST, and the processor validates that field as a
            # scalar and raises on a list. Drop it -- there is no video for it to describe.
            processor_kwargs.pop("fps", None)
        else:
            metadata = self._video_metadata(video_inputs, video_kwargs)
            if metadata is not None:
                # video_metadata carries the sampled rate, so drop the raw per-video `fps` list
                # the processor would reject, and keep frames as already sampled.
                processor_kwargs.pop("fps", None)
                processor_kwargs["video_metadata"] = metadata
                processor_kwargs["do_sample_frames"] = False
        inputs = self.processor(
            text=[chat_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt", **processor_kwargs,
        ).to(self.model.device)

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)

        with self._torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0]
