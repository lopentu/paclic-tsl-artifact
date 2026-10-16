"""Frame decoding shared by the cross-model replication backends (Gemma 4, GLM-4.6V).

The cross-model comparison only means something if every model sees the same pixels. Each
checkpoint ships its own video sampler, and their defaults disagree badly with the main run:
Gemma 4 samples a fixed 32 frames (and raises on a clip shorter than that), and GLM-4.6V ignores
the requested fps altogether and samples ~6 frames/s, collapsing a 0.51 s target clip to one
duplicated frame. Either would make a replication failure uninterpretable -- a model that never
saw the movement cannot be said to have failed to perceive it.

So these backends never use a model's own sampler. Frames are decoded here with the exact call
the Qwen3-VL main run used (qwen_vl_utils.fetch_video at the configured fps and per-frame pixel
budget), and handed to each processor with `do_sample_frames=False` and a VideoMetadata that
places frame i at i/fps seconds -- the same convention as Qwen3VLBackend._video_metadata. What
differs between models is then the model, not the frame selection or resolution.
"""

import os


def load_frames(path, fps, max_pixels):
    """Return (frames uint8 tensor [T, C, H, W], VideoMetadata) for one clip file."""
    import torch
    from qwen_vl_utils.vision_process import fetch_video
    from transformers.video_utils import VideoMetadata

    video, sample_fps = fetch_video(
        {"video": f"file://{os.path.abspath(path)}", "fps": fps, "max_pixels": max_pixels},
        return_video_sample_fps=True,
    )
    frames = video.round().clamp(0, 255).to(torch.uint8)
    n = int(frames.shape[0])
    metadata = VideoMetadata(
        total_num_frames=n, fps=float(sample_fps), frames_indices=list(range(n)),
        duration=n / float(sample_fps),
    )
    return frames, metadata


def build_content(video_path, prompt_text, context_video_path, context_text,
                  video_label, context_video_label, fps, max_pixels):
    """Chat content (video placeholders + text, in the Qwen backends' order) plus decoded videos.

    Ordering mirrors Qwen3VLBackend.generate exactly: context label, context video, context note,
    target label, target video, prompt. Returns (content, videos, metadata, frame_counts).
    """
    content, videos, metadata, frame_counts = [], [], [], []

    def add_video(path):
        frames, meta = load_frames(path, fps, max_pixels)
        content.append({"type": "video"})
        videos.append(frames)
        metadata.append(meta)
        frame_counts.append(int(frames.shape[0]))

    if context_video_path is not None:
        content.append({"type": "text", "text": context_video_label})
        add_video(context_video_path)
    if context_text:
        content.append({"type": "text", "text": f"Context note: {context_text}"})
    if video_path is not None:
        content.append({"type": "text", "text": video_label})
        add_video(video_path)
    content.append({"type": "text", "text": prompt_text})
    return content, videos, metadata, frame_counts
