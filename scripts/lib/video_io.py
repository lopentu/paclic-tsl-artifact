"""Thin ffmpeg/ffprobe wrappers used by the data-prep scripts.

Requires ffmpeg + ffprobe on PATH (part of the standard ffmpeg distribution, open source).
All functions shell out via subprocess rather than depending on a Python binding, so the
scripts keep working regardless of which Python video library (if any) is installed.
"""

import json
import random
import subprocess
import tempfile
from pathlib import Path


def ffprobe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def ffprobe_dimensions(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def cut_clip(src, start_sec, end_sec, dst, reencode=True, keep_audio=True):
    """Cut [start_sec, end_sec) from src into dst. Re-encodes by default for frame-accurate cuts.

    `keep_audio=False` drops the audio stream entirely. The VLM backends only ever
    consume sampled frames, so audio never reaches a model either way.
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, float(end_sec) - float(start_sec))
    cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", str(src), "-t", str(duration)]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        cmd += ["-an"] if not keep_audio else ["-c:a", "aac"]
    else:
        cmd += ["-an", "-c:v", "copy"] if not keep_audio else ["-c", "copy"]
    cmd += [str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(dst)


def crop_scale_clip(src, dst, region, scale=2.0):
    """Crop `region` (fractions of frame width/height, origin top-left) and upscale by `scale`.

    Gives the vision encoder more patches over a small, distant face. Note what this does and does not do: upscaling adds no information, it
    only stops the full frame being downscaled to the per-frame pixel budget and gives the
    encoder a denser patch grid over the face. Any improvement is attributable to patch
    coverage, not to new detail.
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    w, h = ffprobe_dimensions(src)
    x, y = int(round(region["x"] * w)), int(round(region["y"] * h))
    cw, ch = int(round(region["w"] * w)), int(round(region["h"] * h))
    # libx264 needs even dimensions.
    cw, ch = cw - (cw % 2), ch - (ch % 2)
    sw, sh = int(round(cw * scale)), int(round(ch * scale))
    sw, sh = sw - (sw % 2), sh - (sh % 2)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"crop={cw}:{ch}:{x}:{y},scale={sw}:{sh}:flags=lanczos",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(dst)


def crop_blur_regions(src, dst, regions, blur_strength=25):
    """Blur one or more rectangular regions (fractions of frame width/height) and re-composite.

    Used to de-identify burned-in subtitles (and, where present, other on-screen text
    overlays) before a window is sent to the model. Everywhere outside the given regions is left untouched so
    signing stays fully visible. Regions are cropped/blurred independently from the
    original frame and composited in sequence, so overlapping regions are safe too.
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    w, h = ffprobe_dimensions(src)

    filter_parts = []
    stage = "0:v"
    for i, region in enumerate(regions):
        x, y = int(round(region["x"] * w)), int(round(region["y"] * h))
        bw, bh = int(round(region["w"] * w)), int(round(region["h"] * h))
        # ffmpeg's boxblur chroma radius maxes out at 16 on this build, and is further
        # capped by the chroma (half-resolution) plane size of the region being blurred --
        # empirically, radius must stay <= min(bw, bh) // 4 or ffmpeg rejects the filter.
        radius = min(int(blur_strength), 16, max(1, min(bw, bh) // 4))
        blurred = f"blurred{i}"
        next_stage = f"stage{i}"
        filter_parts.append(f"[0:v]crop={bw}:{bh}:{x}:{y},boxblur={radius}:2:{radius}:2[{blurred}]")
        filter_parts.append(f"[{stage}][{blurred}]overlay={x}:{y}[{next_stage}]")
        stage = next_stage
    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-filter_complex", filter_complex,
        "-map", f"[{stage}]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(dst)


def crop_blur_region(src, dst, x_frac, y_frac, w_frac, h_frac, blur_strength=25):
    """Single-region convenience wrapper around crop_blur_regions."""
    return crop_blur_regions(
        src, dst, [{"x": x_frac, "y": y_frac, "w": w_frac, "h": h_frac}], blur_strength
    )


def ffprobe_fps(path):
    """Average frame rate as a float. `avg_frame_rate` rather than `r_frame_rate` because the
    clips here are re-encoded CFR and avg_frame_rate is what actually describes the frames on
    disk; r_frame_rate can report a container-level base rate that is a multiple of it."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,nb_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    num, _, den = stream["avg_frame_rate"].partition("/")
    return float(num) / float(den or 1)


# --- Visual-null / degraded-input variants -------------------------------------------------
# Used by 19_run_visual_null_baselines.py. Each takes a real model-input clip and produces one
# with a specific kind of visual information removed, holding EVERYTHING else -- duration, frame
# rate, resolution, codec -- identical, so the frame count and per-frame timestamps the VLM
# processor derives are unchanged and the only difference reaching the model is pixel content.

def make_black_clip(src, dst):
    """All-black video, same dimensions/duration/frame rate as src. Removes all visual
    information while keeping the video-shaped input the prompt refers to."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    w, h = ffprobe_dimensions(src)
    duration, fps = ffprobe_duration(src), ffprobe_fps(src)
    cmd = ["ffmpeg", "-y", "-f", "lavfi",
           "-i", f"color=c=black:s={w}x{h}:r={fps}:d={duration}",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(dst)


def make_static_clip(src, dst):
    """The middle frame of src, held for src's full duration at src's frame rate.

    Removes motion while keeping the scene, the signer, and their static posture. The
    interesting contrast with `black`: a form description of a *movement* cannot be grounded in
    a single frame, so anything the model still reports about movement path or manner under this
    condition is not read off the video.
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    duration, fps = ffprobe_duration(src), ffprobe_fps(src)
    cmd = ["ffmpeg", "-y", "-ss", f"{duration / 2:.3f}", "-i", str(src), "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "png", "-"]
    frame = subprocess.run(cmd, check=True, capture_output=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        frame_path = Path(tmp) / "mid.png"
        frame_path.write_bytes(frame)
        subprocess.run(
            ["ffmpeg", "-y", "-loop", "1", "-framerate", str(fps), "-i", str(frame_path),
             "-t", f"{duration:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "18", "-pix_fmt", "yuv420p", str(dst)],
            check=True, capture_output=True, text=True)
    return Path(dst)


def make_shuffled_clip(src, dst, seed=20260803):
    """src's own frames in a fixed pseudo-random order, same count and frame rate.

    This is the sharpest of the three: every frame the model would have seen is still present,
    so appearance, handshapes and articulator positions are all intact -- only their temporal
    ORDER is destroyed. A form description that survives shuffling intact was not built from the
    movement trajectory.
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    fps = ffprobe_fps(src)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vsync", "0",
                        str(tmp / "in_%05d.png")], check=True, capture_output=True, text=True)
        frames = sorted(tmp.glob("in_*.png"))
        if not frames:
            raise RuntimeError(f"no frames decoded from {src}")
        order = list(range(len(frames)))
        random.Random(seed).shuffle(order)
        out_dir = tmp / "out"
        out_dir.mkdir()
        for new_index, old_index in enumerate(order, start=1):
            (out_dir / f"out_{new_index:05d}.png").write_bytes(frames[old_index].read_bytes())
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(out_dir / "out_%05d.png"),
             "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-pix_fmt", "yuv420p", str(dst)],
            check=True, capture_output=True, text=True)
    return Path(dst)


def extract_frames(src, out_dir, fps=1.0, prefix="frame"):
    """Sample frames at `fps` frames/sec for manual QC or OCR-based leakage screening."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{prefix}_%04d.jpg")
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", f"fps={fps}", pattern]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sorted(out_dir.glob(f"{prefix}_*.jpg"))
