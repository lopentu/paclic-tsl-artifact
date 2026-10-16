#!/usr/bin/env python3
"""Blur burned-in subtitles (and other identifying overlays) to produce the model-input
copy of each clip.

the project design log (subtitle leakage control): the model must not be able to read
Chinese captions burned into the video, or the experiment stops testing visual
interpretation of TSL. The annotator copy is always left untouched; only the model-input
copy is blurred.

Two sources, selected with --source (default windows_manifest, the original pipeline):

    windows_manifest (default)  data/windows/manifest.csv -> data/windows/clips/
        Reads `<window_id>_annot.mp4` (03_cut_analysis_windows.py), writes `<window_id>_model.mp4`.

    clip_manifest               clip_manifest.csv -> data/clips_c/
        Reads whichever of `<clip_id>_window.mp4` / `_ev1.mp4` / `_ev2.mp4` exist
        (03b_cut_event_clips.py), writes the matching `..._model.mp4` for each.

Per-video crop regions come from config/subtitle_crop_regions.yaml if present (copy
config/subtitle_crop_regions.example.yaml and fill it in after inspecting sample frames --
see 05_check_caption_leakage.py --inspect), otherwise the default band in
pipeline_config.yaml (subtitle_removal.default_region) is used for every video. Both
sources share this same per-video lookup (keyed by video_id).

`subtitle_removal.always_blur_regions` (2026-07-27) is blurred on EVERY video in addition to
whichever caption region applies -- currently the source channel's association logo (top-right)
and the show's own title card (top-left). Unlike the caption band, these sit in the same fixed
screen position across the whole corpus, so they are additive rather than per-video overrides.
Motivation: a recognizable channel/show identifier is a stronger memorization cue than the
prompt's plain "this is TSL" statement -- if this exact channel's videos were in the model's
pretraining data, the logo could trigger recall of memorized captions/summaries of *this specific
content*, which is closer to test contamination than priming. See
the project design log for the frame-by-frame verification that neither region overlaps the signer.

Usage:
    python scripts/04_deidentify_subtitles.py
    python scripts/04_deidentify_subtitles.py --source clip_manifest
    python scripts/04_deidentify_subtitles.py --source clip_manifest --clip-id C07 --overwrite
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, REPO_ROOT
from lib.video_io import crop_blur_regions

log = get_logger("deidentify_subtitles")

DEFAULT_CLIP_MANIFEST = REPO_ROOT / "clip_manifest.csv"
CLIP_KINDS = ["window", "ev1", "ev2"]


def load_regions(scripts_config_dir):
    regions_path = scripts_config_dir / "subtitle_crop_regions.yaml"
    if regions_path.exists():
        with open(regions_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def regions_for_video(video_id, per_video_regions, default_region, always_regions):
    region_entry = per_video_regions.get(video_id, default_region)
    regions = region_entry if isinstance(region_entry, list) else [region_entry]
    return regions + always_regions


def process_windows_manifest(args, cfg, per_video_regions, default_region, always_regions, blur_strength):
    manifest_path = Path(args.manifest) if args.manifest else cfg["paths"]["windows_manifest"]
    clips_dir = cfg["paths"]["clips_dir"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_done = 0
    for row in rows:
        if args.window_id and row["window_id"] != args.window_id:
            continue

        src = clips_dir / f"{row['window_id']}_annot.mp4"
        if not src.exists():
            log.warning("Missing cut clip for %s: %s (run 03_cut_analysis_windows.py first)",
                        row["window_id"], src)
            continue

        dst = clips_dir / f"{row['window_id']}_model.mp4"
        if dst.exists() and not args.overwrite:
            log.info("Skip (exists): %s", dst.name)
            continue

        regions = regions_for_video(row["video_id"], per_video_regions, default_region, always_regions)
        log.info("Blurring %d region(s) for %s: %s", len(regions), row["window_id"], regions)
        crop_blur_regions(src, dst, regions, blur_strength)
        n_done += 1

    log.info("De-identified %d clip(s). Next: python scripts/05_check_caption_leakage.py", n_done)


def process_clip_manifest(args, cfg, per_video_regions, default_region, always_regions, blur_strength):
    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_CLIP_MANIFEST
    c_clips_dir = cfg["paths"]["c_clips_dir"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_done, n_missing_src = 0, 0
    for row in rows:
        clip_id = (row.get("clip_id") or "").strip()
        if args.clip_id and clip_id != args.clip_id:
            continue
        if (row.get("status") or "").strip().lower() == "excluded":
            continue
        video_id = (row.get("video_id") or "").strip()
        if not video_id:
            continue  # manifest row not filled in yet -- nothing to blur

        regions = regions_for_video(video_id, per_video_regions, default_region, always_regions)
        for kind in CLIP_KINDS:
            src = c_clips_dir / f"{clip_id}_{kind}.mp4"
            if not src.exists():
                continue  # normal: no ev2 for this clip, or 03b hasn't been run for it yet
            dst = c_clips_dir / f"{clip_id}_{kind}_model.mp4"
            if dst.exists() and not args.overwrite:
                log.info("Skip (exists): %s", dst.name)
                continue
            log.info("Blurring %d region(s) for %s: %s", len(regions), dst.stem, regions)
            crop_blur_regions(src, dst, regions, blur_strength)
            n_done += 1

    log.info("De-identified %d clip(s) under %s. Next: python scripts/07b_run_inference_events.py",
              n_done, c_clips_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", choices=["windows_manifest", "clip_manifest"],
                        default="windows_manifest",
                        help="windows_manifest (default): the original per-window pipeline. "
                             "clip_manifest: the C01-C40 set (clip_manifest.csv -> data/clips_c/).")
    parser.add_argument("--manifest", default=None, help="override the default manifest path for --source")
    parser.add_argument("--window-id", default=None, help="windows_manifest only: process only this window_id")
    parser.add_argument("--clip-id", default=None, help="clip_manifest only: process only this clip_id")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    default_region = cfg["subtitle_removal"]["default_region"]
    blur_strength = cfg["subtitle_removal"]["blur_strength"]
    always_regions = cfg["subtitle_removal"].get("always_blur_regions") or []

    scripts_dir = Path(__file__).resolve().parent
    per_video_regions = load_regions(scripts_dir / "config")

    if args.source == "clip_manifest":
        process_clip_manifest(args, cfg, per_video_regions, default_region, always_regions, blur_strength)
    else:
        process_windows_manifest(args, cfg, per_video_regions, default_region, always_regions, blur_strength)


if __name__ == "__main__":
    main()
