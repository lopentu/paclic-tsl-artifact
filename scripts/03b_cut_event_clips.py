#!/usr/bin/env python3
"""Cut the C01-C40 set from `clip_manifest.csv`: one analysis-window clip plus one tight clip per event.

Per manifest row this writes, under paths.c_clips_dir:

    <clip_id>_window.mp4   [clip_start_sec, clip_end_sec]                 -- the 8-12s analysis window
    <clip_id>_ev1.mp4      [event1_start - pad, event1_end + pad]         -- tight cut around event 1
    <clip_id>_ev2.mp4      [event2_start - pad, event2_end + pad]         -- only if event 2 exists

Both are needed because the target actions are often under a second. The window clip preserves the
temporal scope D1 partly rests on (does the action span a clause? does a hold outlast the sign it
accompanies?), which a tight cut destroys; the tight clip makes "which action" unambiguous, which a
10-second window at fps=8 does not. They are separate model conditions, never pooled. Marking the target
inside the window instead (boxes/arrows, looping, freeze-frames) would leak the form we are
asking the model to describe.

The padding is FIXED and symmetric (--pad-sec, default 0.3) precisely so it carries no information:
it cannot correlate with channel or function, and it never reveals which articulator moves, so the
ablate_channel / ablate_both cells stay valid on event clips. Kept deliberately small (0.3s/side,
0.6s total) rather than the original 0.75s: TSL signs pass quickly enough that a larger pad risks
pulling in 1-2 whole neighboring signs, which is a worse failure mode than a slightly tighter view
of the target. This assumes reasonably precise event
timestamps; if your source timestamps are only whole-second precision,
raise --pad-sec for that batch rather than relying on this default to absorb the slack.

Event 2 is optional. A clip with one target action has `N/A` in all three event2_* columns and simply
gets no _ev2 clip. A BLANK event2 cell is treated as "not decided yet" and reported as a gap, so a
half-filled manifest cannot silently produce a short dataset.

Cuts are clamped to the source video's duration, and event cuts are NOT clamped to the clip span --
the pad may legitimately extend past the window edge, and the event clip is its own artifact.

These are un-deidentified cuts (the annotator-facing copies, like 03's `_annot.mp4`). Run
04_deidentify_subtitles.py to produce the subtitle-blurred `_model.mp4` copies before any model run.

Usage:
    python scripts/03b_cut_event_clips.py                        # everything not yet cut
    python scripts/03b_cut_event_clips.py --clip-id C07 --overwrite
    python scripts/03b_cut_event_clips.py --pad-sec 1.0
    python scripts/03b_cut_event_clips.py --events-only --slow-factor 0.5   # extra slowed condition
    python scripts/03b_cut_event_clips.py --dry-run              # print planned cuts, touch nothing
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import (load_config, get_logger, ensure_dir, REPO_ROOT, padded_event_bounds,
                        parse_manifest_event, is_na)
from lib.video_io import cut_clip, ffprobe_duration

log = get_logger("cut_event_clips")

DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"


def retime(src, dst, factor):
    """Re-encode `src` at `factor`x speed (0.5 = half speed) by rescaling presentation timestamps."""
    cmd = ["ffmpeg", "-y", "-i", str(src), "-filter:v", f"setpts={1.0 / factor}*PTS",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None, help=f"default: {DEFAULT_MANIFEST.name}")
    parser.add_argument("--clip-id", default=None, help="cut only this clip_id (default: all)")
    parser.add_argument("--pad-sec", type=float, default=0.3,
                        help="symmetric padding around each event span (default: 0.3). Keep this "
                             "the same for every item in a run -- a per-item pad would leak "
                             "information about the target. Raise it if your event timestamps are "
                             "only whole-second precision; lower it further only if 0.3s/side is "
                             "still catching neighboring signs in fast passages.")
    parser.add_argument("--slow-factor", type=float, default=1.0,
                        help="re-time event clips to this speed (e.g. 0.5 = half speed) to get more "
                             "frames on a sub-second action. Output is suffixed _slowNNN and is a "
                             "SEPARATE condition: movement speed/manner is part of the form "
                             "description and feeds D2, so slowed runs are not comparable to "
                             "normal-speed runs. Default 1.0 = off.")
    parser.add_argument("--events-only", action="store_true", help="skip the window clips")
    parser.add_argument("--windows-only", action="store_true", help="skip the event clips")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.pad_sec < 0:
        parser.error("--pad-sec must be >= 0")
    if not 0 < args.slow_factor <= 1.0:
        parser.error("--slow-factor must be in (0, 1]")

    cfg = load_config(args.config)
    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
    raw_dir = cfg["paths"]["raw_videos_dir"]
    out_dir = cfg["paths"].get("c_clips_dir") or (cfg["paths"]["clips_dir"].parent / "clips_c")
    slow_suffix = "" if args.slow_factor == 1.0 else f"_slow{int(round(args.slow_factor * 100)):03d}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    planned, problems, skipped = [], [], []
    for row in rows:
        clip_id = (row.get("clip_id") or "").strip()
        if args.clip_id and clip_id != args.clip_id:
            continue
        if (row.get("status") or "").strip().lower() == "excluded":
            skipped.append(f"{clip_id}: status=excluded")
            continue

        video_id = (row.get("video_id") or "").strip()
        clip_start, clip_end = (row.get("clip_start_sec") or "").strip(), (row.get("clip_end_sec") or "").strip()
        if not video_id or not clip_start or not clip_end:
            problems.append(f"{clip_id}: needs video_id + clip_start_sec + clip_end_sec (not filled in yet)")
            continue

        src = raw_dir / f"{video_id}.mp4"
        if not src.exists():
            problems.append(f"{clip_id}: source video not found: {src} (run 01_fetch_source_videos.py)")
            continue

        clip_start, clip_end = float(clip_start), float(clip_end)
        if clip_end <= clip_start:
            problems.append(f"{clip_id}: clip_end_sec ({clip_end}) is not after clip_start_sec ({clip_start})")
            continue

        try:
            events = {f"ev{i}": parse_manifest_event(row, i) for i in (1, 2)}
        except ValueError as exc:
            problems.append(f"{clip_id}: {exc}")
            continue
        if events["ev1"] is None:
            problems.append(f"{clip_id}: event1 is required but marked N/A")
            continue

        duration = ffprobe_duration(src)
        if not args.events_only:
            planned.append({"clip_id": clip_id, "src": src, "label": "window", "retime": False,
                            "start": max(0.0, clip_start), "end": min(clip_end, duration)})
        if not args.windows_only:
            for label, ev in events.items():
                if ev is None:
                    continue
                ev_start, ev_end, _channel = ev
                if not (clip_start <= ev_start and ev_end <= clip_end):
                    problems.append(f"{clip_id}: {label} span [{ev_start}, {ev_end}] falls outside the "
                                    f"clip span [{clip_start}, {clip_end}]")
                    continue
                ev_clip_start, ev_clip_end = padded_event_bounds(ev_start, ev_end, args.pad_sec, duration)
                planned.append({"clip_id": clip_id, "src": src, "label": label, "retime": True,
                                "start": ev_clip_start, "end": ev_clip_end})

    for msg in problems:
        log.warning(msg)
    for msg in skipped:
        log.info(msg)

    if args.dry_run:
        for job in planned:
            suffix = slow_suffix if job["retime"] else ""
            print(f"[dry-run] {job['clip_id']}_{job['label']}{suffix}.mp4  "
                  f"<- {job['src'].name} [{job['start']:.2f}, {job['end']:.2f}]  "
                  f"({job['end'] - job['start']:.2f}s)")
        log.info("%d cut(s) planned, %d row(s) with problems, %d skipped.",
                 len(planned), len(problems), len(skipped))
        return

    ensure_dir(out_dir)
    n_done = 0
    for job in planned:
        suffix = slow_suffix if job["retime"] else ""
        dst = out_dir / f"{job['clip_id']}_{job['label']}{suffix}.mp4"
        if dst.exists() and not args.overwrite:
            log.info("Skip (exists): %s", dst.name)
            continue

        log.info("Cutting %s [%.2f, %.2f] -> %s", job["src"].name, job["start"], job["end"], dst.name)
        if suffix:
            tmp = out_dir / f".tmp_{dst.name}"
            cut_clip(job["src"], job["start"], job["end"], tmp, keep_audio=False)
            retime(tmp, dst, args.slow_factor)
            tmp.unlink()
        else:
            cut_clip(job["src"], job["start"], job["end"], dst)
        n_done += 1

    log.info("Cut %d clip(s) into %s", n_done, out_dir)
    if problems:
        log.warning("%d manifest row(s) were skipped -- see the warnings above.", len(problems))
    if slow_suffix:
        log.warning("Slowed event clips written (%s). Report these as their own condition: movement "
                    "speed/manner is part of the form description and feeds D2, so they are not "
                    "comparable to normal-speed runs.", slow_suffix)
    log.info("Next: 04_deidentify_subtitles.py to make the subtitle-blurred _model.mp4 copies.")


if __name__ == "__main__":
    main()
