#!/usr/bin/env python3
"""Tier 1 (Localization) for the C01-C40 clip set: did the model anchor on the given span?

For the clip_manifest.csv / 07b_run_inference_events.py workflow. The reference span needs no human annotation here -- clip_manifest.csv's own
event{1,2}_start_sec/end_sec IS the target span (the researcher fixed it when the clips were cut),
so this script reads only clip_manifest.csv and the model outputs, converting the reference into
the same clip-relative units the model's self-reported start_time/end_time is in via
lib.common.kind_clip_bounds -- the identical arithmetic 07b used to build the prompt's timestamp
hint, so the two can never silently disagree about where the target actually is in a given clip.

Every Prompt-C run has exactly one candidate_action (the schema enforces this), so there is no
many-to-many matching to do here -- just one reference span vs one reported span, per
(clip_id, event_id, clip_kind, run_index).

READ THE PER-KIND BREAKDOWN SEPARATELY -- do not average window and event-clip recall together:

  - On a "window" clip, the target sits inside 8-12s of surrounding signing with nothing marking
    it out visually. A high match rate here is genuine evidence the model can anchor a given span
    without help.
  - On an "ev1"/"ev2" clip, the padding already puts the target near the middle of a ~1-2s clip.
    A high match rate here is a much weaker signal (partly built into the framing) -- but a LOW
    match rate is still diagnostically important: it means the model drifted even when the
    target was almost handed to it, which is exactly the "locked onto a neighboring sign" failure
    mode the small pad was chosen to avoid. Watch this rate when
    tuning --pad-sec: a rising miss rate on event clips as pad shrinks means the pad is now too
    tight for the model's own temporal precision, not that the mechanism is broken.
  - On a "combined" run (2026-07-29), the model is shown both the window (as context) and the
    ev1/ev2 clip (as the video it's actually scoring) in one call, so its reference span is
    identical to the plain ev1/ev2 clip's -- same weak-signal caveat applies, and it is its own
    condition besides: never pool "combined" with "window" or with the solo "ev1"/"ev2" runs,
    since a single call now saw more visual information than either of those alone.

Outputs results_dir/localization_matches_events_<mode>.csv (one row per clip_id/event_id/clip_kind/
run) and prints recall broken out by clip_kind.

Usage:
    python scripts/08b_score_localization_events.py --mode grounded
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import (load_config, get_logger, ensure_dir, REPO_ROOT, parse_timestamp,
                        parse_manifest_event, kind_clip_bounds)
from lib.stats_utils import interval_overlap_ratio

log = get_logger("score_localization_events")

# grounded_distractor is not produced by 07b_run_inference_events.py (no distractor_* columns
# in clip_manifest.csv yet); ablate_* modes still have a target span (see that script's docstring).
MODES = ["grounded", "ablate_channel", "ablate_translation", "ablate_both"]
CLIP_KINDS = ["window", "ev1", "ev2", "combined"]
KINDS_FOR_EVENT = {"event1": ["window", "ev1", "combined"], "event2": ["window", "ev2", "combined"]}
# "combined" sends the ev1/ev2 file as the model's actual video_path (window is only
# context_video_path), so its reference span is computed the same way ev1/ev2's is --
# kind_clip_bounds already does this for free (it only special-cases the literal "window").

DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"


def load_reference_spans(manifest_rows, pad_sec):
    """(clip_id, event_id, clip_kind) -> (ref_onset, ref_offset) in that clip's own relative
    seconds, computed straight from clip_manifest.csv -- no human annotation needed (see module
    docstring)."""
    spans = {}
    for row in manifest_rows:
        clip_id = (row.get("clip_id") or "").strip()
        if (row.get("status") or "").strip().lower() == "excluded":
            continue
        clip_start_raw, clip_end_raw = (row.get("clip_start_sec") or "").strip(), (row.get("clip_end_sec") or "").strip()
        if not clip_start_raw or not clip_end_raw:
            continue

        clip_start_sec, clip_end_sec = float(clip_start_raw), float(clip_end_raw)
        for index in (1, 2):
            try:
                ev = parse_manifest_event(row, index)
            except ValueError:
                continue  # malformed row -- 03b/07b already warn about this; skip quietly here
            if ev is None:
                continue
            ev_start, ev_end, _channel = ev
            event_id = f"event{index}"
            for kind in KINDS_FOR_EVENT[event_id]:
                kind_start, _kind_end = kind_clip_bounds(kind, clip_start_sec, clip_end_sec,
                                                          ev_start, ev_end, pad_sec)
                spans[(clip_id, event_id, kind)] = (ev_start - kind_start, ev_end - kind_start)
    return spans


def load_event_model_runs(model_outputs_dir, mode):
    """Every C01-C40-format run for `mode`. Distinguished from the older windows_manifest-format
    outputs (which share the same directory) by the presence of a `clip_id` key rather than
    `window_id` -- more robust than pattern-matching filenames that only differ by underscore
    count."""
    runs = []
    for path in sorted(model_outputs_dir.glob(f"*__run*__{mode}.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "clip_id" in data and data.get("parsed"):
            runs.append(data)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None, help=f"default: {DEFAULT_MANIFEST.name}")
    parser.add_argument("--mode", required=True, choices=MODES,
                        help="which 07b_run_inference_events.py --mode's outputs to score")
    parser.add_argument("--pad-sec", type=float, default=0.3,
                        help="MUST match the --pad-sec used for both 03b_cut_event_clips.py and "
                             "07b_run_inference_events.py for these clips (default: 0.3)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
    model_outputs_dir = cfg["paths"]["model_outputs_dir"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    tolerance = cfg["localization_scoring"]["onset_offset_tolerance_sec"]
    min_overlap = cfg["localization_scoring"]["min_overlap_ratio"]

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    reference = load_reference_spans(manifest_rows, args.pad_sec)

    out_rows = []
    missing_reference = 0
    for run in load_event_model_runs(model_outputs_dir, args.mode):
        key = (run["clip_id"], run["event_id"], run["clip_kind"])
        ref = reference.get(key)
        if ref is None:
            missing_reference += 1
            continue
        ref_onset, ref_offset = ref

        candidates = run["parsed"].get("candidate_actions") or []
        if not candidates:
            out_rows.append({
                "clip_id": run["clip_id"], "event_id": run["event_id"], "clip_kind": run["clip_kind"],
                "run_index": run["run_index"], "matched": False, "iou": "",
                "onset_error_sec": "", "offset_error_sec": "",
                "ref_onset": round(ref_onset, 2), "ref_offset": round(ref_offset, 2),
                "model_onset": "", "model_offset": "",
            })
            continue

        c = candidates[0]  # schema enforces exactly one
        try:
            m_start, m_end = parse_timestamp(c["start_time"]), parse_timestamp(c["end_time"])
        except (ValueError, KeyError):
            m_start = m_end = None

        if m_start is None:
            matched, iou, onset_err, offset_err = False, "", "", ""
        else:
            iou = interval_overlap_ratio(ref_onset, ref_offset, m_start, m_end)
            onset_err = abs(ref_onset - m_start)
            offset_err = abs(ref_offset - m_end)
            matched = iou >= min_overlap or (onset_err <= tolerance and offset_err <= tolerance)

        out_rows.append({
            "clip_id": run["clip_id"], "event_id": run["event_id"], "clip_kind": run["clip_kind"],
            "run_index": run["run_index"], "matched": matched,
            "iou": round(iou, 3) if iou != "" else "",
            "onset_error_sec": round(onset_err, 2) if onset_err != "" else "",
            "offset_error_sec": round(offset_err, 2) if offset_err != "" else "",
            "ref_onset": round(ref_onset, 2), "ref_offset": round(ref_offset, 2),
            "model_onset": round(m_start, 2) if m_start is not None else "",
            "model_offset": round(m_end, 2) if m_start is not None else "",
        })

    out_path = results_dir / f"localization_matches_events_{args.mode}.csv"
    fieldnames = ["clip_id", "event_id", "clip_kind", "run_index", "matched", "iou",
                  "onset_error_sec", "offset_error_sec", "ref_onset", "ref_offset",
                  "model_onset", "model_offset"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    if not out_rows:
        log.warning("No C01-C40-format runs found for --mode %s in %s (missing_reference=%d). "
                    "Run 07b_run_inference_events.py --mode %s first.",
                    args.mode, model_outputs_dir, missing_reference, args.mode)
        log.info("Wrote empty %s", out_path)
        return

    by_kind = {}
    for r in out_rows:
        by_kind.setdefault(r["clip_kind"], []).append(r)

    log.info("Localization recall by clip_kind (mode=%s):", args.mode)
    for kind in CLIP_KINDS:
        rows = by_kind.get(kind)
        if not rows:
            continue
        recall = sum(1 for r in rows if r["matched"]) / len(rows)
        if kind == "window":
            label = "window (real localization test)"
        elif kind == "combined":
            label = "combined (weak signal, own condition -- not comparable to window or solo ev1/ev2)"
        else:
            label = f"{kind} (weak signal -- see docstring)"
        log.info("  %-9s n=%-4d recall=%.3f  [%s]", kind, len(rows), recall, label)
    if missing_reference:
        log.warning("%d run(s) had no matching clip_manifest.csv row for their (clip_id, event_id, "
                    "clip_kind) -- likely the manifest changed after these were run.", missing_reference)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
