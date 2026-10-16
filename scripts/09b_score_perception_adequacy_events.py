#!/usr/bin/env python3
"""Tier 2 (Perception) for the C01-C40 clip set: human rating of the model's form description.

No match-filtering step is needed here: prompt_c_output.schema.json enforces exactly one candidate_action per run, already
targeting the span 07b put in the prompt, so every C01-C40 run has exactly one form_description to
rate. `build` therefore reads every run directly (same `clip_id`-key discrimination
08b_score_localization_events.py uses to tell C01-C40 outputs apart from the older workflow's, since
both live in the same model_outputs directory).

The sheet also carries two channel-reference columns per row (2026-07-29, follow-up), pulled in
alongside the seven model_* form_description fields: `target_channel` (from clip_manifest.csv,
via lib.common.parse_manifest_event -- what articulator the manifest actually specifies for this
event) and `channel_judged` (the model's own self-reported articulator, already present in its
JSON output). Neither existed on the sheet before, so a rater had no way to tell, without leaving
the CSV, which of the seven form_description fields should carry the real content for a given row
-- e.g. under target_channel=face the substantive content lives in `nonmanual`, not `hands`/
`handshape`, and a rater treating hands/handshape as blank-therefore-wrong would be scoring against
the wrong articulator entirely. See perception_rating_guide.md for how raters are told to use
these two columns (channel-conditional "core fields", and what a target_channel/channel_judged
mismatch means under CHANNEL_GIVEN vs CHANNEL_WITHHELD modes). This is a per-row human-facing
convenience only -- it is not an automated compliance metric; no C01-C40-track equivalent of
17_score_cue_ablation.py's automated channel_judged-vs-target_channel check exists yet.

    0 = form description mostly wrong or irrelevant
    1 = partially adequate; captures some visible aspects
    2 = adequate enough to support linguistic-status attribution

--mode selects which 07b_run_inference_events.py output set to build/aggregate against.
--clip-kind restricts `build` to one clip kind (e.g. `combined`) instead of rating every kind
found -- useful when only one condition's form descriptions need human rating, since rating
burden scales with row count (all four kinds is ~3x the rows of one kind alone).

Usage:
    python scripts/09b_score_perception_adequacy_events.py build --mode grounded
    python scripts/09b_score_perception_adequacy_events.py build --mode grounded --clip-kind combined
    python scripts/09b_score_perception_adequacy_events.py aggregate --mode grounded --filled-dir data/reference_annotations/perception_filled_events
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir, REPO_ROOT, parse_manifest_event

log = get_logger("score_perception_adequacy_events")

MODES = ["grounded", "ablate_channel", "ablate_translation", "ablate_both"]
DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"

RATING_COLUMNS = ["clip_id", "event_id", "clip_kind", "run_index",
                  "target_channel", "channel_judged",
                  "model_hands", "model_handshape", "model_movement", "model_location",
                  "model_nonmanual", "model_non_dominant_hand", "model_temporal_alignment",
                  "rater_id", "perception_score", "notes"]


def load_target_channels(manifest_rows):
    """(clip_id, event_id) -> target_channel, straight from clip_manifest.csv (same source 08b
    uses for its reference spans -- see lib.common.parse_manifest_event)."""
    out = {}
    for row in manifest_rows:
        for index, event_id in ((1, "event1"), (2, "event2")):
            parsed = parse_manifest_event(row, index)
            if parsed is not None:
                out[(row["clip_id"], event_id)] = parsed[2]
    return out


def load_event_model_runs(model_outputs_dir, mode):
    """Every C01-C40-format run for `mode`, told apart from the older window_id-keyed outputs
    (that share the same directory) by the presence of a `clip_id` key -- same test
    08b_score_localization_events.py uses."""
    runs = []
    for path in sorted(model_outputs_dir.glob(f"*__run*__{mode}.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "clip_id" in data and data.get("parsed"):
            runs.append(data)
    return runs


def build(cfg, mode, clip_kind=None, manifest_path=DEFAULT_MANIFEST):
    model_outputs_dir = cfg["paths"]["model_outputs_dir"]
    runs = load_event_model_runs(model_outputs_dir, mode)
    if not runs:
        log.warning("No C01-C40-format runs found for --mode %s in %s. "
                    "Run 07b_run_inference_events.py --mode %s first.",
                    mode, model_outputs_dir, mode)
    if clip_kind:
        runs = [r for r in runs if r.get("clip_kind") == clip_kind]

    with open(manifest_path, "r", encoding="utf-8") as f:
        target_channels = load_target_channels(list(csv.DictReader(f)))

    out_rows = []
    for run in runs:
        candidates = run["parsed"].get("candidate_actions") or []
        if not candidates:
            continue
        candidate = candidates[0]  # schema enforces exactly one candidate
        fd = candidate.get("form_description", {})
        out_rows.append({
            "clip_id": run["clip_id"], "event_id": run["event_id"], "clip_kind": run["clip_kind"],
            "run_index": run["run_index"],
            "target_channel": target_channels.get((run["clip_id"], run["event_id"]), ""),
            "channel_judged": candidate.get("channel_judged", ""),
            "model_hands": fd.get("hands", ""), "model_handshape": fd.get("handshape", ""),
            "model_movement": fd.get("movement", ""), "model_location": fd.get("location", ""),
            "model_nonmanual": fd.get("nonmanual", ""), "model_non_dominant_hand": fd.get("non_dominant_hand", ""),
            "model_temporal_alignment": fd.get("temporal_alignment", ""),
            "rater_id": "", "perception_score": "", "notes": "",
        })

    ensure_dir(cfg["paths"]["results_dir"])
    suffix = f"_{clip_kind}" if clip_kind else ""
    out_path = cfg["paths"]["results_dir"] / f"perception_rating_sheet_events_blank{suffix}_{mode}.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RATING_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)
    log.info("Wrote %d row(s) to rate -> %s", len(out_rows), out_path)


def aggregate(cfg, filled_dir, mode, clip_kind=None):
    rows = []
    for csv_path in sorted(Path(filled_dir).glob("*.csv")):
        with open(csv_path, "r", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))

    suffix = f"_{clip_kind}" if clip_kind else ""
    out_path = cfg["paths"]["results_dir"] / f"perception_scores_events{suffix}_{mode}.csv"
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RATING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    scored = [float(r["perception_score"]) for r in rows if r.get("perception_score", "").strip() != ""]
    if scored:
        adequate_rate = sum(1 for s in scored if s >= cfg["attribution_scoring"]["perception_adequate_min_score"]) / len(scored)
        log.info("Perception adequacy rate (score >= %d): %.3f (n=%d)",
                  cfg["attribution_scoring"]["perception_adequate_min_score"], adequate_rate, len(scored))
    log.info("Wrote %s", out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build_p = sub.add_parser("build")
    build_p.add_argument("--config", default=None)
    build_p.add_argument("--mode", required=True, choices=MODES)
    build_p.add_argument("--clip-kind", default=None, choices=["window", "ev1", "ev2", "combined"],
                         help="restrict to one clip kind instead of rating every kind found")
    build_p.add_argument("--manifest", default=None, help=f"default: {DEFAULT_MANIFEST.name}")
    agg_p = sub.add_parser("aggregate")
    agg_p.add_argument("--config", default=None)
    agg_p.add_argument("--mode", required=True, choices=MODES)
    agg_p.add_argument("--filled-dir", required=True)
    agg_p.add_argument("--clip-kind", default=None, choices=["window", "ev1", "ev2", "combined"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.cmd == "build":
        manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
        build(cfg, args.mode, args.clip_kind, manifest_path)
    else:
        aggregate(cfg, args.filled_dir, args.mode, args.clip_kind)


if __name__ == "__main__":
    main()
