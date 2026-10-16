#!/usr/bin/env python3
"""Tier 3 (Attribution) for the C01-C40 clip set: compare model D1/D2 and semiotic-mode ratings
against the human reference (mean across annotators), per clip kind.

Keys by (clip_id, event_id[, clip_kind]) and reads 08b's/09b's outputs.

Gates:
  1. Localization: only rows 08b marked `matched` are compared -- if the model's self-reported
     span drifted from the given target, its D1/D2/mode scores may be about the wrong action, so
     comparing them would be invalid regardless of clip kind. Always applied (no bypass flag,
     matching 10's behaviour).
  2. Perception adequacy: only events with a perception_score >= attribution_scoring.
     perception_adequate_min_score (mean across raters, from 09b's aggregated sheet) are compared,
     so perception failure and attribution failure are never conflated. Pass
     --ignore-perception-gate for an EXPLORATORY run before Tier 2 rating is complete (e.g. this
     round: inference just finished, no perception_score has been filled in yet) -- writes to a
     separately-named `_ungated` file so the gated primary result is never silently replaced.

Reports MAE + Spearman for D1/D2 and MAE for description/indication/depiction, broken out PER
CLIP KIND in one CSV (never pooled across `window`/`ev1`/`ev2`/`combined` -- same principle as
08b: the conditions are not comparable to each other).

--mode selects which 07b_run_inference_events.py output set to score (must match 08b/09b).
--clip-kind restricts to one clip kind (e.g. `combined`, matching what 09b was built for) instead
of reporting all four.

Usage:
    python scripts/10b_score_attribution_events.py --mode grounded --ignore-perception-gate
    python scripts/10b_score_attribution_events.py --mode grounded --clip-kind combined
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir
from lib.stats_utils import mean_absolute_error, spearman_correlation

log = get_logger("score_attribution_events")

MODES = ["grounded", "ablate_channel", "ablate_translation", "ablate_both"]
SEMIOTIC_MODES = ["description", "indication", "depiction"]
CLIP_KINDS = ["window", "ev1", "ev2", "combined"]

HUMAN_FIELDS = {"d1": "semantic_syntactic_load", "d2": "pragmatic_depictive_affective_enrichment",
                "description": "description", "indication": "indication", "depiction": "depiction"}


def load_human_reference(events_csv):
    """(clip_id, event_id) -> dict with mean D1, D2, and mean of each semiotic mode across
    annotators."""
    by_item = defaultdict(lambda: {k: [] for k in HUMAN_FIELDS})
    with open(events_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["clip_id"], row["event_id"])
            for key_name, col in HUMAN_FIELDS.items():
                val = (row.get(col) or "").strip()
                if val != "":
                    by_item[key][key_name].append(float(val))
    out = {}
    for key, v in by_item.items():
        out[key] = {k: (statistics.mean(vals) if vals else None) for k, vals in v.items()}
    return out


def load_perception_scores(perception_csv):
    """(clip_id, event_id, clip_kind) -> mean perception_score across raters."""
    if not Path(perception_csv).exists():
        return {}
    by_item = defaultdict(list)
    with open(perception_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("perception_score") or "").strip() == "":
                continue
            key = (row["clip_id"], row["event_id"], row["clip_kind"])
            by_item[key].append(float(row["perception_score"]))
    return {k: statistics.mean(v) for k, v in by_item.items()}


def load_matches(matches_csv):
    with open(matches_csv, "r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["matched"] == "True"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--events-csv", default=None,
                        help="reference annotation to score against (default: "
                             "paths.human_annotations_csv). Use this to score against the "
                             "post-discussion reference (events_checked.csv) without editing the "
                             "config, so the independent-pass numbers stay reproducible.")
    parser.add_argument("--out-suffix", default="",
                        help="appended to the output filename, so scoring against a different "
                             "reference cannot silently overwrite the previous result")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--clip-kind", default=None, choices=CLIP_KINDS,
                         help="restrict to one clip kind instead of reporting all four")
    parser.add_argument("--ignore-perception-gate", action="store_true",
                         help="EXPLORATORY ONLY: skip the perception-adequacy gate (e.g. before "
                              "Tier 2 rating is complete). Writes to a separately-named "
                              "attribution_scores_events_<mode>_ungated.csv.")
    args = parser.parse_args()
    if args.ignore_perception_gate:
        log.warning("--ignore-perception-gate set: this run does NOT reflect the perception-"
                     "adequate gate. Treat as exploratory only.")

    cfg = load_config(args.config)
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    thresholds = cfg["attribution_scoring"]
    model_outputs_dir = cfg["paths"]["model_outputs_dir"]

    events_csv = Path(args.events_csv) if args.events_csv else cfg["paths"]["human_annotations_csv"]
    human_ref = load_human_reference(events_csv)
    log.info("Reference annotation: %s", events_csv)

    matches_path = results_dir / f"localization_matches_events_{args.mode}.csv"
    if not matches_path.exists():
        raise SystemExit(f"{matches_path} not found -- run 08b_score_localization_events.py "
                          f"--mode {args.mode} first.")
    matches = load_matches(matches_path)
    if args.clip_kind:
        matches = [m for m in matches if m["clip_kind"] == args.clip_kind]

    run_cache, out_rows = {}, []
    for m in matches:
        clip_id, event_id, clip_kind, run_idx = m["clip_id"], m["event_id"], m["clip_kind"], m["run_index"]

        if not args.ignore_perception_gate:
            perception_csv = results_dir / f"perception_scores_events_{clip_kind}_{args.mode}.csv"
            p_score = load_perception_scores(perception_csv).get((clip_id, event_id, clip_kind))
            if p_score is None or p_score < thresholds["perception_adequate_min_score"]:
                continue
        else:
            p_score = None

        ref = human_ref.get((clip_id, event_id))
        if ref is None or ref["d1"] is None or ref["d2"] is None:
            continue

        cache_key = (clip_id, event_id, clip_kind, run_idx)
        if cache_key not in run_cache:
            path = model_outputs_dir / f"{clip_id}__{event_id}__{clip_kind}__run{run_idx}__{args.mode}.json"
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            run_cache[cache_key] = data if (data and data.get("parsed")) else None
        run = run_cache[cache_key]
        if run is None:
            continue
        candidate = run["parsed"]["candidate_actions"][0]  # schema enforces exactly one
        model_d1 = candidate["semantic_syntactic_load"]
        model_d2 = candidate["pragmatic_depictive_affective_enrichment"]

        row_out = {
            "clip_id": clip_id, "event_id": event_id, "clip_kind": clip_kind, "run_index": run_idx,
            "perception_score": p_score if p_score is not None else "",
            "human_d1": round(ref["d1"], 2), "human_d2": round(ref["d2"], 2),
            "model_d1": model_d1, "model_d2": model_d2,
            "abs_err_d1": abs(ref["d1"] - model_d1), "abs_err_d2": abs(ref["d2"] - model_d2),
            "under_attribution_d1": ref["d1"] >= thresholds["high_d1_threshold"] and model_d1 <= thresholds["low_d1_threshold"],
            "over_attribution_d2": ref["d2"] <= thresholds["low_d2_threshold"] and model_d2 >= thresholds["high_d2_threshold"],
        }
        for mode in SEMIOTIC_MODES:
            h, mv = ref.get(mode), candidate.get(mode)
            row_out[f"human_{mode}"] = round(h, 2) if h is not None else ""
            row_out[f"model_{mode}"] = mv if mv is not None else ""
            row_out[f"abs_err_{mode}"] = abs(h - mv) if (h is not None and mv is not None) else ""
        out_rows.append(row_out)

    suffix = (f"{args.mode}_ungated" if args.ignore_perception_gate else args.mode) + args.out_suffix
    out_path = results_dir / f"attribution_scores_events_{suffix}.csv"
    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "clip_id", "event_id", "clip_kind", "run_index", "perception_score",
        "human_d1", "human_d2", "model_d1", "model_d2", "abs_err_d1", "abs_err_d2",
        "under_attribution_d1", "over_attribution_d2",
        "human_description", "model_description", "abs_err_description",
        "human_indication", "model_indication", "abs_err_indication",
        "human_depiction", "model_depiction", "abs_err_depiction"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    if not out_rows:
        log.warning("No matched%s events found -- check that 08b has been run (and 09b/rating is "
                    "complete, unless --ignore-perception-gate).",
                    "" if args.ignore_perception_gate else ", perception-adequate")
        log.info("Wrote (empty) %s", out_path)
        return

    by_kind = defaultdict(list)
    for r in out_rows:
        by_kind[r["clip_kind"]].append(r)

    for kind in CLIP_KINDS:
        rows = by_kind.get(kind)
        if not rows:
            continue
        mae_d1 = mean_absolute_error([r["human_d1"] for r in rows], [r["model_d1"] for r in rows])
        mae_d2 = mean_absolute_error([r["human_d2"] for r in rows], [r["model_d2"] for r in rows])
        rho_d1, p_d1 = spearman_correlation([r["human_d1"] for r in rows], [r["model_d1"] for r in rows])
        rho_d2, p_d2 = spearman_correlation([r["human_d2"] for r in rows], [r["model_d2"] for r in rows])
        under_rate = sum(r["under_attribution_d1"] for r in rows) / len(rows)
        over_rate = sum(r["over_attribution_d2"] for r in rows) / len(rows)
        log.info("--- clip_kind=%s (n=%d) ---", kind, len(rows))
        log.info("  D1 MAE=%.3f  Spearman rho=%.3f (p=%.3f)", mae_d1, rho_d1, p_d1)
        log.info("  D2 MAE=%.3f  Spearman rho=%.3f (p=%.3f)", mae_d2, rho_d2, p_d2)
        log.info("  Under-attribution rate (D1): %.3f", under_rate)
        log.info("  Over-attribution rate (D2): %.3f", over_rate)
        for mode in SEMIOTIC_MODES:
            pairs = [(r[f"human_{mode}"], r[f"model_{mode}"]) for r in rows
                     if r[f"human_{mode}"] != "" and r[f"model_{mode}"] != ""]
            if pairs:
                mae = mean_absolute_error([h for h, _ in pairs], [m for _, m in pairs])
                log.info("  Semiotic mode '%s': MAE=%.3f (n=%d)", mode, mae, len(pairs))
            else:
                log.info("  Semiotic mode '%s': no paired reference/model values -- skipped", mode)

    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
