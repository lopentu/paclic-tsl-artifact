#!/usr/bin/env python3
"""Per-articulator score priors, and the label-only null the no-video result has to clear.

Three robustness numbers live here, all recomputed from files already in the
repository (the reference annotation plus the model's own outputs); no inference and no new
human rating is involved.

  1. THE PRIOR ON BOTH SIDES (the per-articulator table in the paper). The reference's mean per
     articulator against the modal value of the model's own output for the same events. The
     point of putting them side by side is that the reference is itself largely a function
     of the articulator -- D1 has no within-`manual` variance at all, D2 is ~0 on every
     manual and non-dominant-hand event and 2 on every face one -- so a model that is also
     a function of the articulator can correlate well with it while carrying no item-level
     information, and a model-vs-reference gap on those scales is a disagreement between
     two templates rather than evidence about which is right.

  2. THE LABEL-ONLY NULL for `analyze_visual_null.py`'s "identical to real" column. That a
     degraded run reproduces the real run's D1 on 46/55 items is only evidence of
     label-driven scoring if it EXCEEDS what a predictor knowing nothing but the target
     articulator achieves against the same real run. It does not: the real run's own D1 is
     51/55 recoverable that way. The informative residual is the other direction -- of the
     few items where the real run departs from its articulator's modal value, how many does
     the degraded run reproduce? (Almost none.) Without this null the 46/55 reads as a
     stronger claim than the data supports.

  3. CLUSTERING OF THE NON-DOMINANT-HAND EVENTS. The relabelling result is unanimous over 7
     events, but events cluster in source videos, so the exact sign test's p = 0.008 is an
     event-level figure. Taking the video as the unit of analysis is the honest ceiling, and
     it needs the per-video counts printed here.

Usage:
    python scripts/24_analyze_label_priors.py
    python scripts/24_analyze_label_priors.py --model qwen3vl_8b --clip-kind combined
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import get_logger, ensure_dir, REPO_ROOT

log = get_logger("analyze_label_priors")

CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]
MODES = ["grounded", "ablate_translation", "ablate_channel", "ablate_both"]
VISUALS = ["none", "black", "static", "shuffled"]
SCORES = [
    ("semantic_syntactic_load", "D1 load"),
    ("pragmatic_depictive_affective_enrichment", "D2 enrich."),
    ("description", "description"),
    ("indication", "indication"),
    ("depiction", "depiction"),
]


def load_reference(path):
    """(clip, event) -> {score: mean of the annotators}, plus channel and source video."""
    per_annotator = defaultdict(lambda: defaultdict(dict))
    channel, video = {}, {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        key = (row["clip_id"], row["event_id"])
        channel[key] = row["target_channel"]
        video[key] = row["video_id"]
        for score, _ in SCORES:
            value = row.get(score, "").strip()
            if value != "":
                per_annotator[key][score][row["annotator_id"]] = float(value)
    reference = {
        key: {s: statistics.mean(vals.values()) for s, vals in scores.items()}
        for key, scores in per_annotator.items()
    }
    return reference, channel, video


def first_candidate(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    actions = (obj.get("parsed") or {}).get("candidate_actions") or []
    return actions[0] if actions else None


def load_runs(model_dir, clip_kind, mode, visual=None):
    """(clip, event) -> the run's single candidate action, for one cue cell.

    Visual-null runs carry an extra filename field, so they are matched on 6 parts rather
    than 5 and live in a subdirectory; everything else is identical.
    """
    base = model_dir / "visual_null" if visual else model_dir
    n_parts = 6 if visual else 5
    runs = {}
    for path in sorted(base.glob("*.json")):
        parts = path.stem.split("__")
        if len(parts) != n_parts:
            continue
        if visual:
            clip, event, kind, this_visual, _run, this_mode = parts
            if this_visual != visual:
                continue
        else:
            clip, event, kind, _run, this_mode = parts
        if kind != clip_kind or this_mode != mode:
            continue
        candidate = first_candidate(path)
        if candidate is not None:
            runs[(clip, event)] = candidate
    return runs


def modal(values):
    counted = Counter(values)
    return counted.most_common(1)[0] if counted else (None, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--clip-kind", default="combined")
    ap.add_argument("--reference",
                    default="data/reference_annotations/events_checked.csv")
    args = ap.parse_args()

    reference, channel, video = load_reference(REPO_ROOT / args.reference)
    items = sorted(reference)
    by_channel = defaultdict(list)
    for key in items:
        by_channel[channel[key]].append(key)
    model_dir = REPO_ROOT / "data" / "model_outputs" / args.model
    cells = {mode: load_runs(model_dir, args.clip_kind, mode) for mode in MODES}
    log.info("%d reference events; runs per cell: %s", len(items),
             {m: len(r) for m, r in cells.items()})

    out = [f"# Per-articulator priors and the label-only null\n\n"
           f"Model outputs `{args.model}`, clip kind `{args.clip_kind}`, {len(items)} events, "
           f"reference `{Path(args.reference).name}` (annotators averaged). Generated by "
           f"`scripts/24_analyze_label_priors.py`; recomputed from released files only.\n\n"]

    # --- 1. the prior on both sides -----------------------------------------------------
    out.append("## 1. Reference mean against the model's modal output, per articulator\n\n"
               "`ref.` is the mean of the two experts over every event with that target "
               "articulator; `mod.` is the modal value of the model's own output on the same "
               "events, with the share at that mode in brackets. A column where `mod.` is "
               "constant down the table carries no articulator information; a row where "
               "`ref.` has no spread carries no item information.\n\n")
    for mode in MODES:
        runs = cells[mode]
        out.append(f"### `{mode}`\n\n| articulator | n | "
                   + " | ".join(f"{label} ref. | {label} mod." for _, label in SCORES) + " |\n"
                   + "|---" * (2 + 2 * len(SCORES)) + "|\n")
        for ch in CHANNELS:
            keys = [k for k in by_channel[ch] if k in runs]
            if not keys:
                continue
            cells_text = []
            for score, _ in SCORES:
                ref_mean = statistics.mean(reference[k].get(score, 0.0) for k in by_channel[ch])
                value, count = modal(runs[k][score] for k in keys
                                     if isinstance(runs[k].get(score), (int, float)))
                cells_text.append(f"{ref_mean:.2f} | {value} ({count}/{len(keys)})")
            out.append(f"| `{ch}` | {len(keys)} | " + " | ".join(cells_text) + " |\n")
        out.append("\nShare of the model's own outputs equal to the modal value for their "
                   "articulator, and the range of values it uses at all:\n\n"
                   "| score | at articulator mode | values used |\n|---|---|---|\n")
        for score, label in SCORES:
            modes_by_channel = {}
            for ch in CHANNELS:
                keys = [k for k in by_channel[ch] if k in runs]
                value, _ = modal(runs[k][score] for k in keys
                                 if isinstance(runs[k].get(score), (int, float)))
                modes_by_channel[ch] = value
            hits = sum(1 for k in items if k in runs
                       and runs[k].get(score) == modes_by_channel[channel[k]])
            spread = dict(sorted(Counter(runs[k][score] for k in items if k in runs
                                         and isinstance(runs[k].get(score), (int, float))).items()))
            out.append(f"| {label} | {hits}/{len(items)} = {hits / len(items):.3f} | {spread} |\n")
        out.append("\n")

    out.append("The reference's own values, which are the same in every cell:\n\n"
               "| score | values used |\n|---|---|\n")
    for score, label in SCORES:
        spread = dict(sorted(Counter(reference[k].get(score) for k in items).items()))
        out.append(f"| {label} | {spread} |\n")
    out.append("\n")

    # --- 2. the label-only null ---------------------------------------------------------
    out.append("## 2. The label-only null for the visual-null 'identical to real' column\n\n"
               "`label only` predicts, for every item, the modal value the REAL run gave to "
               "that item's target articulator, and nothing else. It is the rate any "
               "articulator-conditioned process reproduces the real run at, so a degraded "
               "condition only shows label-driven scoring if it reaches that rate -- and only "
               "shows recovered item-level information if it reproduces the real run's "
               "OFF-MODAL items.\n\n")
    real = cells["grounded"]
    for score, label in SCORES[:2]:
        modes_by_channel = {}
        for ch in CHANNELS:
            keys = [k for k in by_channel[ch] if k in real]
            value, _ = modal(real[k][score] for k in keys
                             if isinstance(real[k].get(score), (int, float)))
            modes_by_channel[ch] = value
        null_hits = sum(1 for k in items if k in real
                        and real[k].get(score) == modes_by_channel[channel[k]])
        off_modal = [k for k in items if k in real
                     and real[k].get(score) != modes_by_channel[channel[k]]]
        out.append(f"### {label}\n\nReal-run modal value per articulator: "
                   f"{ {c: modes_by_channel[c] for c in CHANNELS} }. "
                   f"**label only reproduces the real run on {null_hits}/{len(items)} items** "
                   f"({null_hits / len(items):.3f}); {len(off_modal)} items are off-modal.\n\n"
                   "| visual | identical to real | of the off-modal items | own rate at mode |\n"
                   "|---|---|---|---|\n")
        for visual in VISUALS:
            degraded = load_runs(model_dir, args.clip_kind, "grounded", visual=visual)
            shared = [k for k in items if k in real and k in degraded]
            if not shared:
                out.append(f"| `{visual}` | not run | -- | -- |\n")
                continue
            same = sum(1 for k in shared if degraded[k].get(score) == real[k].get(score))
            off_same = sum(1 for k in off_modal
                           if k in degraded and degraded[k].get(score) == real[k].get(score))
            own = {}
            for ch in CHANNELS:
                keys = [k for k in by_channel[ch] if k in degraded]
                value, _ = modal(degraded[k][score] for k in keys
                                 if isinstance(degraded[k].get(score), (int, float)))
                own[ch] = value
            own_hits = sum(1 for k in shared if degraded[k].get(score) == own[channel[k]])
            out.append(f"| `{visual}` | {same}/{len(shared)} | {off_same}/{len(off_modal)} | "
                       f"{own_hits}/{len(shared)} = {own_hits / len(shared):.3f} |\n")
        out.append("\n")

    # --- 3. clustering ------------------------------------------------------------------
    out.append("## 3. Source-video clustering of the target events\n\n"
               "The relabelling and self-selection results rest on the non-dominant-hand "
               "events. Their per-video counts bound what an exact test can show when the "
               "video rather than the event is the unit: with k clusters all agreeing in "
               "direction, the one-sided floor is 2^-k.\n\n"
               "| articulator | n | source videos | events per video |\n|---|---|---|---|\n")
    for ch in CHANNELS:
        keys = by_channel[ch]
        counts = dict(sorted(Counter(video[k] for k in keys).items()))
        out.append(f"| `{ch}` | {len(keys)} | {len(counts)} | {counts} |\n")
    out.append("\n")
    for ch in ["non_dominant_hand", "face"]:
        n_videos = len(set(video[k] for k in by_channel[ch]))
        out.append(f"- `{ch}`: {len(by_channel[ch])} events over {n_videos} source videos; "
                   f"if all {n_videos} agree in direction, the exact one-sided floor is "
                   f"p = {2.0 ** -n_videos:.4f}.\n")
    out.append("\nEvent identifiers, `non_dominant_hand`: "
               + ", ".join(f"{c}/{e} ({video[(c, e)]})"
                           for c, e in by_channel["non_dominant_hand"]) + "\n")

    results_dir = ensure_dir(REPO_ROOT / "data" / "results" / args.model)
    out_path = results_dir / "label_priors.md"
    out_path.write_text("".join(out), encoding="utf-8")
    log.info("Wrote %s", out_path)
    print("".join(out))


if __name__ == "__main__":
    main()
