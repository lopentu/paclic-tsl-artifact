#!/usr/bin/env python3
"""Inter-rater agreement AND descriptive analysis of the Tier-2 perception scores, C01-C40 track.

Parallel scorer to 12_compute_annotator_agreement.py's Part 2 for the clip_manifest.csv /
07b_run_inference_events.py workflow. 12's `load_perception_by_item` keys rows by
(window_id, run_index, event_key) and reads one un-suffixed `perception_scores.csv`, so it finds
zero rows for this track -- the same silent-empty failure mode 09/09b and 10/10b already had. This
one keys by (clip_id, event_id) within a mode, reads 09b's
`perception_scores_events_{clip_kind}_{mode}.csv`, and loops over all four cue modes.

12's Part 1 (the five reference scores off events.csv) already works for this track unchanged and
is NOT duplicated here -- run 12 for that.

Two things are reported, and they answer different questions:

1. **Inter-rater agreement on the 0-2 perception_score.** Reported PER MODE (n=55 items, the
   natural unit) and then pooled over all four modes (n=220). The pooled number is reported
   second and labelled, because the 220 rows are four ratings of 55 items, not 220 independent
   items: each item recurs once per cue mode. Pooling inflates n without adding items, so the
   per-mode numbers are the ones to read for reliability.

2. **What the scores say about the model's form descriptions**: adequacy rate per cue mode, mean
   score, the breakdown by the manifest's `target_channel`, and the breakdown by whether the
   model's self-reported `channel_judged` matched that `target_channel`. The last one is the link
   between Tier 2 and the channel-cue question: it separates "the model described the right
   articulator badly" from "the model described a different articulator".

Two gate definitions are both reported, since they differ and the primary result uses the second:
  - per-rating adequacy: fraction of the 110 individual ratings that are 2 (what 09b logs)
  - per-item gated: fraction of the 55 items whose MEAN across raters is >= 2, i.e. what
    10b_score_attribution_events.py actually admits to Tier 3. With two raters and a threshold of
    2 on a 0-2 scale, this requires BOTH raters to have said 2.

Usage:
    python scripts/12b_compute_perception_agreement_events.py
    python scripts/12b_compute_perception_agreement_events.py --clip-kind combined
"""

import argparse
import csv
import itertools
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir
from lib.stats_utils import weighted_cohen_kappa, krippendorff_alpha_ordinal

log = get_logger("compute_perception_agreement_events")

MODES = ["grounded", "ablate_translation", "ablate_channel", "ablate_both"]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]

# How each mode renders the two cue blocks, for the report's own table -- the whole point of the
# per-mode breakdown is which cue was present, so spell it out rather than making the reader
# remember the mode names.
CUE_LABEL = {"grounded": "translation + channel",
             "ablate_translation": "channel only",
             "ablate_channel": "translation only",
             "ablate_both": "neither (span only)"}


def load_mode_rows(results_dir, clip_kind, mode):
    path = results_dir / f"perception_scores_events_{clip_kind}_{mode}.csv"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if (r.get("perception_score") or "").strip() != ""
                and (r.get("rater_id") or "").strip() != ""]
    return rows


def by_item(rows):
    """(clip_id, event_id) -> rater_id -> score, keeping items rated by >=2 raters."""
    out = defaultdict(dict)
    for r in rows:
        out[(r["clip_id"], r["event_id"])][r["rater_id"]] = float(r["perception_score"])
    return {k: v for k, v in out.items() if len(v) >= 2}


def agreement_block(items, label, lines):
    """items: (key -> rater -> score). Writes alpha / weighted kappa / raw agreement lines."""
    raters = sorted({r for v in items.values() for r in v})
    matrix = [[items[k].get(rater, float("nan")) for k in sorted(items)] for rater in raters]
    values = {v for row in matrix for v in row if v == v}
    lines.append(f"\n### {label} (items={len(items)}, raters={len(raters)})\n\n")
    if len(values) < 2:
        lines.append(f"- Agreement undefined: every rating is {values.pop():g} (no variance to "
                     "correct against chance). Unanimous, not missing.\n")
        return
    alpha = krippendorff_alpha_ordinal(matrix, level_of_measurement="ordinal")
    lines.append(f"- Krippendorff's alpha (ordinal): {alpha:.3f}\n")
    for a, b in itertools.combinations(raters, 2):
        pairs = [(v[a], v[b]) for v in items.values() if a in v and b in v]
        kappa = weighted_cohen_kappa([x for x, _ in pairs], [y for _, y in pairs])
        exact = sum(1 for x, y in pairs if x == y) / len(pairs)
        within1 = sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs)
        lines.append(f"- Weighted kappa {a} vs {b} (n={len(pairs)}): {kappa:.3f}\n")
        lines.append(f"- Raw agreement {a} vs {b} (n={len(pairs)}): exact {exact:.3f}, "
                     f"within 1 {within1:.3f}\n")
    for rater in raters:
        scores = [v[rater] for v in items.values() if rater in v]
        dist = {s: scores.count(s) for s in sorted(set(scores))}
        lines.append(f"- {rater} marginal: mean {statistics.mean(scores):.2f}, "
                     f"distribution {{{', '.join(f'{int(k)}: {v}' for k, v in dist.items())}}}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--clip-kind", default="combined",
                        choices=["window", "ev1", "ev2", "combined"],
                        help="which clip kind's rating sheets to read (default: combined, the "
                             "only kind Tier 2 was scoped to this round)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    min_adequate = cfg["attribution_scoring"]["perception_adequate_min_score"]

    lines = ["# Tier-2 perception scores: inter-rater agreement and score analysis\n\n",
             f"Clip kind: `{args.clip_kind}`. Threshold for \"adequate\": mean rating >= "
             f"{min_adequate} (`attribution_scoring.perception_adequate_min_score`).\n"]

    per_mode_rows, missing = {}, []
    for mode in MODES:
        rows = load_mode_rows(results_dir, args.clip_kind, mode)
        if rows is None:
            missing.append(mode)
            continue
        per_mode_rows[mode] = rows
    if missing:
        lines.append(f"\n**Missing sheets** for: {', '.join(missing)} -- run "
                     f"`09b_score_perception_adequacy_events.py aggregate --mode <mode> "
                     f"--clip-kind {args.clip_kind}` first.\n")
    if not per_mode_rows:
        raise SystemExit("No aggregated perception sheets found for clip_kind="
                         f"{args.clip_kind} in {results_dir}.")

    # --- 1. Agreement ----------------------------------------------------------------------
    lines.append("\n## 1. Inter-rater agreement on perception_score (0-2)\n")
    lines.append("\nPer cue mode first: 55 items is the real unit of analysis, and each mode is a "
                 "separate set of model descriptions being judged.\n")
    for mode, rows in per_mode_rows.items():
        agreement_block(by_item(rows), f"`{mode}` ({CUE_LABEL[mode]})", lines)

    pooled = {}
    for mode, rows in per_mode_rows.items():
        for k, v in by_item(rows).items():
            pooled[(mode,) + k] = v
    lines.append("\n### Pooled over all four cue modes\n")
    lines.append("\n_Read second, not first: these rows are four ratings of the same 55 items "
                 "(one per cue mode), so pooling raises n without adding items. Reported because "
                 "it is the agreement figure over the whole rated set, not because it is a "
                 "stronger reliability estimate than the per-mode numbers._\n")
    agreement_block(pooled, "all modes", lines)

    # --- 2. Score analysis -----------------------------------------------------------------
    lines.append("\n\n## 2. What the ratings say about the model's form descriptions\n")
    lines.append("\n### Adequacy by cue mode\n\n")
    lines.append("| mode | cues given | mean rating | per-rating adequate | per-item gated "
                 "(mean >= 2) |\n|---|---|---|---|---|\n")
    for mode, rows in per_mode_rows.items():
        scores = [float(r["perception_score"]) for r in rows]
        items = by_item(rows)
        means = {k: statistics.mean(v.values()) for k, v in items.items()}
        n_gated = sum(1 for m in means.values() if m >= min_adequate)
        per_rating = sum(1 for s in scores if s >= min_adequate) / len(scores)
        lines.append(f"| `{mode}` | {CUE_LABEL[mode]} | {statistics.mean(scores):.2f} | "
                     f"{sum(1 for s in scores if s >= min_adequate)}/{len(scores)} "
                     f"({per_rating:.1%}) | {n_gated}/{len(means)} ({n_gated/len(means):.1%}) |\n")

    lines.append("\n### Mean rating by target channel (rows = 2 raters x 55 items per mode)\n\n")
    header = " | ".join(f"`{m}`" for m in per_mode_rows)
    lines.append(f"| target_channel | n items | {header} |\n|---|---|"
                 + "---|" * len(per_mode_rows) + "\n")
    for channel in CHANNELS:
        cells, n_items = [], None
        for mode, rows in per_mode_rows.items():
            sel = [float(r["perception_score"]) for r in rows if r["target_channel"] == channel]
            n_items = len(sel) // 2
            cells.append(f"{statistics.mean(sel):.2f}" if sel else "--")
        if n_items:
            lines.append(f"| `{channel}` | {n_items} | " + " | ".join(cells) + " |\n")

    lines.append("\n### Mean rating by whether the model's `channel_judged` matched "
                 "`target_channel`\n\n")
    lines.append("| mode | matched: n / mean | mismatched: n / mean |\n|---|---|---|\n")
    for mode, rows in per_mode_rows.items():
        match = [float(r["perception_score"]) for r in rows
                 if r["channel_judged"] == r["target_channel"]]
        mism = [float(r["perception_score"]) for r in rows
                if r["channel_judged"] != r["target_channel"]]
        m_txt = f"{len(match)} / {statistics.mean(match):.2f}" if match else "0 / --"
        x_txt = f"{len(mism)} / {statistics.mean(mism):.2f}" if mism else "0 / --"
        lines.append(f"| `{mode}` | {m_txt} | {x_txt} |\n")

    out_path = results_dir / f"perception_agreement_events_{args.clip_kind}.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    log.info("Wrote %s", out_path)
    print("".join(lines))


if __name__ == "__main__":
    main()
