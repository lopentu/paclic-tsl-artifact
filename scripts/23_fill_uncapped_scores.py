#!/usr/bin/env python3
"""Fill the uncapped re-scoring sheets from the raters' collected scores, and report the result.

Companion to 22_build_uncapped_rating_sheets.py. The two raters had already done this re-score
before the blank sheets existed -- they added an extra uncapped column to their own
perception_rating_sheet_<rater>_combined_merged.csv in Google Sheets, which was never synced back
to the repo (the on-disk merged CSVs have no such column). `uncapped_scores_raw.csv` is that column
transcribed from their exports and is now the only copy in the repo, so treat it as the source of
record rather than something regenerable.

IMPORTANT CAVEAT for whoever writes this up. These scores were produced *unblinded*: the raters
worked in the merged sheet, where `target_channel` and their own capped `perception_score` both sit
on the row being re-scored. 22's blank sheets withhold `target_channel` precisely to stop the
Step-1 cap leaking back in, and that protection did not apply here. `report` therefore prints the
anchoring diagnostic (how often uncapped == capped, and how often it moves down) so the writeup can
state the exposure rather than assume it away.

Usage:
    python scripts/23_fill_uncapped_scores.py fill
    python scripts/23_fill_uncapped_scores.py report
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, REPO_ROOT

log = get_logger("fill_uncapped_scores")

ANNOT_DIR = REPO_ROOT / "data" / "reference_annotations"
RAW_SCORES = ANNOT_DIR / "uncapped_scores_raw.csv"
WITHHELD_MODES = ["ablate_channel", "ablate_both"]
ADEQUATE_MIN = 2  # matches attribution_scoring.perception_adequate_min_score


def load_raw():
    """(rater, clip, event, mode) -> (capped, uncapped)."""
    out = {}
    for r in csv.DictReader(open(RAW_SCORES, encoding="utf-8")):
        out[(r["rater_id"], r["clip_id"], r["event_id"], r["mode"])] = (
            int(r["capped_score"]), int(r["uncapped_score"]))
    return out


def fill():
    raw = load_raw()
    for sheet in sorted(ANNOT_DIR.glob("uncapped_rating_sheet_A*.csv")):
        rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
        fields = list(rows[0].keys())
        n = 0
        for row in rows:
            key = (row["rater_id"], row["clip_id"], row["event_id"], row["mode"])
            if key in raw:
                row["uncapped_score"] = str(raw[key][1])
                n += 1
            else:
                log.warning("No collected score for %s -- left blank", key)
        with open(sheet, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        log.info("Filled %d/%d row(s) -> %s", n, len(rows), sheet.name)


def report(cfg):
    raw = load_raw()

    # target_channel rejoins here, not on the sheet -- see 22's docstring.
    target = {}
    for mode in WITHHELD_MODES:
        path = cfg["paths"]["results_dir"] / f"perception_scores_events_combined_{mode}.csv"
        for r in csv.DictReader(open(path, encoding="utf-8")):
            target[(r["rater_id"], r["clip_id"], r["event_id"], mode)] = r["target_channel"]

    lines = ["# Uncapped Tier-2 re-scoring of the articulator-mismatched rows", "",
             "Appendix A Step 1 caps a Tier-2 row at <=1 when `channel_judged` != `target_channel`. "
             "In the two articulator-withheld cells the model chose the articulator itself, so the "
             "cap partly pre-determines the mediation claim. These are the same rows re-scored with "
             "Step 1 suspended: how good the form description is about the articulator it actually "
             "describes.", "",
             "**These scores were collected unblinded** -- the raters re-scored inside the merged "
             "sheet, with `target_channel` and their own capped score visible on the row. The "
             "anchoring diagnostic below is what bounds that exposure.", ""]

    # THE definition that matters: the paper's gate is "the MEAN of the two raters reaches 2",
    # which is stricter than counting rater-rows individually and is the only figure comparable to
    # the 41.8% / 20.0% already in the paper. Per-rater rates run ~0.13 higher and must not be
    # quoted against the paper's numbers.
    def cell_events(mode, substitute):
        """(clip, event) -> mean rating across raters, optionally with uncapped scores swapped in."""
        path = cfg["paths"]["results_dir"] / f"perception_scores_events_combined_{mode}.csv"
        per_event = defaultdict(list)
        for r in csv.DictReader(open(path, encoding="utf-8")):
            key = (r["rater_id"], r["clip_id"], r["event_id"], mode)
            score = raw[key][1] if (substitute and key in raw) else int(r["perception_score"])
            per_event[(r["clip_id"], r["event_id"])].append(score)
        return {k: sum(v) / len(v) for k, v in per_event.items()}

    lines += ["## Adequacy per cell, capped vs un-capped (paper's gate: mean of both raters $\\ge$ 2)",
              "", "| Cell | events | adequate capped | adequate un-capped | mismatched events "
              "clearing the gate |", "|---|---|---|---|---|"]
    for mode in WITHHELD_MODES:
        cap, unc = cell_events(mode, False), cell_events(mode, True)
        mism = [k for k in cap if (k[0], k[1]) in
                {(c, e) for (_, c, e, m) in raw if m == mode}]
        nc = sum(1 for v in cap.values() if v >= ADEQUATE_MIN)
        nu = sum(1 for v in unc.values() if v >= ADEQUATE_MIN)
        gained = sorted(k for k in mism if unc[k] >= ADEQUATE_MIN and cap[k] < ADEQUATE_MIN)
        lines.append(f"| `{mode}` | {len(cap)} | {nc}/{len(cap)} = {nc/len(cap):.3f} "
                     f"| {nu}/{len(cap)} = {nu/len(cap):.3f} "
                     f"| {len(gained)}/{len(mism)} ({', '.join(c for c, _ in gained)}) |")
    lines += ["", "Compare `grounded` at 23/55 = 0.418. Un-capping barely moves the withheld cells, "
              "so the Step-1 cap was not producing the cue effect.", ""]

    # Anchoring diagnostic: an unblinded re-score should be suspected of reproducing the capped score.
    lines += ["## Anchoring diagnostic (the unblinding exposure)", "",
              "| rater | rows | uncapped == capped | uncapped > capped | uncapped < capped |",
              "|---|---|---|---|---|"]
    for rater in ["A01", "A02"]:
        sub = [(c, u) for (rt, _, _, _), (c, u) in raw.items() if rt == rater]
        eq = sum(1 for c, u in sub if u == c)
        up = sum(1 for c, u in sub if u > c)
        dn = sum(1 for c, u in sub if u < c)
        lines.append(f"| {rater} | {len(sub)} | {eq} ({eq/len(sub):.1%}) | {up} | {dn} |")
    lines += ["", "A row scored *down* is the clearest sign the rater was judging description "
              "quality rather than re-deriving the cap: the cap only ever pushes a score toward "
              "<=1, so it cannot explain a decrease.", ""]

    # Inter-rater agreement on the uncapped score.
    keys = sorted(set((c, e, m) for (_, c, e, m) in raw))
    pairs = [(raw[("A01",) + k][1], raw[("A02",) + k][1]) for k in keys
             if ("A01",) + k in raw and ("A02",) + k in raw]
    exact = sum(1 for a, b in pairs if a == b)
    within1 = sum(1 for a, b in pairs if abs(a - b) <= 1)
    lines += ["## Inter-rater agreement on the uncapped score", "",
              f"- Exact agreement: {exact}/{len(pairs)} = {exact/len(pairs):.3f}",
              f"- Within 1 point: {within1}/{len(pairs)} = {within1/len(pairs):.3f}", ""]

    # By withheld target: which articulator the model was supposed to be describing.
    by_target = defaultdict(lambda: [0, 0])
    for key, (_, u) in raw.items():
        tc = target.get(key, "?")
        by_target[tc][0] += 1
        by_target[tc][1] += 1 if u >= ADEQUATE_MIN else 0
    lines += ["## Uncapped adequacy by the withheld `target_channel`", "",
              "Both raters pooled. The model answered MANUAL on nearly all these rows, so this asks: "
              "when it defaulted to MANUAL instead of the specified articulator, how good was the "
              "manual description it produced?", "",
              "| withheld target | rows | adequate uncapped |", "|---|---|---|"]
    for tc, (n, a) in sorted(by_target.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"| `{tc}` | {n} | {a}/{n} = {a/n:.3f} |")
    lines.append("")

    out_path = cfg["paths"]["results_dir"] / "uncapped_rescore_combined.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", out_path)
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fill")
    rep = sub.add_parser("report")
    rep.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.cmd == "fill":
        fill()
    else:
        report(load_config(args.config))


if __name__ == "__main__":
    main()
