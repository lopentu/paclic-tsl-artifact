#!/usr/bin/env python3
"""Uncapped Tier-2 re-scoring sheets for the articulator-mismatched rows.

Appendix A Step 1 caps a Tier-2 row at <=1 whenever the model's self-reported articulator
(`channel_judged`) differs from the specified one (`target_channel`). In the two articulator-
withheld cells the model was *asked* to choose the articulator, so the cap makes "the collapse is
mediated by picking the wrong articulator" partly true by construction. Section 9 of
`18_analyze_robustness_checks.py`'s report flags this as the one check that
cannot be answered from existing files because it needs a human to re-score with Step 1 suspended.

This builds those re-scoring sheets: the mismatched rows only, one sheet per rater, with just the
score column left to fill.

Two things differ from the `09b` blank sheet on purpose:

1. **`target_channel` is withheld from the sheet.** The rating guide's Step 0 tells the rater to
   pick the "core fields" from `target_channel`, and Step 1 tells them to score a mismatched row
   0-or-1 on sight. Both rules are exactly what this re-score must suspend, and leaving the column
   on the sheet invites re-applying them from habit. `channel_judged` stays, and takes over as the
   core-field selector: the question is whether the description is accurate about the articulator
   it actually describes. `target_channel` is rejoinable afterwards from clip_manifest.csv on
   (clip_id, event_id), so nothing is lost analytically.
2. **The score column is `uncapped_score`, not `perception_score`,** so a filled sheet can never be
   pooled with the original capped ratings by accident. The same reason the sheets carry a `mode`
   column: rows from both withheld cells sit in one file per rater, which the `09b` naming
   (one file per mode) cannot express.

Rows are selected per (mode, clip_id, event_id) -- mismatch is a property of the model output, not
of the rater, so both raters get an identical row set, as inter-rater agreement on the re-score
requires. The same event appears once per withheld cell; the two descriptions are different text
(different cue condition), so they are separate rating tasks, and the mode-blocked row order keeps
the pair well apart in the file.

Usage:
    python scripts/22_build_uncapped_rating_sheets.py
    python scripts/22_build_uncapped_rating_sheets.py --out-dir /tmp/sheets
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir, REPO_ROOT

log = get_logger("build_uncapped_rating_sheets")

# Only the articulator-withheld cells can have mismatched rows; the given cells are 0/55 by
# construction (the articulator is supplied), so there is nothing to re-score there.
WITHHELD_MODES = ["ablate_channel", "ablate_both"]

MODEL_FIELDS = ["model_hands", "model_handshape", "model_movement", "model_location",
                "model_nonmanual", "model_non_dominant_hand", "model_temporal_alignment"]

# NOTE: no `target_channel` -- withheld deliberately, see module docstring.
SHEET_COLUMNS = (["mode", "clip_id", "event_id", "clip_kind", "run_index", "channel_judged"]
                 + MODEL_FIELDS + ["rater_id", "uncapped_score", "notes"])


def collect_mismatched(results_dir, clip_kind):
    """One row per (mode, clip_id, event_id) whose channel_judged != target_channel.

    Reads the *filled* score files because they are where channel_judged and target_channel sit
    side by side; the rater's own perception_score is dropped on the way out, so nothing from the
    capped round leaks onto the new sheet.
    """
    rows, raters = [], set()
    for mode in WITHHELD_MODES:
        path = results_dir / f"perception_scores_events_{clip_kind}_{mode}.csv"
        if not path.exists():
            log.warning("Missing %s -- skipping mode %s", path, mode)
            continue
        seen, n_total = set(), 0
        for row in csv.DictReader(open(path, encoding="utf-8")):
            n_total += 1
            if row.get("rater_id"):
                raters.add(row["rater_id"])
            if row["channel_judged"] == row["target_channel"]:
                continue
            key = (row["clip_id"], row["event_id"])
            if key in seen:  # same event, other rater -- identical model output
                continue
            seen.add(key)
            out = {"mode": mode, "channel_judged": row["channel_judged"], "notes": ""}
            for col in ["clip_id", "event_id", "clip_kind", "run_index"] + MODEL_FIELDS:
                out[col] = row[col]
            rows.append(out)
        log.info("%s: %d mismatched event(s) out of %d rated row(s)", mode, len(seen), n_total)
    return rows, sorted(raters)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--clip-kind", default="combined",
                        help="which clip kind's score files to read (default: combined)")
    parser.add_argument("--out-dir", default=None,
                        help="default: data/reference_annotations/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows, raters = collect_mismatched(cfg["paths"]["results_dir"], args.clip_kind)
    if not rows:
        log.error("No mismatched rows found -- nothing to re-score.")
        return 1
    if not raters:
        log.error("No rater_id values found in the source files.")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / "reference_annotations"
    ensure_dir(out_dir)

    for rater in raters:
        out_path = out_dir / f"uncapped_rating_sheet_{rater}.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "rater_id": rater, "uncapped_score": ""})
        log.info("Wrote %d row(s) to rate -> %s", len(rows), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
