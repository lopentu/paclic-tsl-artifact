#!/usr/bin/env python3
"""Inter-annotator agreement on the five reference scores, and inter-rater agreement on the
model's form descriptions.

Two agreement questions, both reported here:

1. **Reference scores** (from the annotators' own sheets): D1, D2, and the three semiotic-mode
   ratings description / indication / depiction, each 0-3. The target span and channel are given
   by the researcher, so every annotator's row for an item is about the same action.

2. **Model form-description ratings** (from 09_score_perception_adequacy.py's filled sheets, if
   present): the 0-2 perception_score that each rater gave the *model's* description. As of
   2026-07-27 the annotators no longer write form descriptions themselves -- free-text
   descriptions gave no defensible way to decide whether two annotators had said the same thing,
   so only the model describes, and the humans rate those descriptions. This is where the
   description-side agreement number now comes from, and it is a rating of one shared text rather
   than a comparison of two different texts.

Reports, for each score:
    - Krippendorff's alpha (ordinal) over every annotator who rated an item with >=2 raters --
      the general-case measure regardless of annotator count.
    - Pairwise weighted Cohen's kappa for every annotator pair with overlapping items -- easier
      to interpret than alpha when there are exactly 2 raters.

Disagreement cases are reported as qualitative evidence, not just noise --
disagreement here may itself indicate genuine sign-gesture gradience.

Item keys are read flexibly so both sheet layouts work: the C01-C40 sheets from
build_annotation_sheets.py are keyed by (clip_id, event_id), the older per-window packets by
(window_id, event_key).

Usage:
    python scripts/12_compute_annotator_agreement.py
    python scripts/12_compute_annotator_agreement.py --perception-mode grounded
"""

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir, INFERENCE_MODES
from lib.stats_utils import weighted_cohen_kappa, krippendorff_alpha_ordinal

log = get_logger("compute_annotator_agreement")

# (primary, secondary) identifier columns, tried in order. The C01-C40 sheets use the first pair.
ITEM_KEY_CANDIDATES = [("clip_id", "event_id"), ("window_id", "event_key")]


def item_key_columns(fieldnames):
    for primary, secondary in ITEM_KEY_CANDIDATES:
        if primary in fieldnames:
            return primary, (secondary if secondary in fieldnames else None)
    raise SystemExit(
        f"Cannot find an item identifier in {fieldnames}. Expected one of "
        f"{[c[0] for c in ITEM_KEY_CANDIDATES]}."
    )


def load_by_item(events_csv):
    """item key -> annotator_id -> row, keeping only items rated by >=2 annotators."""
    by_item = defaultdict(dict)
    with open(events_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        primary, secondary = item_key_columns(reader.fieldnames)
        for row in reader:
            key = (row[primary], row[secondary]) if secondary else (row[primary],)
            by_item[key][row["annotator_id"]] = row
    return {k: v for k, v in by_item.items() if len(v) >= 2}


def load_perception_by_item(perception_csv):
    """(window_id, run_index, event_key) -> rater_id -> row, for the model-description ratings."""
    by_item = defaultdict(dict)
    if not Path(perception_csv).exists():
        return {}
    with open(perception_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rater = (row.get("rater_id") or "").strip()
            if not rater or (row.get("perception_score") or "").strip() == "":
                continue
            by_item[(row["window_id"], row["run_index"], row["event_key"])][rater] = row
    return {k: v for k, v in by_item.items() if len(v) >= 2}


def build_value_matrix(by_item, field, cast=float):
    annotators = sorted({a for item in by_item.values() for a in item})
    items = sorted(by_item.keys())
    matrix = []
    for annotator in annotators:
        row = []
        for item in items:
            r = by_item[item].get(annotator)
            val = r.get(field, "") if r is not None else ""
            row.append(cast(val) if str(val).strip() != "" else float("nan"))
        matrix.append(row)
    return matrix, annotators, items


def pairwise_kappa(by_item, field):
    annotators = sorted({a for item in by_item.values() for a in item})
    results = {}
    for a, b in itertools.combinations(annotators, 2):
        pairs = [(item[a].get(field, ""), item[b].get(field, "")) for item in by_item.values() if a in item and b in item]
        pairs = [(x, y) for x, y in pairs if str(x).strip() != "" and str(y).strip() != ""]
        if len(pairs) < 2:
            continue
        ratings_a = [float(x) for x, _ in pairs]
        ratings_b = [float(y) for _, y in pairs]
        results[(a, b)] = (weighted_cohen_kappa(ratings_a, ratings_b), len(pairs))
    return results


def report_field(by_item, field, label, lines, annotator_col_label="annotators",
                 measurement="ordinal", tolerance=1):
    """measurement="ordinal" for the 0-3 integer scores (weighted kappa applies); "interval" for
    the continuous 0.0-1.0 optional fields (confidence, semantic_dependence), where kappa is
    undefined for continuous data and alpha needs the interval, not ordinal, distance function.
    `tolerance` sets what "close enough" means for the always-defined raw-agreement line below --
    1 scale point for the 0-3 fields, 0.2 for the 0.0-1.0 fields.
    """
    matrix, annotators, _ = build_value_matrix(by_item, field)
    lines.append(f"\n## {label}\n")

    values = {v for row in matrix for v in row if v == v}  # v == v filters out nan
    if not values:
        lines.append("- No values filled in for this field.\n")
        return

    # Chance-corrected agreement is undefined without variance -- if every rater gave the same
    # score to every item, there is no chance level to correct against. That is unanimous
    # agreement, not missing data, so say which it is rather than printing nan.
    degenerate = len(values) < 2
    if degenerate:
        only = values.pop()
        lines.append(f"- Krippendorff's alpha / weighted kappa: **undefined** -- every rating is "
                     f"{only:g}, so there is no variance to correct for chance against. This is "
                     f"unanimous agreement on a constant field, not a failure; report it as such "
                     f"(alpha would be misleading) and rely on the raw-agreement rate below.\n")
    else:
        try:
            alpha = krippendorff_alpha_ordinal(matrix, level_of_measurement=measurement)
            lines.append(f"- Krippendorff's alpha ({measurement}, {len(annotators)} "
                         f"{annotator_col_label}): {alpha:.3f}\n")
        except ImportError as exc:
            lines.append(f"- Krippendorff's alpha: SKIPPED ({exc})\n")
        if measurement == "ordinal":
            # Weighted kappa needs a small set of discrete labels; skip it for the continuous
            # 0.0-1.0 fields (sklearn rejects continuous input outright as "not supported").
            for (a, b), (kappa, n) in pairwise_kappa(by_item, field).items():
                lines.append(f"- Weighted kappa {a} vs {b} (n={n}): {kappa:.3f}\n")

    # Always defined, and readable at pilot sample sizes: how often the raters simply matched,
    # and how often they were within `tolerance` of each other.
    for (a, b), (exact, within_tol, n) in pairwise_exact_agreement(by_item, field, tolerance).items():
        lines.append(f"- Raw agreement {a} vs {b} (n={n}): exact {exact:.3f}, within {tolerance:g} "
                     f"{within_tol:.3f}\n")


def pairwise_exact_agreement(by_item, field, tolerance=1):
    """(a, b) -> (exact-match rate, within-tolerance rate, n) over items both rated."""
    annotators = sorted({a for item in by_item.values() for a in item})
    results = {}
    for a, b in itertools.combinations(annotators, 2):
        pairs = []
        for item in by_item.values():
            if a not in item or b not in item:
                continue
            x, y = str(item[a].get(field, "")).strip(), str(item[b].get(field, "")).strip()
            if x != "" and y != "":
                pairs.append((float(x), float(y)))
        if not pairs:
            continue
        exact = sum(1 for x, y in pairs if x == y) / len(pairs)
        within_tol = sum(1 for x, y in pairs if abs(x - y) <= tolerance) / len(pairs)
        results[(a, b)] = (exact, within_tol, len(pairs))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--events-csv", default=None,
                        help="default: paths.human_annotations_csv")
    parser.add_argument("--out-suffix", default="",
                        help="appended to the output filename, so scoring a different reference "
                             "annotation cannot silently overwrite the previous result "
                             "(same convention as 10b/analyze_attribution_summary)")
    parser.add_argument("--perception-mode", default="grounded", choices=INFERENCE_MODES,
                        help="which mode's filled perception sheet to compute description-rating "
                             "agreement on (default: grounded)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    events_csv = Path(args.events_csv) if args.events_csv else cfg["paths"]["human_annotations_csv"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    lines = ["# Inter-annotator agreement\n"]

    # --- 1. The five reference scores -------------------------------------------------------
    by_item = load_by_item(events_csv) if Path(events_csv).exists() else {}
    lines.append("\n# Part 1 -- reference scores (annotators' own judgments)\n")
    if not by_item:
        lines.append(f"\n- Not yet available: no item in `{Path(events_csv).name}` has >=2 "
                     "annotators. Each annotator fills their own sheet "
                     "(build_annotation_sheets.py), then consolidate_human_annotations.py merges "
                     "them.\n")
        log.warning("No items with >=2 annotators in %s -- skipping Part 1.", events_csv)
    else:
        lines.append(f"\nItems with >=2 annotators: {len(by_item)}\n")
        ordinal_fields = [("semantic_syntactic_load", "D1 (semantic/syntactic load), 0-3"),
                          ("pragmatic_depictive_affective_enrichment", "D2 (pragmatic/depictive/affective enrichment), 0-3"),
                          ("description", "Semiotic mode: description, 0-3"),
                          ("indication", "Semiotic mode: indication, 0-3"),
                          ("depiction", "Semiotic mode: depiction, 0-3")]
        for field, label in ordinal_fields:
            report_field(by_item, field, label, lines)

        # Optional columns: reported when filled, since they say how the scores were reached
        # rather than what they were. Absent columns are simply skipped. Continuous 0.0-1.0
        # fields need interval (not ordinal) treatment and a finer tolerance for "close enough".
        for field, label in [("confidence", "Confidence (optional, 0.0-1.0)"),
                             ("semantic_dependence", "Semantic dependence (optional, 0.0-1.0)")]:
            if any(str(r.get(field, "")).strip() for item in by_item.values() for r in item.values()):
                report_field(by_item, field, label, lines, measurement="interval", tolerance=0.2)

    # --- 2. Ratings of the model's form descriptions ----------------------------------------
    perception_csv = cfg["paths"]["perception_scores_csv"].with_name(
        cfg["paths"]["perception_scores_csv"].stem + f"_{args.perception_mode}.csv")
    perception_by_item = load_perception_by_item(perception_csv)
    lines.append("\n\n# Part 2 -- ratings of the MODEL's form descriptions "
                 f"(mode={args.perception_mode})\n")
    lines.append("\n_Agreement on the 0-2 perception_score each rater gave the model's form "
                 "description. Since 2026-07-27 this is where description-side agreement comes "
                 "from: the annotators no longer write competing free-text descriptions, they rate "
                 "one shared text, so the number is interpretable._\n")
    if not perception_by_item:
        lines.append(f"\n- Not yet available: `{perception_csv.name}` has no item rated by >=2 "
                     "raters with a `rater_id`. Build the sheet with "
                     f"`09_score_perception_adequacy.py build --mode {args.perception_mode}`, have "
                     "each rater fill their own copy (setting `rater_id`), then `aggregate` them.\n")
        log.warning("No perception items with >=2 raters in %s -- skipping Part 2.", perception_csv)
    else:
        lines.append(f"\nItems with >=2 raters: {len(perception_by_item)}\n")
        report_field(perception_by_item, "perception_score",
                     "Model form-description adequacy (0-2)", lines, annotator_col_label="raters")

    out_path = results_dir / f"annotator_agreement{args.out_suffix}.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    log.info("Wrote %s", out_path)
    print("".join(lines))


if __name__ == "__main__":
    main()
