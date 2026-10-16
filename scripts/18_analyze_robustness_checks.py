#!/usr/bin/env python3
"""Re-analyses of the ALREADY-COLLECTED data: the robustness checks that need no new
inference and no new human rating.

Every section here is computable from files already in the repo -- clip_manifest.csv, the
reference annotation, 09b's Tier-2 rating sheets, and the 07b model-output JSONs. Nothing in
this script sends a prompt to a model. It exists because several inferences in the write-up can be
adjudicated by data already collected, and were not initially put to it.

Sections, each keyed to the objection it answers:

  1. WITHIN-ARTICULATOR reference reliability. Table 1's alpha is computed across all 55 events,
     where the reference is near-constant WITHIN each articulator (D1 = 3 for all 32 manual
     events) and sharply different BETWEEN them. Two raters agreeing only on "manual high, face
     low" would produce alpha ~= 0.95 with zero item-level discrimination -- the same criticism
     the paper levels at the model's rho = 0.81. Recomputing alpha per articulator says which
     claim Table 1 can actually license. Undefined-through-no-variance is reported as such
     rather than as a number.

  2. channel_judged CONFUSION MATRIX, per-class recall, macro-F1, balanced accuracy. The paper
     reports overall self-selection accuracy (60%/58%) and the 0/110 non-dominant-hand rate. But
     103 of those 110 runs are items where non-dominant hand would have been the WRONG answer,
     so 0/110 is a mostly-irrelevant denominator; the recall figure on the 14 positive trials is
     the honest one. A full matrix also shows where the misses go.

  3. D1 x channel_judged in the articulator-WITHHELD cells. The the paper's per-articulator table flip ("the same
     15 face events score 1 when the articulator is named and 2 when it is not") is confounded:
     under withholding the model mostly self-selects MANUAL on those items, so the D1 = 2 is its
     score for a hand action it thinks it saw, not a different score for the same percept.
     Cross-tabbing D1 by what the model actually self-selected separates "the score follows the
     cue TEXT" from "the score follows the articulator the model is attending to".

  4. RELABELLING CONTROL on gate-failing items. Section 6.4's J(X,A) > J(X,B) signature is
     measured only on the 7 non-dominant-hand events, all of which failed the Tier-2 gate. So the
     signature might be a general property of bad descriptions rather than of the two-hand case.
     This recomputes the same J measures on the FACE and MANUAL items that ALSO failed the gate.

  5. CHANNEL-CONDITIONAL text similarity. Table 3's Jaccard is taken over handshape + movement +
     location for every articulator. For a face target the model correctly moves its content into
     the `nonmanual` field and writes N/A in the hand fields, so a low Jaccard there is partly
     forced by the output schema rather than evidence of redirected attention. This recomputes
     with the fields that carry content for each articulator, plus a label-stripped all-field
     variant, plus per-item values and an exact sign test for the 7/7 claim.

  6. PAIRED tests on Tier-2 adequacy. 41.8% vs 20.0% is a within-item comparison reported as two
     descriptive percentages. McNemar (exact) on the discordant pairs, Wilson intervals on each
     rate, and a cluster bootstrap resampling SOURCE VIDEOS (55 events come from 11 videos and
     40 clips, so events are not independent) turn it into an inferential claim.

  7. BENJAMINI-HOCHBERG correction across every Spearman test the paper reports.

  8. UNGATED / GATED / JOINT endpoints for D1 and D2. Tier 3 conditions on a post-treatment
     variable; reporting the same comparison ungated, gated, and as a joint success criterion
     (description adequate AND functional score within 0.5 of the reference) shows how much of
     the result is the gate.

  9. RUBRIC-CAP EXPOSURE. Appendix A's Step 1 caps a Tier-2 row at <= 1 when the model's
     self-reported articulator does not match the specified one. In the articulator-withheld
     cells the model was asked to choose, so the mediation sentence in the self-selection analysis ("the collapse
     is mediated by picking the wrong articulator") is partly built into the rubric. This counts
     exactly how many rows the cap can reach and reports the withheld-cell adequacy rate with
     those rows removed, which bounds how much of the effect is rubric-constructed. It does NOT
     recover uncapped scores -- that needs a rater.

 10. TRANSLATION-DRIVEN LEXICAL APPEAL, counted rather than illustrated. The paper's C15 case is
     one worked example. This counts, across all 55 items and all four cells, how many
     descriptions appeal to a named lexical sign (a glossed word in quotes, "the sign for X",
     "signing X") -- the observable trace of reasoning from the supplied translation to an
     expected form.

Usage:
    python scripts/18_analyze_robustness_checks.py
    python scripts/18_analyze_robustness_checks.py --clip-kind combined --config scripts/config/pipeline_config.qwen3vl_8b.yaml
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir, REPO_ROOT, parse_manifest_event
from lib.stats_utils import krippendorff_alpha_ordinal, mean_absolute_error

log = get_logger("analyze_robustness_checks")

DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"

MODES = ["grounded", "ablate_translation", "ablate_channel", "ablate_both"]
CHANNEL_GIVEN_MODES = ["grounded", "ablate_translation"]
CHANNEL_WITHHELD_MODES = ["ablate_channel", "ablate_both"]
CUE_PAIRS = [("grounded", "ablate_channel"), ("ablate_translation", "ablate_both")]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]

SCALES = {
    "D1 sem./syn. load": "semantic_syntactic_load",
    "D2 prag./dep./aff.": "pragmatic_depictive_affective_enrichment",
    "description": "description",
    "indication": "indication",
    "depiction": "depiction",
}
MODEL_SCALE_KEYS = {
    "semantic_syntactic_load": "semantic_syntactic_load",
    "pragmatic_depictive_affective_enrichment": "pragmatic_depictive_affective_enrichment",
    "description": "description",
    "indication": "indication",
    "depiction": "depiction",
}

# Which form-description fields carry the content for each target articulator. Section 5's
# answer to "Table 3's measure is not symmetric across channels": under a face target the model
# is *supposed* to write N/A in the hand fields, so comparing hand fields across the cue
# manipulation measures schema relocation, not redirected attention.
CORE_FIELDS = {
    "manual": ["handshape", "movement", "location"],
    "face": ["nonmanual"],
    "head_torso": ["nonmanual"],
    "non_dominant_hand": ["non_dominant_hand"],
}
ALL_CONTENT_FIELDS = ["hands", "handshape", "movement", "location", "nonmanual",
                      "non_dominant_hand"]
ACTION_FIELDS = ["handshape", "movement", "location"]  # Table 3's original field set

# Same stopword handling as analyze_channel_relabeling.py: the words carrying the LABEL rather
# than the described action are removed, so a pure relabelling cannot register as new content.
STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "then", "with", "is",
             "are", "as", "it", "its", "s", "signer", "signer's", "hand", "hands"}

# Trace of reasoning from the supplied translation to an expected form (section 10): a glossed
# lexical item named as such. Deliberately narrow -- it must name a WORD, not merely describe.
LEXICAL_APPEAL_RE = re.compile(
    r"(?:sign(?:ing|s|ed)?\s+(?:for\s+)?[\"'“‘「])"      # signing "X" / sign for 'X'
    r"|(?:\bthe\s+sign\s+(?:for|meaning)\b)"
    r"|(?:\blexical\s+sign\s+[\"'“‘「])"
    r"|(?:[「『][^」』]{1,12}[」』])"          # CJK corner-bracket gloss
    r"|(?:[\"'“‘][一-鿿]{1,8}[\"'”’])",    # quoted CJK term
    re.IGNORECASE)


# --------------------------------------------------------------------------------------------
# loading

def tokens(text):
    return {t for t in re.findall(r"[a-z']+", (text or "").lower()) if t not in STOPWORDS}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_manifest_events(manifest_path):
    """[(clip_id, event_id, target_channel, video_id)] in manifest order."""
    out = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("status") or "").strip().lower() == "excluded":
                continue
            for index in (1, 2):
                parsed = parse_manifest_event(row, index)
                if parsed is not None:
                    out.append((row["clip_id"], f"event{index}", parsed[2], row["video_id"]))
    return out


def load_reference(events_csv):
    """(clip_id, event_id) -> {rater_id: {scale: value}}."""
    ref = defaultdict(dict)
    with open(events_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["clip_id"], row["event_id"])
            scores = {}
            for col in SCALES.values():
                raw = (row.get(col) or "").strip()
                scores[col] = float(raw) if raw not in ("", "N/A", "n/a") else float("nan")
            ref[key][row["annotator_id"]] = scores
    return ref


def load_tier2(results_dir, clip_kind, mode):
    """(clip_id, event_id) -> {rater_id: perception_score}, plus the model text columns."""
    path = results_dir / f"perception_scores_events_{clip_kind}_{mode}.csv"
    scores, notes = defaultdict(dict), defaultdict(dict)
    if not path.exists():
        log.warning("missing Tier-2 sheet %s", path)
        return scores, notes
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["clip_id"], row["event_id"])
            raw = (row.get("perception_score") or "").strip()
            if raw:
                scores[key][row["rater_id"]] = float(raw)
            notes[key][row["rater_id"]] = (row.get("notes") or "").strip()
    return scores, notes


def load_run(model_outputs_dir, clip_id, event_id, clip_kind, mode):
    path = model_outputs_dir / f"{clip_id}__{event_id}__{clip_kind}__run1__{mode}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("parsed"):
        return None
    cands = data["parsed"].get("candidate_actions") or []
    return cands[0] if cands else None


def load_all_runs(model_outputs_dir, events, clip_kind):
    """{mode: {(clip_id, event_id): candidate_dict}}."""
    runs = {}
    for mode in MODES:
        runs[mode] = {}
        for clip_id, event_id, _, _ in events:
            cand = load_run(model_outputs_dir, clip_id, event_id, clip_kind, mode)
            if cand is not None:
                runs[mode][(clip_id, event_id)] = cand
    return runs


# --------------------------------------------------------------------------------------------
# small statistics helpers

def wilson_ci(successes, n, z=1.96):
    """Wilson score interval -- well-behaved at the small n and near-boundary rates here, where
    the normal-approximation interval would run outside [0, 1]."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(pairs):
    """Exact McNemar on paired binary outcomes [(a, b), ...]: b01/b10 are the discordant counts,
    tested against Binomial(n_discordant, 0.5). Exact rather than chi-square because the
    discordant counts here are small."""
    b01 = sum(1 for a, b in pairs if a and not b)
    b10 = sum(1 for a, b in pairs if b and not a)
    n_disc = b01 + b10
    if n_disc == 0:
        return b01, b10, float("nan")
    p = binomtest(b01, n_disc, 0.5, alternative="two-sided").pvalue
    return b01, b10, float(p)


def cluster_bootstrap_diff(items, n_boot=10000, seed=20260803):
    """Bootstrap CI for the paired difference in rate, resampling CLUSTERS (source videos) rather
    than events: 55 events come from 11 source videos, so events within a video are not
    independent and an event-level bootstrap would understate the interval.

    items: [(cluster_id, outcome_a_bool, outcome_b_bool)]
    """
    by_cluster = defaultdict(list)
    for cluster, a, b in items:
        by_cluster[cluster].append((a, b))
    clusters = sorted(by_cluster)
    if len(clusters) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        rows = [row for i in picked for row in by_cluster[clusters[i]]]
        if not rows:
            continue
        diffs.append(sum(r[0] for r in rows) / len(rows) - sum(r[1] for r in rows) / len(rows))
    if not diffs:
        return float("nan"), float("nan")
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def benjamini_hochberg(pvalues):
    """BH-adjusted p-values, order preserved with the input."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_min = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        k = m - rank + 1
        running_min = min(running_min, pvalues[idx] * m / k)
        adjusted[idx] = min(1.0, running_min)
    return adjusted


def alpha_or_undefined(matrix):
    """Krippendorff alpha, or an explanation string when it is undefined.

    Alpha is undefined when the observed values carry no variance at all: with every rater
    giving every item the same score there is no disagreement to be explained AND no expected
    disagreement to normalise by, so the statistic is 0/0 rather than 1.0. Reporting it as
    "perfect agreement" would be exactly the overclaim section 1 exists to test for.
    """
    arr = np.array(matrix, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, "no ratings"
    if np.unique(finite).size < 2:
        return None, f"undefined (no variance; all ratings = {finite[0]:g})"
    try:
        return float(krippendorff_alpha_ordinal(matrix)), None
    except Exception as exc:  # degenerate matrices still reach here on some inputs
        return None, f"undefined ({type(exc).__name__})"


def fmt(value, spec="{:.3f}", dash="--"):
    if value is None:
        return dash
    if isinstance(value, float) and not math.isfinite(value):
        return dash
    return spec.format(value)


def mean_or_none(values):
    return statistics.mean(values) if values else None


# --------------------------------------------------------------------------------------------
# sections

def section_within_articulator_alpha(out, events, ref):
    out.append("\n## 1. Reference reliability WITHIN articulator\n")
    out.append("\nThe headline agreement figures pool all 55 events. If the reference is near-constant "
               "inside each articulator and sharply different between them, two raters agreeing "
               "only on the between-articulator ordering already produce a high alpha with no "
               "item-level discrimination -- the same objection the paper raises against the "
               "model's rho = 0.81. Alpha is reported as *undefined* (not 1.0) wherever the "
               "ratings carry no variance, since it is 0/0 there.\n\n")

    raters = sorted({r for v in ref.values() for r in v})
    channel_of = {(c, e): ch for c, e, ch, _ in events}
    subsets = [("all events", None)] + [(f"within {ch}", ch) for ch in CHANNELS]

    out.append("| Score | " + " | ".join(
        f"{label} (n)" for label, _ in subsets) + " |\n")
    out.append("|---" * (len(subsets) + 1) + "|\n")
    for scale_label, col in SCALES.items():
        cells = []
        for _, ch in subsets:
            keys = [k for k in sorted(ref) if ch is None or channel_of.get(k) == ch]
            matrix = [[ref[k].get(r, {}).get(col, float("nan")) for k in keys] for r in raters]
            value, why = alpha_or_undefined(matrix)
            cells.append(f"{fmt(value)} ({len(keys)})" if value is not None else f"{why}")
        out.append(f"| {scale_label} | " + " | ".join(cells) + " |\n")

    out.append("\nReference score spread per articulator (mean, and how many distinct values the "
               "two raters used between them) -- this is what drives the row above:\n\n")
    out.append("| Score | " + " | ".join(CHANNELS) + " |\n" + "|---" * (len(CHANNELS) + 1) + "|\n")
    for scale_label, col in SCALES.items():
        cells = []
        for ch in CHANNELS:
            keys = [k for k in sorted(ref) if channel_of.get(k) == ch]
            vals = [ref[k][r][col] for k in keys for r in raters
                    if r in ref[k] and math.isfinite(ref[k][r][col])]
            cells.append(f"{mean_or_none(vals):.2f} ({len(set(vals))} distinct)"
                         if vals else "--")
        out.append(f"| {scale_label} | " + " | ".join(cells) + " |\n")


def section_confusion(out, events, runs):
    out.append("\n## 2. `channel_judged` confusion matrix, per-class recall, macro-F1 "
               "\n")
    out.append("\nA non-dominant-hand rate computed over all 110 withheld runs has a denominator of "
               "which 103 runs "
               "are items where non-dominant hand would have been WRONG. The recall on the "
               "positive trials -- 7 events x 2 withheld cells = 14 -- is the figure that "
               "bears the claim. Macro-F1 and balanced accuracy are reported because overall "
               "accuracy is dominated by the 32 manual items.\n")

    channel_of = {(c, e): ch for c, e, ch, _ in events}
    for mode in CHANNEL_WITHHELD_MODES:
        pairs = [(channel_of[k], (runs[mode][k].get("channel_judged") or "").strip())
                 for k in sorted(runs[mode])]
        out.append(f"\n### `{mode}` (n = {len(pairs)})\n\n")
        out.append("| target \\ judged | " + " | ".join(f"`{c}`" for c in CHANNELS) +
                   " | other | recall |\n" + "|---" * (len(CHANNELS) + 3) + "|\n")
        f1s, recalls = [], []
        for target in CHANNELS:
            row = [p for p in pairs if p[0] == target]
            if not row:
                continue
            counts = Counter(j for _, j in row)
            tp = counts.get(target, 0)
            fp = sum(1 for t, j in pairs if j == target and t != target)
            recall = tp / len(row)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            f1s.append(f1)
            recalls.append(recall)
            other = sum(v for k, v in counts.items() if k not in CHANNELS)
            out.append(f"| `{target}` ({len(row)}) | " +
                       " | ".join(str(counts.get(c, 0)) for c in CHANNELS) +
                       f" | {other} | **{tp}/{len(row)}** |\n")
        overall = sum(1 for t, j in pairs if t == j) / len(pairs)
        out.append(f"\nOverall accuracy {overall:.3f} ({sum(1 for t, j in pairs if t == j)}/"
                   f"{len(pairs)}); macro-F1 {mean_or_none(f1s):.3f}; balanced accuracy "
                   f"(mean per-class recall) {mean_or_none(recalls):.3f}.\n")

    ndh_pos = [(k, mode) for mode in CHANNEL_WITHHELD_MODES for k in sorted(runs[mode])
               if channel_of[k] == "non_dominant_hand"]
    hits = sum(1 for k, mode in ndh_pos
               if (runs[mode][k].get("channel_judged") or "").strip() == "non_dominant_hand")
    lo, hi = wilson_ci(hits, len(ndh_pos))
    out.append(f"\n**Non-dominant-hand recall on positive trials: {hits}/{len(ndh_pos)}** "
               f"(95% Wilson CI {lo:.3f}-{hi:.3f}). The 0/110 figure counts every run in both "
               f"withheld cells, most of which are not non-dominant-hand items.\n")

    out.append("\nLabel-overlap caveat, which the paper does not state: `manual` is a SUPERSET "
               "of `non_dominant_hand` (a non-dominant-hand action is a manual action), so a "
               "model answering `manual` on a non-dominant-hand target has not made an "
               "unambiguous error under the inventory's own label semantics. Where the "
               "misses go:\n\n")
    out.append("| item | " + " | ".join(f"`{m}`" for m in CHANNEL_WITHHELD_MODES) + " |\n" +
               "|---" * (len(CHANNEL_WITHHELD_MODES) + 1) + "|\n")
    for k in sorted({k for k, _ in ndh_pos}):
        out.append(f"| {k[0]}/{k[1]} | " + " | ".join(
            (runs[m].get(k, {}).get("channel_judged") or "--") for m in CHANNEL_WITHHELD_MODES
        ) + " |\n")


def section_d1_by_judged(out, events, runs, ref):
    out.append("\n## 3. D1 by what the model SELF-SELECTED, articulator-withheld cells "
               "\n")
    out.append("\nThe paper's D1 flip result holds the video constant and changes only the cue text, "
               "then reads the D1 change as one percept scored two ways. But under withholding "
               "the model mostly self-selects `manual` on face items, so its D1 = 2 there is a "
               "score for a hand action it believes it saw. Cross-tabbing D1 against "
               "`channel_judged` separates the two readings: if D1 tracks the *judged* "
               "articulator rather than the cue text, the dependence is an internal "
               "articulator->score mapping, which is a stronger claim than prompt compliance.\n")

    channel_of = {(c, e): ch for c, e, ch, _ in events}
    for mode in CHANNEL_WITHHELD_MODES:
        out.append(f"\n### `{mode}`\n\n")
        out.append("| target | judged | n | model D1 values | mean model D1 | mean ref. D1 |\n"
                   "|---|---|---|---|---|---|\n")
        buckets = defaultdict(list)
        for k in sorted(runs[mode]):
            judged = (runs[mode][k].get("channel_judged") or "?").strip()
            buckets[(channel_of[k], judged)].append(k)
        for (target, judged), keys in sorted(buckets.items()):
            d1s = [runs[mode][k].get("semantic_syntactic_load") for k in keys]
            d1s = [v for v in d1s if isinstance(v, (int, float))]
            refs = [statistics.mean([ref[k][r]["semantic_syntactic_load"] for r in ref[k]])
                    for k in keys if k in ref]
            out.append(f"| `{target}` | `{judged}` | {len(keys)} | "
                       f"{dict(sorted(Counter(d1s).items()))} | {fmt(mean_or_none(d1s), '{:.2f}')} "
                       f"| {fmt(mean_or_none(refs), '{:.2f}')} |\n")

    out.append("\nFace targets specifically -- the paper's flip case. Model D1 under the "
               "articulator-given cells, against D1 under withholding split by what was "
               "self-selected:\n\n")
    face_keys = [k for k in sorted(runs["grounded"]) if channel_of[k] == "face"]
    out.append("| item | given D1 (`grounded`) | withheld D1 (`ablate_channel`) | "
               "withheld `channel_judged` |\n|---|---|---|---|\n")
    for k in face_keys:
        g = runs["grounded"].get(k, {})
        w = runs["ablate_channel"].get(k, {})
        out.append(f"| {k[0]}/{k[1]} | {g.get('semantic_syntactic_load')} | "
                   f"{w.get('semantic_syntactic_load')} | "
                   f"{(w.get('channel_judged') or '--')} |\n")


def _core_text(cand, channel, fields=None):
    fd = cand.get("form_description", {}) or {}
    keys = fields if fields is not None else CORE_FIELDS.get(channel, ACTION_FIELDS)
    return " ".join((fd.get(k) or "") for k in keys)


def section_relabel_gate_control(out, events, runs, tier2_by_mode, adequate_min):
    out.append("\n## 4. Relabelling signature: is it specific to the two-hand case, or a "
               "property of any failed description?\n")
    out.append("\nAll 7 non-dominant-hand events fail the Tier-2 gate, so the relabelling analysis's "
               "J(X,A) > J(X,B) signature is measured entirely on descriptions already judged "
               "inadequate. The control: recompute the same measures on the FACE and MANUAL "
               "items that ALSO fail the gate. If the signature appears there too, it is what "
               "bad descriptions look like; if it appears only on non-dominant-hand items, the "
               "the paper's reading survives.\n")

    channel_of = {(c, e): ch for c, e, ch, _ in events}
    for given, withheld in CUE_PAIRS:
        t2 = tier2_by_mode[given]
        out.append(f"\n### `{given}` vs `{withheld}`\n\n")
        out.append("| target_channel | gate | n | mean J(X,A) | mean J(X,B) | A>B |\n"
                   "|---|---|---|---|---|---|\n")
        for channel in CHANNELS:
            for gate_label, want_pass in (("passes Tier 2", True), ("FAILS Tier 2", False)):
                ja, jb = [], []
                for k in sorted(runs[given]):
                    if channel_of[k] != channel or k not in runs[withheld]:
                        continue
                    scores = list(t2.get(k, {}).values())
                    if not scores:
                        continue
                    passed = statistics.mean(scores) >= adequate_min
                    if passed != want_pass:
                        continue
                    gf = runs[given][k].get("form_description", {}) or {}
                    wf = runs[withheld][k].get("form_description", {}) or {}
                    x = tokens(" ".join((gf.get(f) or "") for f in ACTION_FIELDS))
                    a = tokens(" ".join((wf.get(f) or "") for f in ACTION_FIELDS))
                    b = tokens(wf.get("non_dominant_hand") or "")
                    ja.append(jaccard(x, a))
                    jb.append(jaccard(x, b))
                if not ja:
                    continue
                wins = sum(1 for a, b in zip(ja, jb) if a > b)
                out.append(f"| `{channel}` | {gate_label} | {len(ja)} | "
                           f"{fmt(mean_or_none(ja))} | {fmt(mean_or_none(jb))} | "
                           f"{wins}/{len(ja)} |\n")


def section_channel_conditional_jaccard(out, events, runs):
    out.append("\n## 5. Channel-conditional and label-stripped text similarity "
               "\n")
    out.append("\nTable 3 compares handshape + movement + location for every articulator. Under a "
               "face target the model is instructed to put its content in `nonmanual` and mark "
               "the hand fields not-applicable, so the low face Jaccard is partly forced by "
               "schema relocation rather than being a positive control for redirected attention. "
               "Three measures side by side: the paper's original field set, the fields that "
               "carry content for each articulator, and all content fields merged with the field "
               "labels discarded.\n")

    channel_of = {(c, e): ch for c, e, ch, _ in events}
    for given, withheld in CUE_PAIRS:
        out.append(f"\n### `{given}` vs `{withheld}`\n\n")
        out.append("| target_channel | n | J (paper: hs+mv+loc) | J (channel-conditional core) | "
                   "J (all fields, labels stripped) |\n|---|---|---|---|---|\n")
        for channel in CHANNELS:
            orig, cond, merged = [], [], []
            for k in sorted(runs[given]):
                if channel_of[k] != channel or k not in runs[withheld]:
                    continue
                g, w = runs[given][k], runs[withheld][k]
                orig.append(jaccard(tokens(_core_text(g, channel, ACTION_FIELDS)),
                                    tokens(_core_text(w, channel, ACTION_FIELDS))))
                cond.append(jaccard(tokens(_core_text(g, channel)),
                                    tokens(_core_text(w, channel))))
                merged.append(jaccard(tokens(_core_text(g, channel, ALL_CONTENT_FIELDS)),
                                      tokens(_core_text(w, channel, ALL_CONTENT_FIELDS))))
            if not orig:
                continue
            out.append(f"| `{channel}` | {len(orig)} | {fmt(mean_or_none(orig))} | "
                       f"{fmt(mean_or_none(cond))} | {fmt(mean_or_none(merged))} |\n")

    out.append("\n### Per-item non-dominant-hand values, and an exact sign test on the 7/7 claim\n")
    for given, withheld in CUE_PAIRS:
        rows = []
        for k in sorted(runs[given]):
            if channel_of[k] != "non_dominant_hand" or k not in runs[withheld]:
                continue
            gf = runs[given][k].get("form_description", {}) or {}
            wf = runs[withheld][k].get("form_description", {}) or {}
            x = tokens(" ".join((gf.get(f) or "") for f in ACTION_FIELDS))
            a = tokens(" ".join((wf.get(f) or "") for f in ACTION_FIELDS))
            b = tokens(wf.get("non_dominant_hand") or "")
            identical_movement = ((gf.get("movement") or "").strip().lower() ==
                                  (wf.get("movement") or "").strip().lower())
            rows.append((k, jaccard(x, a), jaccard(x, b), identical_movement))
        if not rows:
            continue
        wins = sum(1 for _, a, b, _ in rows if a > b)
        p = binomtest(wins, len(rows), 0.5, alternative="greater").pvalue
        out.append(f"\n`{given}` vs `{withheld}`: A>B on {wins}/{len(rows)} items, "
                   f"exact sign test p = {p:.4f}. Byte-identical `movement` string on "
                   f"{sum(1 for r in rows if r[3])}/{len(rows)}.\n\n")
        out.append("| item | J(X,A) | J(X,B) | identical `movement` |\n|---|---|---|---|\n")
        for k, a, b, same in rows:
            out.append(f"| {k[0]}/{k[1]} | {a:.3f} | {b:.3f} | {'yes' if same else 'no'} |\n")


def _adequate_map(tier2, adequate_min):
    """(clip_id, event_id) -> bool, mean rating across raters reaches the gate."""
    return {k: statistics.mean(v.values()) >= adequate_min for k, v in tier2.items() if v}


def section_paired_adequacy(out, events, tier2_by_mode, adequate_min):
    out.append("\n## 6. Paired inference on Tier-2 adequacy\n")
    out.append("\n41.8% vs 20.0% is a within-item comparison reported as two independent "
               "percentages. Below: Wilson intervals on each rate, exact McNemar on the "
               "discordant pairs, and a bootstrap CI for the paired difference that resamples "
               "SOURCE VIDEOS -- 55 events come from 11 videos, so an event-level interval "
               "would treat correlated observations as independent.\n\n")

    video_of = {(c, e): v for c, e, _, v in events}
    adequate = {m: _adequate_map(tier2_by_mode[m], adequate_min) for m in MODES}

    out.append("| Cue cell | adequate | rate | 95% Wilson CI |\n|---|---|---|---|\n")
    for mode in MODES:
        a = adequate[mode]
        n, k = len(a), sum(a.values())
        lo, hi = wilson_ci(k, n)
        out.append(f"| `{mode}` | {k}/{n} | {k / n:.3f} | {lo:.3f}-{hi:.3f} |\n")

    comparisons = [
        ("articulator cue", "grounded", "ablate_channel"),
        ("articulator cue, translation withheld", "ablate_translation", "ablate_both"),
        ("translation cue", "grounded", "ablate_translation"),
        ("translation cue, articulator withheld", "ablate_channel", "ablate_both"),
    ]
    out.append("\n| Contrast | cell A | cell B | A only | B only | McNemar exact p | "
               "diff | 95% cluster-bootstrap CI |\n" + "|---" * 8 + "|\n")
    for label, mode_a, mode_b in comparisons:
        keys = sorted(set(adequate[mode_a]) & set(adequate[mode_b]))
        pairs = [(adequate[mode_a][k], adequate[mode_b][k]) for k in keys]
        b01, b10, p = mcnemar_exact(pairs)
        diff = (sum(a for a, _ in pairs) - sum(b for _, b in pairs)) / len(pairs)
        lo, hi = cluster_bootstrap_diff(
            [(video_of[k], adequate[mode_a][k], adequate[mode_b][k]) for k in keys])
        out.append(f"| {label} | `{mode_a}` | `{mode_b}` | {b01} | {b10} | {fmt(p, '{:.4f}')} | "
                   f"{diff:+.3f} | {fmt(lo, '{:+.3f}')} to {fmt(hi, '{:+.3f}')} |\n")

    out.append(f"\nCluster structure: {len(set(video_of.values()))} source videos, "
               f"{len(set(k[0] for k in video_of))} clips, {len(video_of)} events. Events per "
               f"video: {dict(sorted(Counter(video_of.values()).items()))}.\n")


def section_bh(out, events, runs, ref, tier2_by_mode, adequate_min):
    out.append("\n## 7. Benjamini-Hochberg across every Spearman test\n")
    out.append("\nThe paper reports Spearman tests over five scores x up to four cells and marks "
               "significance per test. Collected and BH-adjusted at a 5% false-discovery rate:\n\n")

    tests = []
    for mode in MODES:
        adequate = _adequate_map(tier2_by_mode[mode], adequate_min)
        keys = [k for k in sorted(runs[mode]) if adequate.get(k) and k in ref]
        for scale_label, col in SCALES.items():
            model_vals, ref_vals = [], []
            for k in keys:
                mv = runs[mode][k].get(MODEL_SCALE_KEYS[col])
                if not isinstance(mv, (int, float)):
                    continue
                model_vals.append(float(mv))
                ref_vals.append(statistics.mean([ref[k][r][col] for r in ref[k]]))
            if len(model_vals) < 3 or len(set(model_vals)) < 2 or len(set(ref_vals)) < 2:
                tests.append((mode, scale_label, len(model_vals), None, None))
                continue
            rho, p = spearmanr(ref_vals, model_vals)
            tests.append((mode, scale_label, len(model_vals), float(rho), float(p)))

    live = [t for t in tests if t[4] is not None]
    adj = benjamini_hochberg([t[4] for t in live])
    adj_by = {(t[0], t[1]): a for t, a in zip(live, adj)}
    out.append("| Cell | Score | n | rho | raw p | BH-adjusted p | survives 5% FDR |\n" +
               "|---" * 7 + "|\n")
    for mode, scale_label, n, rho, p in tests:
        if p is None:
            out.append(f"| `{mode}` | {scale_label} | {n} | -- | -- | -- | not testable "
                       f"(constant) |\n")
            continue
        a = adj_by[(mode, scale_label)]
        out.append(f"| `{mode}` | {scale_label} | {n} | {rho:+.3f} | {p:.4f} | {a:.4f} | "
                   f"{'yes' if a < 0.05 else '**no**'} |\n")
    out.append(f"\n{len(live)} testable correlations; "
               f"{sum(1 for a in adj if a < 0.05)} survive BH at 5% FDR, against "
               f"{sum(1 for t in live if t[4] < 0.05)} nominally significant at raw p < .05.\n")


def section_endpoints(out, events, runs, ref, tier2_by_mode, adequate_min):
    out.append("\n## 8. Ungated, gated, and joint endpoints\n")
    out.append("\nTier 3 conditions on a post-treatment variable, so the same comparison is "
               "reported three ways: over all 55 events (ungated), over gate-passing events "
               "only (the paper's Tier 3), and as a joint criterion -- description adequate AND "
               "functional score within 0.5 of the reference mean -- which has the same "
               "denominator in every cell and therefore does not move with the gate.\n\n")

    out.append("| Cell | Score | ungated n | ungated MAE | gated n | gated MAE | "
               "gated within 0.5 | joint success /55 |\n" + "|---" * 8 + "|\n")
    for mode in MODES:
        adequate = _adequate_map(tier2_by_mode[mode], adequate_min)
        for scale_label, col in [("D1", "semantic_syntactic_load"),
                                 ("D2", "pragmatic_depictive_affective_enrichment")]:
            all_pairs, gated_pairs, joint = [], [], 0
            for k in sorted(runs[mode]):
                mv = runs[mode][k].get(MODEL_SCALE_KEYS[col])
                if not isinstance(mv, (int, float)) or k not in ref:
                    continue
                rv = statistics.mean([ref[k][r][col] for r in ref[k]])
                all_pairs.append((rv, float(mv)))
                if adequate.get(k):
                    gated_pairs.append((rv, float(mv)))
                    if abs(rv - float(mv)) <= 0.5:
                        joint += 1
            if not all_pairs:
                continue
            within = sum(1 for r, m in gated_pairs if abs(r - m) <= 0.5)
            out.append(
                f"| `{mode}` | {scale_label} | {len(all_pairs)} | "
                f"{mean_absolute_error([p[0] for p in all_pairs], [p[1] for p in all_pairs]):.2f} | "
                f"{len(gated_pairs)} | "
                f"{mean_absolute_error([p[0] for p in gated_pairs], [p[1] for p in gated_pairs]):.2f} | "
                f"{within}/{len(gated_pairs)} | {joint}/{len(all_pairs)} |\n")


def section_cap_exposure(out, events, runs, tier2_by_mode, adequate_min):
    out.append("\n## 9. How much of the Tier-2 collapse is the rubric's own cap? "
               "\n")
    out.append("\nAppendix A's Step 1 caps a row at <= 1 when the model's self-reported "
               "articulator does not match the specified one. In the articulator-withheld cells "
               "the model was asked to choose it, so 'the collapse is mediated by picking the "
               "wrong articulator' is partly true by construction: a mismatch forces an "
               "inadequate rating regardless of how the description reads. Below: how many rows "
               "the cap can reach, and what the withheld-cell adequacy rate is on the "
               "cap-unreachable rows alone -- an upper bound on what the cue could be doing "
               "beyond redirection.\n\n")
    out.append("This does NOT recover uncapped scores. Reporting the description quality of a "
               "mismatched row independently of which articulator it describes needs a rater to "
               "re-score those rows with Step 1 suspended; it is the one check in this file that "
               "cannot be done from existing files.\n\n")

    channel_of = {(c, e): ch for c, e, ch, _ in events}
    out.append("| Cell | rows | cap-reachable (judged != target) | adequacy, all rows | "
               "adequacy, cap-unreachable rows only |\n" + "|---" * 5 + "|\n")
    for mode in MODES:
        adequate = _adequate_map(tier2_by_mode[mode], adequate_min)
        keys = [k for k in sorted(runs[mode]) if k in adequate]
        mismatch = [k for k in keys
                    if (runs[mode][k].get("channel_judged") or "").strip() != channel_of[k]]
        unreached = [k for k in keys if k not in set(mismatch)]
        rate_all = sum(adequate[k] for k in keys) / len(keys) if keys else float("nan")
        rate_un = (sum(adequate[k] for k in unreached) / len(unreached)
                   if unreached else float("nan"))
        out.append(f"| `{mode}` | {len(keys)} | {len(mismatch)} | "
                   f"{sum(adequate[k] for k in keys)}/{len(keys)} = {rate_all:.3f} | "
                   f"{sum(adequate[k] for k in unreached)}/{len(unreached)} = "
                   f"{fmt(rate_un)} |\n")


def section_lexical_appeal(out, events, runs):
    out.append("\n## 10. Translation-driven lexical appeal, counted\n")
    out.append("\nThe paper illustrates translation-driven distortion with one item (C15) and "
               "declines to quantify it. A description that names a lexical sign -- a glossed "
               "word in quotes, `the sign for X`, `signing X` -- is the observable trace of "
               "reasoning from the supplied translation to an expected form rather than "
               "reporting visible form. Counting that pattern per cell makes the direction "
               "testable: it should be commoner where the translation is supplied.\n\n")

    channel_of = {(c, e): ch for c, e, ch, _ in events}
    out.append("| Cell | translation given | items with a lexical appeal | rate |\n"
               + "|---" * 4 + "|\n")
    flagged = defaultdict(set)
    for mode in MODES:
        given = mode in ("grounded", "ablate_channel")
        n_hit = 0
        for k in sorted(runs[mode]):
            fd = runs[mode][k].get("form_description", {}) or {}
            text = " ".join((fd.get(f) or "") for f in ALL_CONTENT_FIELDS) + " " + \
                   (runs[mode][k].get("observable_evidence") or "")
            if LEXICAL_APPEAL_RE.search(text):
                n_hit += 1
                flagged[mode].add(k)
        n = len(runs[mode])
        out.append(f"| `{mode}` | {'yes' if given else 'no'} | {n_hit}/{n} | {n_hit / n:.3f} |\n")

    pairs = [("grounded", "ablate_translation"), ("ablate_channel", "ablate_both")]
    out.append("\nPaired, same items -- does supplying the translation add lexical appeals?\n\n")
    out.append("| translation given | translation withheld | appeal only when given | "
               "only when withheld | McNemar exact p |\n" + "|---" * 5 + "|\n")
    for given_mode, withheld_mode in pairs:
        keys = sorted(set(runs[given_mode]) & set(runs[withheld_mode]))
        prs = [(k in flagged[given_mode], k in flagged[withheld_mode]) for k in keys]
        b01, b10, p = mcnemar_exact(prs)
        out.append(f"| `{given_mode}` | `{withheld_mode}` | {b01} | {b10} | "
                   f"{fmt(p, '{:.4f}')} |\n")

    out.append("\nItems flagged under `grounded`, by target articulator: " +
               str(dict(sorted(Counter(channel_of[k] for k in flagged["grounded"]).items()))) +
               "\n\nFlagged item list (`grounded`): " +
               ", ".join(f"{c}/{e}" for c, e in sorted(flagged["grounded"])) + "\n")
    out.append("\nThis is a lexical-pattern count, not a validated hallucination judgement: a "
               "named sign that IS present in the target span is a correct description, not a "
               "distortion. It bounds how often the pattern occurs and shows which direction the "
               "cue pushes it; deciding correctness per item still needs the reference.\n")


# --------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--clip-kind", default="combined",
                        choices=["window", "ev1", "ev2", "combined"])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--reliability-csv", default=None,
                        help="independent (pre-discussion) reference used for section 1's "
                             "agreement statistics. Default: paths.human_annotations_csv.")
    parser.add_argument("--reference-csv", default=None,
                        help="post-discussion reference the MODEL is scored against in sections "
                             "3/7/8. Default: <reliability-csv stem>_checked.csv when it exists, "
                             "matching what 10b was run with for the paper's Tier-2 tables.")
    parser.add_argument("--out-name", default=None,
                        help="output filename under results_dir (default: robustness_checks_"
                             "<clip_kind>.md)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_outputs_dir = cfg["paths"]["model_outputs_dir"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    adequate_min = cfg["attribution_scoring"]["perception_adequate_min_score"]

    # Two reference files, deliberately not interchangeable (consolidate_checked_annotation.py):
    # the independent pass is the RELIABILITY estimate, and the post-discussion pass is the
    # REFERENCE the model is scored against. Using the reconciled scores for agreement would
    # inflate it (the raters had seen each other's); using the independent scores for Tier 3
    # would not reproduce the paper's tables. Section 1 gets the former, sections 3/7/8 the latter.
    reliability_csv = Path(args.reliability_csv) if args.reliability_csv \
        else cfg["paths"]["human_annotations_csv"]
    if args.reference_csv:
        reference_csv = Path(args.reference_csv)
    else:
        checked = reliability_csv.with_name(reliability_csv.stem + "_checked.csv")
        reference_csv = checked if checked.exists() else reliability_csv

    events = load_manifest_events(Path(args.manifest) if args.manifest else DEFAULT_MANIFEST)
    reliability = load_reference(reliability_csv)
    ref = load_reference(reference_csv)
    runs = load_all_runs(model_outputs_dir, events, args.clip_kind)
    tier2_by_mode, _ = {}, None
    for mode in MODES:
        tier2_by_mode[mode], _ = load_tier2(results_dir, args.clip_kind, mode)

    log.info("%d events, %d reference items, runs per mode: %s", len(events), len(ref),
             {m: len(runs[m]) for m in MODES})

    out = [
        "# Robustness re-analyses of the existing data\n",
        f"\nModel outputs: `{model_outputs_dir.name}`, clip kind `{args.clip_kind}`. "
        f"{len(events)} events; Tier-2 gate = mean rating >= {adequate_min}. "
        "Generated by `scripts/18_analyze_robustness_checks.py`; no inference and no new human "
        "rating is involved, so every number here is recomputed from files already in the "
        "repository.\n",
        (f"\nEvery statistic here, agreement included, is computed on `{reference_csv.name}`, "
         f"which holds the two annotators' final scores and is the annotation of record for this "
         f"study. An earlier pass exists in which the annotators were still applying subtly "
         f"different readings of the rubric; its disagreement measures that drift rather than any "
         f"property of the construct or of the items, so it is not a reliability estimate for this "
         f"reference. It is neither released nor used anywhere below.\n"
         if reliability_csv == reference_csv else
         f"\nReliability statistics (section 1) use the INDEPENDENT annotation pass "
         f"`{reliability_csv.name}`; every model-vs-reference comparison (sections 3, 7, 8) uses "
         f"the post-discussion reference `{reference_csv.name}`, which is what 10b was run with "
         f"for the paper's Tier-2 tables. The two are not interchangeable: reconciled scores would "
         f"inflate agreement, and the independent scores do not reproduce the published Tier-3 "
         f"numbers.\n"),
    ]
    section_within_articulator_alpha(out, events, reliability)
    section_confusion(out, events, runs)
    section_d1_by_judged(out, events, runs, ref)
    section_relabel_gate_control(out, events, runs, tier2_by_mode, adequate_min)
    section_channel_conditional_jaccard(out, events, runs)
    section_paired_adequacy(out, events, tier2_by_mode, adequate_min)
    section_bh(out, events, runs, ref, tier2_by_mode, adequate_min)
    section_endpoints(out, events, runs, ref, tier2_by_mode, adequate_min)
    section_cap_exposure(out, events, runs, tier2_by_mode, adequate_min)
    section_lexical_appeal(out, events, runs)

    out_path = results_dir / (args.out_name or f"robustness_checks_{args.clip_kind}.md")
    out_path.write_text("".join(out), encoding="utf-8")
    log.info("Wrote %s", out_path)
    print("".join(out))


if __name__ == "__main__":
    main()
