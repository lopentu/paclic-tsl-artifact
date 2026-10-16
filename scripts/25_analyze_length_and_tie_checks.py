#!/usr/bin/env python3
"""Six checks on numbers the paper reports, all recomputed from files already in the
repository. No new inference and no new human rating.

Each section answers one objection that can be raised against a reported number:

  1. **The similarity measure is length-biased.** `B` (a run's own `non_dominant_hand` field) is
     a short dedicated field; `X` (a full structured description) is long. Jaccard's denominator
     is the union, so J(X, B) is bounded above by |B|/|X| even when B is wholly contained in X --
     with |B| ~ 7 and |X| ~ 18 tokens that ceiling is ~0.39, and the reported 0.111 sits under
     it. The merged-field and character-ratio variants in the paper inherit the same bias, so
     neither answers the objection. Containment (Szymkiewicz-Simpson: intersection over the
     SMALLER set) removes it by construction, and a mismatched-pair permutation says whether the
     matched B is present in X any more than an unrelated event's B would be.

  2. **J(X, A) alone carries the claim, and has no interval.** The three-way J(X, A) comparison
     (manual no-switch anchor / face redirection anchor / non-dominant hand) is length-comparable
     on both sides and therefore immune to (1). It needs a CI, clustered on source video, since
     that is the unit the paper's other resampling uses.

  3. **Spearman rho is unreliable under heavy ties.** The model uses only D1 in {1, 2} on gated
     rows, so a large share of pairs are tied. Kendall's tau-b applies a tie correction; the tie
     fraction itself should be reported either way.

  4. **Tier-1 alpha is low where raw agreement is high.** The paper explains this as skewed
     marginals. Gwet's AC1 and Bennett's S are the standard chance corrections that do not
     inflate the expected-agreement term under skew, and turn the explanation into a number.

  5. **The degradation ladder reports stated confidence for two conditions only.** `static` and
     `shuffled` confidence are what separate "the model detects an absent signal" from "the model
     detects a broken signal", and D1-agreement under shuffling needs splitting by articulator
     because face targets are the least motion-dependent items in the set.

  6. **The shuffled span's frame count is never reported.** Shuffling 8 frames and shuffling 100
     are different manipulations; the paper gives fps and clip durations but never the product.

Usage:
    python scripts/25_analyze_length_and_tie_checks.py
    python scripts/25_analyze_length_and_tie_checks.py --config scripts/config/pipeline_config.qwen3vl_8b.yaml
"""

import argparse
import csv
import json
import math
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir, REPO_ROOT, parse_manifest_event

log = get_logger("analyze_length_and_tie_checks")

DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"

CUE_PAIRS = [("grounded", "ablate_channel", "translation given in both"),
             ("ablate_translation", "ablate_both", "translation withheld in both")]
ACTION_FIELDS = ["handshape", "movement", "location"]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]
MODES = ["grounded", "ablate_translation", "ablate_channel", "ablate_both"]
MODE_LABEL = {"grounded": "both", "ablate_translation": "-transl.",
              "ablate_channel": "-artic.", "ablate_both": "neither"}
VISUAL_CONDITIONS = ["real", "none", "black", "static", "shuffled"]
SCORE_COLS = [("D1 load", "d1"), ("D2 enrich.", "d2"), ("description", "description"),
              ("indication", "indication"), ("depiction", "depiction")]

# Identical to analyze_channel_relabeling.py: `hand`/`hands`/`signer` are dropped so a pure
# relabelling cannot register as a change in described content.
STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "then", "with", "is",
             "are", "as", "it", "its", "s", "signer", "signer's", "hand", "hands"}

BOOTSTRAP_REPLICATES = 10000
PERMUTATION_SEED = 20260814


# --------------------------------------------------------------------------------------
# text measures
# --------------------------------------------------------------------------------------

def tokens(text):
    return {t for t in re.findall(r"[a-z']+", (text or "").lower()) if t not in STOPWORDS}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a, b):
    """Szymkiewicz-Simpson overlap: intersection over the smaller set.

    Unlike Jaccard this has no length ceiling -- a short B fully present in a long X scores 1.0,
    so a low value cannot be produced by the field-size asymmetry alone.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def load_events(manifest_path):
    """[(clip_id, event_id, target_channel, video_id)] in manifest order."""
    out = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("status") or "").strip().lower() == "excluded":
                continue
            for index in (1, 2):
                parsed = parse_manifest_event(row, index)
                if parsed is not None:
                    out.append((row["clip_id"], f"event{index}", parsed[2],
                                row.get("video_id", "")))
    return out


def load_run(model_dir, clip_id, event_id, mode, clip_kind="combined", visual=None):
    name = (f"{clip_id}__{event_id}__{clip_kind}__run1__{mode}.json" if visual is None
            else f"{clip_id}__{event_id}__{clip_kind}__{visual}__run1__{mode}.json")
    path = model_dir / name
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates = (data.get("parsed") or {}).get("candidate_actions") or []
    return candidates[0] if candidates else None


def triples(model_dir, events, given_mode, withheld_mode):
    """Per event: X (cued description), A (withheld run's self-selected description),
    B (that same withheld run's own non-dominant-hand field)."""
    out = []
    for clip_id, event_id, channel, video_id in events:
        g = load_run(model_dir, clip_id, event_id, given_mode)
        w = load_run(model_dir, clip_id, event_id, withheld_mode)
        if g is None or w is None:
            continue
        gf, wf = g.get("form_description", {}), w.get("form_description", {})
        out.append({
            "clip_id": clip_id, "event_id": event_id, "channel": channel, "video_id": video_id,
            "X": tokens(" ".join(gf.get(k, "") or "" for k in ACTION_FIELDS)),
            "A": tokens(" ".join(wf.get(k, "") or "" for k in ACTION_FIELDS)),
            "B": tokens(wf.get("non_dominant_hand", "") or ""),
        })
    for row in out:
        row["jA"], row["jB"] = jaccard(row["X"], row["A"]), jaccard(row["X"], row["B"])
        row["cA"], row["cB"] = containment(row["X"], row["A"]), containment(row["X"], row["B"])
    return out


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------

def binom_tail(k, n, p=0.5):
    """P(X >= k) for X ~ Binomial(n, p) -- the one-sided exact sign-test p value."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def cluster_bootstrap_ci(rows, value_key, cluster_key="video_id", replicates=BOOTSTRAP_REPLICATES,
                         seed=PERMUTATION_SEED):
    """Percentile CI for a mean, resampling CLUSTERS (source videos) with replacement.

    Events are unevenly distributed across source videos -- four of the seven non-dominant-hand
    events come from one video -- so an event-level interval would understate the uncertainty.
    This is the same resampling unit the paper's RQ1 bootstrap uses.
    """
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[row[cluster_key]].append(row[value_key])
    clusters = list(by_cluster.values())
    if len(clusters) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        drawn = [v for _ in clusters for v in rng.choice(clusters)]
        means.append(statistics.mean(drawn))
    means.sort()
    lo = means[int(0.025 * (replicates - 1))]
    hi = means[int(0.975 * (replicates - 1))]
    return (lo, hi)


def cluster_bootstrap_diff_ci(rows_a, rows_b, value_key, cluster_key="video_id",
                              replicates=BOOTSTRAP_REPLICATES, seed=PERMUTATION_SEED):
    """CI for mean(rows_a) - mean(rows_b), resampling each group's clusters independently."""
    def draw(rows, rng):
        by_cluster = defaultdict(list)
        for row in rows:
            by_cluster[row[cluster_key]].append(row[value_key])
        clusters = list(by_cluster.values())
        return statistics.mean([v for _ in clusters for v in rng.choice(clusters)])

    rng = random.Random(seed)
    diffs = sorted(draw(rows_a, rng) - draw(rows_b, rng) for _ in range(replicates))
    return (diffs[int(0.025 * (replicates - 1))], diffs[int(0.975 * (replicates - 1))])


def kendall_tau_b(x, y):
    """tau-b with the standard tie correction, plus the share of tied pairs on each side."""
    n = len(x)
    concordant = discordant = tie_x = tie_y = tie_both = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                tie_both += 1
            elif dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    n_pairs = n * (n - 1) / 2
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    tau = (concordant - discordant) / denom if denom else float("nan")
    return tau, (tie_x + tie_both) / n_pairs, (tie_y + tie_both) / n_pairs


def identity_weights(categories):
    return {(a, b): 1.0 if a == b else 0.0 for a in categories for b in categories}


def linear_weights(categories):
    """Ordinal (linear) weights, so a 1-vs-2 disagreement counts as a partial agreement -- the
    same treatment ordinal Krippendorff's alpha gives it, which is what makes the coefficients
    comparable to the alphas the paper already reports."""
    span = max(categories) - min(categories)
    return {(a, b): 1 - abs(a - b) / span for a in categories for b in categories}


def gwet_ac(ratings_a, ratings_b, categories, weights=None):
    """Gwet's AC1 (identity weights) / AC2 (weighted): chance agreement estimated from the
    marginal SPREAD rather than the product of marginals, so a near-unanimous category cannot
    drive expected agreement toward 1 and the coefficient toward 0 -- the paradox the paper
    attributes the low Tier-1 alpha to."""
    n = len(ratings_a)
    if n == 0:
        return float("nan")
    w = weights or identity_weights(categories)
    k = len(categories)
    p_agree = sum(w[(a, b)] for a, b in zip(ratings_a, ratings_b)) / n
    pi = {c: (ratings_a.count(c) + ratings_b.count(c)) / (2 * n) for c in categories}
    t_w = sum(w.values())
    p_chance = (t_w / (k * (k - 1))) * sum(p * (1 - p) for p in pi.values())
    return (p_agree - p_chance) / (1 - p_chance) if p_chance != 1 else float("nan")


def bennett_s(ratings_a, ratings_b, categories, weights=None):
    """Bennett's S: chance agreement fixed by the weight matrix alone, marginal-free by
    construction (1/k in the unweighted case)."""
    n = len(ratings_a)
    if n == 0:
        return float("nan")
    w = weights or identity_weights(categories)
    k = len(categories)
    p_agree = sum(w[(a, b)] for a, b in zip(ratings_a, ratings_b)) / n
    p_chance = sum(w.values()) / (k * k)
    return (p_agree - p_chance) / (1 - p_chance)


def ffprobe_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


# --------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------

def section_length_controls(out, model_dir, events):
    out.append("\n## 1. Is the J(X, A) > J(X, B) ordering a length artifact?\n")
    out.append("\n`B` is a short dedicated field and `X` a full structured description, so "
               "Jaccard's union denominator caps J(X, B) at |B|/|X| even under total containment. "
               "Containment divides by the smaller set instead and has no such ceiling. The "
               "permutation null pairs each X with ANOTHER event's B, which holds the field's "
               "format and register constant and varies only whose non-dominant hand it "
               "describes.\n")
    ndh_by_pair = {}
    for given, withheld, held in CUE_PAIRS:
        rows = triples(model_dir, events, given, withheld)
        by_channel = defaultdict(list)
        for row in rows:
            by_channel[row["channel"]].append(row)
        out.append(f"\n### `{given}` vs `{withheld}` ({held})\n\n")
        out.append("| target_channel | n | J(X,A) | J(X,B) | C(X,A) | C(X,B) | A>B by J | "
                   "A>B by C | mean \\|X\\| | mean \\|A\\| | mean \\|B\\| | J(X,B) ceiling |\n"
                   + "|---" * 12 + "|\n")
        for channel in CHANNELS:
            vals = by_channel.get(channel)
            if not vals:
                continue
            ceiling = statistics.mean(
                min(len(v["B"]), len(v["X"])) / max(len(v["X"]), 1) for v in vals)
            out.append(
                f"| `{channel}` | {len(vals)} | "
                f"{statistics.mean(v['jA'] for v in vals):.3f} | "
                f"{statistics.mean(v['jB'] for v in vals):.3f} | "
                f"{statistics.mean(v['cA'] for v in vals):.3f} | "
                f"{statistics.mean(v['cB'] for v in vals):.3f} | "
                f"{sum(1 for v in vals if v['jA'] > v['jB'])}/{len(vals)} | "
                f"{sum(1 for v in vals if v['cA'] > v['cB'])}/{len(vals)} | "
                f"{statistics.mean(len(v['X']) for v in vals):.1f} | "
                f"{statistics.mean(len(v['A']) for v in vals):.1f} | "
                f"{statistics.mean(len(v['B']) for v in vals):.1f} | {ceiling:.3f} |\n")

        ndh = by_channel["non_dominant_hand"]
        ndh_by_pair[(given, withheld)] = ndh
        n = len(ndh)
        k_j = sum(1 for v in ndh if v["jA"] > v["jB"])
        k_c = sum(1 for v in ndh if v["cA"] > v["cB"])
        out.append(f"\nSign test, non-dominant hand: Jaccard {k_j}/{n}, one-sided exact "
                   f"p = {binom_tail(k_j, n):.4f}; containment {k_c}/{n}, "
                   f"p = {binom_tail(k_c, n):.4f}.\n")

        matched = [v["cB"] for v in ndh]
        mismatched = [containment(ndh[i]["X"], ndh[j]["B"])
                      for i, j in product(range(n), repeat=2) if i != j]
        obs_gap = statistics.mean(v["cA"] - v["cB"] for v in ndh)
        null_gaps = [ndh[i]["cA"] - containment(ndh[i]["X"], ndh[j]["B"])
                     for i, j in product(range(n), repeat=2) if i != j]
        # One-sided in the direction that would REFUTE us: matched B nearer to X than a random B
        # means X does carry item-specific non-dominant-hand content.
        p_perm = (sum(1 for g in null_gaps if g <= obs_gap) + 1) / (len(null_gaps) + 1)
        out.append(f"\nPermutation (matched vs mismatched `B`): mean C(X, B) matched "
                   f"{statistics.mean(matched):.3f} vs mismatched "
                   f"{statistics.mean(mismatched):.3f}; mean gap C(X,A) - C(X,B) observed "
                   f"{obs_gap:.3f} vs null {statistics.mean(null_gaps):.3f}, one-sided "
                   f"p = {p_perm:.4f} for the observed gap being SMALLER than the null. A small "
                   f"p would mean the matched `B` is present in `X` more than an unrelated "
                   f"event's `B` is, i.e. that `X` carries item-specific non-dominant-hand "
                   f"content; a large p means the residual X-B overlap is the generic overlap "
                   f"of two same-format fields.\n")
        out.append("\nPer item, non-dominant-hand targets:\n\n")
        out.append("| item | video | J(X,A) | J(X,B) | C(X,A) | C(X,B) | \\|X\\| | \\|A\\| | "
                   "\\|B\\| |\n" + "|---" * 9 + "|\n")
        for v in ndh:
            out.append(f"| {v['clip_id']}/{v['event_id']} | {v['video_id']} | {v['jA']:.3f} | "
                       f"{v['jB']:.3f} | {v['cA']:.3f} | {v['cB']:.3f} | {len(v['X'])} | "
                       f"{len(v['A'])} | {len(v['B'])} |\n")
    return ndh_by_pair


def section_jxa_intervals(out, model_dir, events):
    out.append("\n## 2. J(X, A) three-way comparison with clustered intervals\n")
    out.append("\nBoth sides of J(X, A) are full structured descriptions of comparable length, so "
               "this comparison is unaffected by section 1's objection. `manual` is the "
               "no-switch anchor (the cue names what the model selects anyway) and `face` the "
               "redirection anchor (the cue demonstrably moves content across the "
               "manual/nonmanual boundary). Intervals resample SOURCE VIDEOS.\n")
    for given, withheld, held in CUE_PAIRS:
        rows = triples(model_dir, events, given, withheld)
        by_channel = defaultdict(list)
        for row in rows:
            by_channel[row["channel"]].append(row)
        out.append(f"\n### `{given}` vs `{withheld}` ({held})\n\n")
        out.append("| target_channel | n | videos | J(X,A) | 95% CI (video-clustered) |\n"
                   + "|---" * 5 + "|\n")
        for channel in CHANNELS:
            vals = by_channel.get(channel)
            if not vals:
                continue
            lo, hi = cluster_bootstrap_ci(vals, "jA")
            out.append(f"| `{channel}` | {len(vals)} | "
                       f"{len({v['video_id'] for v in vals})} | "
                       f"{statistics.mean(v['jA'] for v in vals):.3f} | "
                       f"[{lo:.3f}, {hi:.3f}] |\n")
        ndh, manual, face = (by_channel["non_dominant_hand"], by_channel["manual"],
                             by_channel["face"])
        for label, other in (("manual", manual), ("face", face)):
            lo, hi = cluster_bootstrap_diff_ci(ndh, other, "jA")
            diff = statistics.mean(v["jA"] for v in ndh) - statistics.mean(v["jA"] for v in other)
            out.append(f"\n- non-dominant hand minus {label}: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]\n")


def section_ties(out, results_dir):
    out.append("\n## 3. Kendall's tau-b and the tie load behind each Spearman rho\n")
    out.append("\nThe model's gated D1 takes two distinct values, so a large share of item pairs "
               "are tied on the model side and rho's assumption of few ties fails. tau-b applies "
               "the standard tie correction; the tie fractions say how much correction is being "
               "asked for.\n\n")
    out.append("| score | cues | n | rho | tau-b | tied pairs (model) | tied pairs (ref.) |\n"
               + "|---" * 7 + "|\n")
    from scipy.stats import spearmanr
    for label, col in SCORE_COLS:
        for mode in MODES:
            path = results_dir / f"attribution_scores_events_{mode}_checked.csv"
            if not path.exists():
                continue
            human, model = [], []
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    h, m = (row.get(f"human_{col}") or "").strip(), \
                           (row.get(f"model_{col}") or "").strip()
                    if h and m:
                        human.append(float(h))
                        model.append(float(m))
            if len(human) < 3:
                continue
            if len(set(model)) < 2 or len(set(human)) < 2:
                out.append(f"| {label} | {MODE_LABEL[mode]} | {len(human)} | "
                           f"not testable (no variance) | | | |\n")
                continue
            rho = spearmanr(human, model).statistic
            tau, tie_model, tie_ref = kendall_tau_b(model, human)
            out.append(f"| {label} | {MODE_LABEL[mode]} | {len(human)} | {rho:+.2f} | "
                       f"{tau:+.2f} | {tie_model:.0%} | {tie_ref:.0%} |\n")


def section_agreement(out, ref_dir, results_dir):
    out.append("\n## 4. Tier-1 agreement under marginal-free chance corrections\n")
    out.append("\nKrippendorff's alpha divides by an expected-agreement term computed from the "
               "observed marginals, which collapses when one category dominates. Gwet's AC1 and "
               "Bennett's S do not, so they say whether the low alpha reflects rater "
               "disagreement or the marginal distribution.\n\n")
    out.append("| cue cell | n items | raw agreement | within 1 point | AC1 | AC2 (ordinal) | "
               "S | S (ordinal) |\n" + "|---" * 8 + "|\n")
    for mode in MODES:
        path = results_dir / f"perception_scores_events_combined_{mode}.csv"
        if not path.exists():
            continue
        by_item = defaultdict(dict)
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                score = (row.get("perception_score") or "").strip()
                if score:
                    by_item[(row["clip_id"], row["event_id"])][row["rater_id"]] = float(score)
        pairs = [(v["A01"], v["A02"]) for v in by_item.values() if "A01" in v and "A02" in v]
        if not pairs:
            continue
        a = [int(p[0]) for p in pairs]
        b = [int(p[1]) for p in pairs]
        raw = sum(1 for x, y in zip(a, b) if x == y) / len(pairs)
        within1 = sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(pairs)
        cats = [0, 1, 2]
        lin = linear_weights(cats)
        out.append(f"| `{mode}` | {len(pairs)} | {raw:.3f} | {within1:.3f} | "
                   f"{gwet_ac(a, b, cats):.3f} | {gwet_ac(a, b, cats, lin):.3f} | "
                   f"{bennett_s(a, b, cats):.3f} | {bennett_s(a, b, cats, lin):.3f} |\n")
    out.append("\nMarginal distribution of the two raters' Tier-1 scores per cell "
               "(what the correction is reacting to):\n\n")
    out.append("| cue cell | rater | 0 | 1 | 2 |\n" + "|---" * 5 + "|\n")
    for mode in MODES:
        path = results_dir / f"perception_scores_events_combined_{mode}.csv"
        if not path.exists():
            continue
        counts = defaultdict(Counter)
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                score = (row.get("perception_score") or "").strip()
                if score:
                    counts[row["rater_id"]][int(float(score))] += 1
        for rater in sorted(counts):
            c = counts[rater]
            out.append(f"| `{mode}` | {rater} | {c[0]} | {c[1]} | {c[2]} |\n")


def section_degradation(out, model_dir, events, subdir="visual_null", mode="grounded"):
    out.append("\n## 5. Stated confidence and D1 survival across the degradation ladder\n")
    out.append("\nThe paper reports confidence for `none` (0.89) and `black` (0.36) only. The "
               "claim that the model registers an ABSENT signal but not a BROKEN one needs the "
               "`static` and `shuffled` figures, and the D1-survival counts need splitting by "
               "articulator, since face targets are the least motion-dependent items in the "
               "set.\n")
    null_dir = model_dir / subdir
    channel_of = {(c, e): ch for c, e, ch, _ in events}
    real, degraded = {}, defaultdict(dict)
    for clip_id, event_id, _, _ in events:
        run = load_run(model_dir, clip_id, event_id, mode)
        if run is not None:
            real[(clip_id, event_id)] = run
    for path in sorted(null_dir.glob(f"*__combined__*__run1__{mode}.json")):
        m = re.match(r"(?P<clip>[^_]+)__(?P<event>event\d)__combined__(?P<visual>[a-z]+)__"
                     r"run1__(?P<mode>\w+)\.json", path.name)
        if not m:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        cands = (data.get("parsed") or {}).get("candidate_actions") or []
        degraded[m.group("visual")][(m.group("clip"), m.group("event"))] = cands[0] if cands \
            else None

    out.append("\n\n| visual | n | mean stated confidence | D1 same as real | "
               "D1 same, manual | D1 same, face | D1 same, non-dom. |\n" + "|---" * 7 + "|\n")
    for visual in VISUAL_CONDITIONS:
        items = degraded.get(visual) if visual != "real" else real
        if not items:
            continue
        confs, same, per_channel = [], 0, defaultdict(lambda: [0, 0])
        n = 0
        for key, run in items.items():
            if key not in real:
                continue
            n += 1
            conf = (run or {}).get("confidence")
            if isinstance(conf, (int, float)):
                confs.append(float(conf))
            rv = (real[key] or {}).get("semantic_syntactic_load")
            dv = (run or {}).get("semantic_syntactic_load")
            channel = channel_of.get(key, "?")
            per_channel[channel][1] += 1
            if isinstance(rv, (int, float)) and isinstance(dv, (int, float)) and rv == dv:
                same += 1
                per_channel[channel][0] += 1
        cell = lambda ch: (f"{per_channel[ch][0]}/{per_channel[ch][1]}"
                           if per_channel[ch][1] else "--")
        out.append(f"| `{visual}` | {n} | "
                   f"{statistics.mean(confs):.2f} |" .replace("nan", "--")
                   if confs else f"| `{visual}` | {n} | -- |")
        out.append(f" {same}/{n} | {cell('manual')} | {cell('face')} | "
                   f"{cell('non_dominant_hand')} |\n")


def section_frames(out, cfg, events, manifest_path):
    out.append("\n## 6. How many frames does the shuffled condition actually shuffle?\n")
    out.append("\nBoth the target clip and the context clip are degraded and both are sent, so "
               "the shuffled input is the sum of the two. Frames consumed = clip duration times "
               "the configured sampling rate.\n")
    fps = cfg.get("model", {}).get("fps") or cfg.get("inference", {}).get("fps")
    c_clips_dir = cfg["paths"]["c_clips_dir"]
    target_durations, context_durations = [], []
    seen_context = set()
    for clip_id, event_id, _, _ in events:
        kind = "ev1" if event_id == "event1" else "ev2"
        d = ffprobe_duration(c_clips_dir / f"{clip_id}_{kind}_model.mp4")
        if d:
            target_durations.append(d)
        if clip_id not in seen_context:
            seen_context.add(clip_id)
            d = ffprobe_duration(c_clips_dir / f"{clip_id}_window_model.mp4")
            if d:
                context_durations.append(d)
    if not target_durations:
        out.append("\n(model-input clips not found; skipping)\n")
        return
    fps = float(fps or 16)
    t_mean, c_mean = statistics.mean(target_durations), statistics.mean(context_durations)
    out.append(f"\n- sampling rate: {fps:g} fps\n")
    out.append(f"- padded target clip: mean {t_mean:.2f} s "
               f"(range {min(target_durations):.2f}-{max(target_durations):.2f}) "
               f"-> mean {t_mean * fps:.0f} frames "
               f"(range {min(target_durations) * fps:.0f}-{max(target_durations) * fps:.0f})\n")
    out.append(f"- context window clip: mean {c_mean:.2f} s "
               f"(range {min(context_durations):.2f}-{max(context_durations):.2f}) "
               f"-> mean {c_mean * fps:.0f} frames "
               f"(range {min(context_durations) * fps:.0f}-{max(context_durations) * fps:.0f})\n")
    out.append(f"- both clips together: mean {(t_mean + c_mean) * fps:.0f} frames per item\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = cfg["paths"]["model_outputs_dir"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    ref_dir = cfg["paths"]["human_annotations_csv"].parent
    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
    events = load_events(manifest_path)

    out = ["# Length, tie, agreement, degradation and frame-count checks\n",
           f"\nModel outputs `{model_dir.name}`, clip kind `combined`, {len(events)} events. "
           "Recomputed from files already in the repository -- no new inference and no new human "
           "rating. Generated by `scripts/25_analyze_length_and_tie_checks.py`.\n"]

    section_length_controls(out, model_dir, events)
    section_jxa_intervals(out, model_dir, events)
    section_ties(out, results_dir)
    section_agreement(out, ref_dir, results_dir)
    section_degradation(out, model_dir, events)
    section_frames(out, cfg, events, manifest_path)

    path = results_dir / "length_and_tie_checks.md"
    path.write_text("".join(out), encoding="utf-8")
    log.info("Wrote %s", path)
    print("".join(out))


if __name__ == "__main__":
    main()
