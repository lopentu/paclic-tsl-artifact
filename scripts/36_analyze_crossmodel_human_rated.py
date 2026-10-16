#!/usr/bin/env python3
"""Human-rated cross-model comparison: Qwen3-VL-8B vs GLM-4.6V-Flash on
the two cue cells both models had Tier-1 rated -- `grounded` and `ablate_channel` (-articulator).

Nothing here sends a prompt to a model. Every statistic is the one the Qwen tables already use,
taken from the same code so the GLM column cannot drift from the Qwen column:

  - Wilson interval, exact McNemar, video-clustered bootstrap of the paired difference (10,000
    replicates, seed 20260803) and Benjamini-Hochberg: imported from
    18_analyze_robustness_checks.py.
  - Kendall tau-b, Gwet AC1/AC2, Bennett S, linear weights: imported from
    25_analyze_length_and_tie_checks.py.
  - Krippendorff alpha (ordinal) and linear weighted kappa: lib/stats_utils.py.
  - The `art.` leave-one-out articulator-mean predictor and `det.` (share of outputs equal to the
    modal value for their target articulator): the definitions in analyze_attribution_summary.py
    section 6, applied here to all five scores rather than D1/D2 only.

Two subcommands:

  split   The round-2 raters filled one merged sheet each
          (data/reference_annotations/perception_round2/perception_rating_sheet_<rater>_combined_
          merged.csv, 110 rows, `experiment_type` = cue cell, plus an `uncapped_score` column).
          09b's `aggregate` reads one directory per cue cell with its own fixed column set, so the
          merged sheets are split -- read-only -- into
          perception_round2/filled_glm46v_flash/<cell>/<rater>.csv, and the uncapped column is
          written to filled_glm46v_flash/uncapped_scores_glm46v_flash.csv in the same long layout
          as the Qwen round's uncapped_scores_raw.csv.

  report  Reads 09b's aggregated sheets for both models, the uncapped scores, the adjudicated
          reference (events_checked.csv) and the model outputs, and writes
          data/results/glm46v_flash/human_rated_crossmodel_combined.md.

Usage:
    python scripts/36_analyze_crossmodel_human_rated.py split
    python scripts/36_analyze_crossmodel_human_rated.py report
"""

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import get_logger, ensure_dir, REPO_ROOT, parse_manifest_event
from lib.stats_utils import krippendorff_alpha_ordinal, weighted_cohen_kappa

log = get_logger("analyze_crossmodel_human_rated")


def _import(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S18 = _import("s18", "18_analyze_robustness_checks.py")
S25 = _import("s25", "25_analyze_length_and_tie_checks.py")

CELLS = ["grounded", "ablate_channel"]
ALL_MODES = ["grounded", "ablate_translation", "ablate_channel", "ablate_both"]
CELL_LABEL = {"grounded": "grounded", "ablate_translation": "-translation",
              "ablate_channel": "-articulator", "ablate_both": "-both"}
RATERS = ["A01", "A02"]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]
SCORES = [("D1", "semantic_syntactic_load"),
          ("D2", "pragmatic_depictive_affective_enrichment"),
          ("description", "description"), ("indication", "indication"),
          ("depiction", "depiction")]
ADEQUATE_MIN = 2

ANNOT = REPO_ROOT / "data" / "reference_annotations"
ROUND2 = ANNOT / "perception_round2"
FILLED_GLM = ROUND2 / "filled_glm46v_flash"
UNCAPPED_GLM = FILLED_GLM / "uncapped_scores_glm46v_flash.csv"
UNCAPPED_QWEN = ANNOT / "uncapped_scores_raw.csv"
REFERENCE = ANNOT / "events_checked.csv"
MANIFEST = REPO_ROOT / "clip_manifest.csv"

MODELS = {
    "Qwen": {"outputs": REPO_ROOT / "data" / "model_outputs" / "qwen3vl_8b",
             "results": REPO_ROOT / "data" / "results" / "qwen3vl_8b", "uncapped": UNCAPPED_QWEN},
    "GLM": {"outputs": REPO_ROOT / "data" / "model_outputs" / "glm46v_flash",
            "results": REPO_ROOT / "data" / "results" / "glm46v_flash", "uncapped": UNCAPPED_GLM},
}
OUT = REPO_ROOT / "data" / "results" / "glm46v_flash" / "human_rated_crossmodel_combined.md"

RATING_COLUMNS = ["clip_id", "event_id", "clip_kind", "run_index", "target_channel",
                  "channel_judged", "model_hands", "model_handshape", "model_movement",
                  "model_location", "model_nonmanual", "model_non_dominant_hand",
                  "model_temporal_alignment", "rater_id", "perception_score", "notes"]


# ------------------------------------------------------------------------------------------
# split

def split():
    long_rows = []
    for rater in RATERS:
        path = ROUND2 / f"perception_rating_sheet_{rater}_combined_merged.csv"
        with open(path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for cell in CELLS:
            sub = [r for r in rows if r["experiment_type"] == cell]
            if len(sub) != 55:
                raise SystemExit(f"{path.name}: {len(sub)} rows for {cell}, expected 55")
            out_dir = ensure_dir(FILLED_GLM / cell)
            with open(out_dir / f"{rater}.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RATING_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(sub)
            log.info("%s %s: %d rows -> %s", rater, cell, len(sub), out_dir / f"{rater}.csv")
        for r in rows:
            if r["uncapped_score"].strip().upper() == "N/A":
                continue
            long_rows.append({"rater_id": r["rater_id"], "clip_id": r["clip_id"],
                              "event_id": r["event_id"], "mode": r["experiment_type"],
                              "capped_score": r["perception_score"].strip(),
                              "uncapped_score": r["uncapped_score"].strip()})
    with open(UNCAPPED_GLM, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rater_id", "clip_id", "event_id", "mode",
                                               "capped_score", "uncapped_score"])
        writer.writeheader()
        writer.writerows(long_rows)
    log.info("%d uncapped rater-rows -> %s", len(long_rows), UNCAPPED_GLM)


# ------------------------------------------------------------------------------------------
# loading

def load_events():
    """{(clip, event): (target_channel, video_id)}"""
    out = {}
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("status") or "").strip().lower() == "excluded":
                continue
            for i in (1, 2):
                parsed = parse_manifest_event(row, i)
                if parsed is not None:
                    out[(row["clip_id"], f"event{i}")] = (parsed[2], row["video_id"])
    return out


def load_ratings(results_dir, mode):
    """{key: {rater: score}}, {key: channel_judged on the sheet}"""
    path = results_dir / f"perception_scores_events_combined_{mode}.csv"
    scores, judged = defaultdict(dict), {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("perception_score") or "").strip() == "":
                continue
            key = (r["clip_id"], r["event_id"])
            scores[key][r["rater_id"]] = int(float(r["perception_score"]))
            judged[key] = r["channel_judged"]
    return scores, judged


def load_uncapped(path):
    """{(mode, key): {rater: (capped, uncapped)}}"""
    out = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["mode"], (r["clip_id"], r["event_id"]))][r["rater_id"]] = (
                int(r["capped_score"]), int(r["uncapped_score"]))
    return out


def load_reference():
    """{key: {col: mean across annotators}}"""
    vals = defaultdict(lambda: defaultdict(list))
    with open(REFERENCE, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["clip_id"], r["event_id"])
            for _, col in SCORES:
                raw = (r.get(col) or "").strip()
                if raw not in ("", "N/A", "n/a"):
                    vals[key][col].append(float(raw))
    return {k: {c: statistics.mean(v) for c, v in d.items()} for k, d in vals.items()}


def load_outputs(outputs_dir, mode, events):
    out = {}
    for clip_id, event_id in events:
        path = outputs_dir / f"{clip_id}__{event_id}__combined__run1__{mode}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        cands = (data.get("parsed") or {}).get("candidate_actions") or []
        if cands:
            out[(clip_id, event_id)] = cands[0]
    return out


# ------------------------------------------------------------------------------------------
# statistics helpers

def adequate_map(scores):
    return {k: statistics.mean(v.values()) >= ADEQUATE_MIN for k, v in scores.items()
            if len(v) == 2}


def substitute_uncapped(scores, uncapped, mode):
    out = {}
    for k, v in scores.items():
        new = dict(v)
        for rater, (capped, unc) in uncapped.get((mode, k), {}).items():
            if new.get(rater) != capped:
                raise SystemExit(f"capped score mismatch for {mode} {k} {rater}: sheet "
                                 f"{new.get(rater)} vs uncapped file {capped}")
            new[rater] = unc
        out[k] = new
    return out


def wilson_txt(k, n):
    lo, hi = S18.wilson_ci(k, n)
    return f"{k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]"


def mcnemar_block(adeq_a, adeq_b, video_of):
    keys = sorted(set(adeq_a) & set(adeq_b))
    pairs = [(adeq_a[k], adeq_b[k]) for k in keys]
    b01, b10, p = S18.mcnemar_exact(pairs)
    diff = (sum(a for a, _ in pairs) - sum(b for _, b in pairs)) / len(pairs)
    lo, hi = S18.cluster_bootstrap_diff([(video_of[k], adeq_a[k], adeq_b[k]) for k in keys])
    return b01, b10, p, diff, lo, hi


def did_bootstrap(items, n_boot=10000, seed=20260803):
    """Video-clustered bootstrap of (A1 - B1) - (A2 - B2) over the same events, resampling
    exactly as 18.cluster_bootstrap_diff does. items: [(video, a1, b1, a2, b2)]."""
    by_cluster = defaultdict(list)
    for video, *vals in items:
        by_cluster[video].append(vals)
    clusters = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        rows = [row for i in picked for row in by_cluster[clusters[i]]]
        n = len(rows)
        d1 = (sum(r[0] for r in rows) - sum(r[1] for r in rows)) / n
        d2 = (sum(r[2] for r in rows) - sum(r[3] for r in rows)) / n
        diffs.append(d1 - d2)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def agreement(scores):
    keys = sorted(k for k, v in scores.items() if len(v) == 2)
    a = [scores[k]["A01"] for k in keys]
    b = [scores[k]["A02"] for k in keys]
    cats = [0, 1, 2]
    lin = S25.linear_weights(cats)
    n = len(keys)
    out = {"n": n,
           "exact": sum(1 for x, y in zip(a, b) if x == y) / n,
           "within1": sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / n,
           "ac1": S25.gwet_ac(a, b, cats), "ac2": S25.gwet_ac(a, b, cats, lin),
           "s": S25.bennett_s(a, b, cats), "s_lin": S25.bennett_s(a, b, cats, lin),
           "marg_a": Counter(a), "marg_b": Counter(b),
           "mean_a": statistics.mean(a), "mean_b": statistics.mean(b)}
    if len(set(a) | set(b)) >= 2:
        out["alpha"] = float(krippendorff_alpha_ordinal([a, b]))
        try:
            out["kappa"] = float(weighted_cohen_kappa(a, b))
        except Exception:
            out["kappa"] = float("nan")
    else:
        out["alpha"] = out["kappa"] = float("nan")
    return out


def tier2_rows(adequate, outputs, ref, events):
    rows = []
    for k in sorted(adequate):
        if not adequate[k] or k not in outputs or k not in ref:
            continue
        rows.append((k, events[k][0], ref[k], outputs[k]))
    return rows


def score_stats(rows, col):
    items = [(ch, r[col], float(o[col])) for _, ch, r, o in rows
             if col in r and isinstance(o.get(col), (int, float))]
    n = len(items)
    if n == 0:
        return None
    h = [x[1] for x in items]
    m = [x[2] for x in items]
    loo = []
    for i, (ch, _, _) in enumerate(items):
        others = [x[1] for j, x in enumerate(items) if j != i and x[0] == ch]
        loo.append(statistics.mean(others) if others else
                   statistics.mean([x[1] for j, x in enumerate(items) if j != i]))
    by_ch = defaultdict(list)
    for ch, _, mv in items:
        by_ch[ch].append(mv)
    det = sum(Counter(v).most_common(1)[0][1] for v in by_ch.values()) / n
    loo_rho = (float(spearmanr(h, loo)[0]) if n >= 3 and len(set(h)) >= 2 and
               len(set(loo)) >= 2 else None)
    testable = n >= 3 and len(set(h)) >= 2 and len(set(m)) >= 2
    if testable:
        rho, p = spearmanr(h, m)
        tau, tie_m, tie_h = S25.kendall_tau_b(m, h)
    else:
        rho = p = tau = tie_m = tie_h = None
    return {"n": n, "mae": float(np.mean(np.abs(np.array(h) - np.array(m)))),
            "art": float(np.mean(np.abs(np.array(h) - np.array(loo)))), "art_rho": loo_rho,
            "rho": None if rho is None else float(rho), "p": None if p is None else float(p),
            "tau": tau, "tie_m": tie_m, "tie_h": tie_h,
            "half": sum(1 for x, y in zip(h, m) if abs(x - y) <= 0.5), "det": det,
            "bias": statistics.mean(y - x for x, y in zip(h, m)),
            "mdist": dict(sorted(Counter(m).items())), "hdist": dict(sorted(Counter(h).items())),
            "testable": testable}


def fmt(x, spec="{:.2f}"):
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    return spec.format(x)


def dist(c):
    return "{" + ", ".join(f"{k:g}: {v}" for k, v in sorted(c.items())) + "}"


# ------------------------------------------------------------------------------------------
# report

def report():
    events = load_events()
    video_of = {k: v[1] for k, v in events.items()}
    channel_of = {k: v[0] for k, v in events.items()}
    ref = load_reference()

    data = {}
    for name, cfg in MODELS.items():
        d = {"scores": {}, "judged": {}, "outputs": {}}
        uncapped = load_uncapped(cfg["uncapped"])
        modes = ALL_MODES if name == "Qwen" else CELLS
        for mode in modes:
            d["scores"][mode], d["judged"][mode] = load_ratings(cfg["results"], mode)
            d["outputs"][mode] = load_outputs(cfg["outputs"], mode, events)
        d["uncapped_raw"] = uncapped
        d["unc_scores"] = {m: substitute_uncapped(d["scores"][m], uncapped, m) for m in CELLS}
        d["adeq"] = {m: adequate_map(d["scores"][m]) for m in d["scores"]}
        d["adeq_unc"] = {m: adequate_map(d["unc_scores"][m]) for m in CELLS}
        data[name] = d

    L = ["# Human-rated cross-model comparison: Qwen3-VL-8B vs GLM-4.6V-Flash\n\n",
         "Cue cells `grounded` and `ablate_channel` (-articulator) only -- the two cells rated "
         "for both models. Same two raters (A01, A02), same rubric (Step-1 cap included in "
         "`perception_score`), same gate (adequate = both raters give 2), reference = "
         "`events_checked.csv`. Generated by `scripts/36_analyze_crossmodel_human_rated.py`; no "
         "inference, no new rating.\n\n",
         "Qwen inputs: `data/results/qwen3vl_8b/perception_scores_events_combined_<cell>.csv`, "
         "`data/reference_annotations/uncapped_scores_raw.csv`. GLM inputs: "
         "`data/results/glm46v_flash/perception_scores_events_combined_<cell>.csv` (09b "
         "aggregate of `perception_round2/filled_glm46v_flash/`), "
         "`perception_round2/filled_glm46v_flash/uncapped_scores_glm46v_flash.csv`.\n"]

    # --- 1. RQ1 adequacy ------------------------------------------------------------------
    L.append("\n## 1. RQ1: Tier-1 adequacy per cell\n\n")
    L.append("| model | cell | mean rating | per-rating adequate | adequate items (95% Wilson) "
             "| channel_judged != target (events) |\n|---|---|---|---|---|---|\n")
    for name, d in data.items():
        for mode in CELLS:
            sc = d["scores"][mode]
            allr = [s for v in sc.values() for s in v.values()]
            adq = d["adeq"][mode]
            k = sum(adq.values())
            mism = sum(1 for key in sc if d["judged"][mode][key] != channel_of[key])
            L.append(f"| {name} | `{mode}` | {statistics.mean(allr):.2f} | "
                     f"{sum(1 for s in allr if s >= 2)}/{len(allr)} | {wilson_txt(k, len(adq))} "
                     f"| {mism} |\n")

    L.append("\n### Paired contrast: articulator cue (`grounded` vs `ablate_channel`)\n\n")
    L.append("| model | scoring | in (grounded only) | out (-artic. only) | exact McNemar p | "
             "diff | 95% video-clustered bootstrap |\n|---|---|---|---|---|---|---|\n")
    for name, d in data.items():
        for label, a, b in [("capped (pre-specified rubric)", d["adeq"]["grounded"],
                             d["adeq"]["ablate_channel"]),
                            ("uncapped in both cells", d["adeq_unc"]["grounded"],
                             d["adeq_unc"]["ablate_channel"]),
                            ("grounded capped vs -artic. uncapped", d["adeq"]["grounded"],
                             d["adeq_unc"]["ablate_channel"])]:
            b01, b10, p, diff, lo, hi = mcnemar_block(a, b, video_of)
            L.append(f"| {name} | {label} | {b01} | {b10} | {p:.4f} | {diff:+.3f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] |\n")
    q = data["Qwen"]
    L.append("\nQwen's other two contrasts (paper Table 2 text), recomputed here as a check: ")
    for label, a, b in [("articulator cue, translation withheld", "ablate_translation",
                         "ablate_both"), ("translation cue", "grounded", "ablate_translation")]:
        b01, b10, p, diff, lo, hi = mcnemar_block(q["adeq"][a], q["adeq"][b], video_of)
        L.append(f"{label} {b01} in / {b10} out, p = {p:.4f}, {diff:+.3f} "
                 f"[{lo:+.3f}, {hi:+.3f}]; ")
    L.append("\n")

    L.append("\n### Between-model contrasts (post-hoc; same 55 items, same raters)\n\n")
    items = [(video_of[k], data["Qwen"]["adeq"]["grounded"][k],
              data["Qwen"]["adeq"]["ablate_channel"][k], data["GLM"]["adeq"]["grounded"][k],
              data["GLM"]["adeq"]["ablate_channel"][k]) for k in sorted(events)]
    did = (sum(r[1] - r[2] for r in items) - sum(r[3] - r[4] for r in items)) / len(items)
    lo, hi = did_bootstrap(items)
    L.append(f"- Difference in the articulator-cue effect (Qwen effect minus GLM effect, capped): "
             f"{did:+.3f}, 95% video-clustered bootstrap [{lo:+.3f}, {hi:+.3f}].\n")
    items_u = [(video_of[k], data["Qwen"]["adeq_unc"]["grounded"][k],
                data["Qwen"]["adeq_unc"]["ablate_channel"][k],
                data["GLM"]["adeq_unc"]["grounded"][k],
                data["GLM"]["adeq_unc"]["ablate_channel"][k]) for k in sorted(events)]
    did_u = (sum(r[1] - r[2] for r in items_u) - sum(r[3] - r[4] for r in items_u)) / len(items_u)
    lo_u, hi_u = did_bootstrap(items_u)
    L.append(f"- Same, uncapped in both cells of both models: {did_u:+.3f} "
             f"[{lo_u:+.3f}, {hi_u:+.3f}].\n")
    for mode in CELLS:
        b01, b10, p, diff, lo, hi = mcnemar_block(data["Qwen"]["adeq"][mode],
                                                  data["GLM"]["adeq"][mode], video_of)
        L.append(f"- `{mode}` adequacy, Qwen vs GLM (paired by item): Qwen-only {b01}, "
                 f"GLM-only {b10}, exact McNemar p = {p:.4f}, diff {diff:+.3f} "
                 f"[{lo:+.3f}, {hi:+.3f}].\n")

    L.append("\n### Mean Tier-1 rating and adequate count by target articulator\n\n")
    L.append("| model | target | n | grounded mean | -artic. mean | grounded adequate | "
             "-artic. adequate |\n|---|---|---|---|---|---|---|\n")
    for name, d in data.items():
        for ch in CHANNELS:
            keys = [k for k in events if channel_of[k] == ch]
            cells = []
            for mode in CELLS:
                vals = [s for k in keys for s in d["scores"][mode][k].values()]
                cells.append(f"{statistics.mean(vals):.2f}")
            adq = [f"{sum(d['adeq'][m][k] for k in keys)}/{len(keys)}" for m in CELLS]
            L.append(f"| {name} | `{ch}` | {len(keys)} | {cells[0]} | {cells[1]} | {adq[0]} | "
                     f"{adq[1]} |\n")

    L.append("\n### Rubric cap exposure (Step 1: channel_judged != target_channel caps at <= 1)\n\n")
    L.append("| model | cell | cap-reachable events (by target) | adequacy, all | adequacy, "
             "cap-unreachable events only | mean rating, matched / mismatched rows |\n"
             "|---|---|---|---|---|---|\n")
    for name, d in data.items():
        for mode in CELLS:
            adq = d["adeq"][mode]
            mism = [k for k in adq if d["judged"][mode][k] != channel_of[k]]
            un = [k for k in adq if k not in set(mism)]
            by = Counter(channel_of[k] for k in mism)
            m_vals = [s for k in un for s in d["scores"][mode][k].values()]
            x_vals = [s for k in mism for s in d["scores"][mode][k].values()]
            L.append(f"| {name} | `{mode}` | {len(mism)} {dict(by)} | {sum(adq.values())}/55 | "
                     f"{sum(adq[k] for k in un)}/{len(un)} | "
                     f"{statistics.mean(m_vals):.2f} / "
                     f"{fmt(statistics.mean(x_vals) if x_vals else None)} |\n")
    gl = data["GLM"]
    g_mism = sorted(k for k in gl["adeq"]["grounded"]
                    if gl["judged"]["grounded"][k] != channel_of[k])
    L.append("\nGLM `grounded` label-mismatch events (model rejected the given articulator): "
             + ", ".join(f"{c}/{e} ({channel_of[(c, e)]}->{gl['judged']['grounded'][(c, e)]}; "
                         f"capped {gl['scores']['grounded'][(c, e)]['A01']}/"
                         f"{gl['scores']['grounded'][(c, e)]['A02']}, uncapped "
                         f"{gl['unc_scores']['grounded'][(c, e)]['A01']}/"
                         f"{gl['unc_scores']['grounded'][(c, e)]['A02']})" for c, e in g_mism)
             + ".\n")

    L.append("\n### Discordant items (capped)\n\n")
    for name, d in data.items():
        a, b = d["adeq"]["grounded"], d["adeq"]["ablate_channel"]
        into = sorted(k for k in a if a[k] and not b[k])
        out_ = sorted(k for k in a if b[k] and not a[k])
        L.append(f"- {name} in ({len(into)}): " + ", ".join(
            f"{c}/{e}[{channel_of[(c, e)]}]" for c, e in into) + "\n")
        L.append(f"- {name} out ({len(out_)}): " + ", ".join(
            f"{c}/{e}[{channel_of[(c, e)]}]" for c, e in out_) + "\n")

    # --- 2. Uncapped ----------------------------------------------------------------------
    L.append("\n## 2. Uncapped re-score (Step 1 suspended, core fields chosen by "
             "channel_judged)\n\n")
    L.append("Qwen: collected unblinded in a separate pass (2026-08-06), withheld cells only. "
             "GLM: collected in the same pass as the capped score, on every mismatched row "
             "including `grounded`; `target_channel` and the capped score were visible on the row "
             "in both rounds.\n\n")
    L.append("| model | cell | mismatched events | adequate capped | adequate uncapped | "
             "mismatched events clearing the gate uncapped | mean rating uncapped |\n"
             "|---|---|---|---|---|---|---|\n")
    for name, d in data.items():
        for mode in CELLS:
            cap, unc = d["adeq"][mode], d["adeq_unc"][mode]
            mism = [k for k in cap if d["judged"][mode][k] != channel_of[k]]
            gained = sorted(k for k in mism if unc[k] and not cap[k])
            allr = [s for v in d["unc_scores"][mode].values() for s in v.values()]
            L.append(f"| {name} | `{mode}` | {len(mism)} | {wilson_txt(sum(cap.values()), 55)} "
                     f"| {wilson_txt(sum(unc.values()), 55)} | {len(gained)}/{len(mism)} "
                     f"({', '.join(f'{c}/{e}' for c, e in gained)}) | "
                     f"{statistics.mean(allr):.2f} |\n")

    L.append("\n### Anchoring diagnostic: uncapped vs capped, per rater\n\n")
    L.append("| model | cell | rater | rows | equal | up | down |\n|---|---|---|---|---|---|---|\n")
    for name, d in data.items():
        for mode in CELLS:
            for rater in RATERS:
                sub = [v[rater] for (m, _), v in d["uncapped_raw"].items()
                       if m == mode and rater in v]
                if not sub:
                    continue
                L.append(f"| {name} | `{mode}` | {rater} | {len(sub)} | "
                         f"{sum(1 for c, u in sub if u == c)} | "
                         f"{sum(1 for c, u in sub if u > c)} | "
                         f"{sum(1 for c, u in sub if u < c)} |\n")
    L.append("\nQwen's paper figure (31/45 exact, 45/45 within one) pools both withheld cells "
             "(`ablate_channel` + `ablate_both`); the `ablate_channel` subset is shown below "
             "for like-for-like comparison.\n")

    L.append("\n### Agreement on the uncapped score, mismatched rows only\n\n")
    L.append("| model | rows | exact | within 1 | ordinal alpha |\n|---|---|---|---|---|\n")
    for name, d in data.items():
        for label, modes in [("ablate_channel", ["ablate_channel"]),
                             ("all rated mismatched rows", None)]:
            pairs = [(v["A01"][1], v["A02"][1]) for (m, _), v in d["uncapped_raw"].items()
                     if (modes is None or m in modes) and "A01" in v and "A02" in v]
            if not pairs:
                continue
            a = [x for x, _ in pairs]
            b = [y for _, y in pairs]
            alpha = float(krippendorff_alpha_ordinal([a, b])) if len(set(a) | set(b)) > 1 \
                else float("nan")
            L.append(f"| {name} ({label}) | {len(pairs)} | "
                     f"{sum(1 for x, y in pairs if x == y)}/{len(pairs)} | "
                     f"{sum(1 for x, y in pairs if abs(x - y) <= 1)}/{len(pairs)} | "
                     f"{fmt(alpha, '{:.3f}')} |\n")

    # --- 3. Agreement ---------------------------------------------------------------------
    L.append("\n## 3. Tier-1 inter-rater agreement (A01 vs A02, 55 items per cell)\n\n")
    L.append("| model | scoring | cell | ordinal alpha | weighted kappa | exact | within 1 | AC1 | "
             "AC2 (ordinal) | S | S (ordinal) | A01 marginal 0/1/2 | A02 marginal 0/1/2 |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for name, d in data.items():
        for scoring, src in [("capped", d["scores"]), ("uncapped", d["unc_scores"])]:
            for mode in CELLS:
                g = agreement(src[mode])
                ma = "/".join(str(g["marg_a"].get(c, 0)) for c in (0, 1, 2))
                mb = "/".join(str(g["marg_b"].get(c, 0)) for c in (0, 1, 2))
                L.append(f"| {name} | {scoring} | `{mode}` | {g['alpha']:.4f} | "
                         f"{g['kappa']:.3f} | {g['exact']:.4f} | {g['within1']:.4f} | "
                         f"{g['ac1']:.4f} | {g['ac2']:.4f} | {g['s']:.3f} | {g['s_lin']:.3f} | "
                         f"{ma} | {mb} |\n")

    # --- 4. Tier-2 ------------------------------------------------------------------------
    L.append("\n## 4. Tier-2 (RQ3) on Tier-1-gated rows, reference = events_checked.csv\n\n")
    L.append("`art.` = leave-one-out articulator-mean predictor MAE; `det.` = share of the "
             "model's gated outputs equal to the modal value for their target articulator; "
             "`+-.5` = model within 0.5 of the reference mean; `art. rho` = Spearman of the `art.` "
             "predictor with the reference (the rank-order side of the paper's `art.` comparison). "
             "Span-match gate (10b/08b): not applied here, matching the paper's Tier-2 definition "
             "(rater gate only). Its effect is reported in the span-gate line below.\n")
    tier2 = {}
    for name, d in data.items():
        modes = ALL_MODES if name == "Qwen" else CELLS
        for mode in modes:
            rows = tier2_rows(d["adeq"][mode], d["outputs"][mode], ref, events)
            tier2[(name, mode)] = rows
    L.append("\n### Gate composition by target articulator\n\n")
    L.append("| model | cell | manual | face | non_dominant_hand | head_torso | total |\n"
             "|---|---|---|---|---|---|---|\n")
    for (name, mode), rows in tier2.items():
        if mode not in CELLS:
            continue
        c = Counter(ch for _, ch, _, _ in rows)
        L.append(f"| {name} | `{mode}` | " + " | ".join(str(c.get(ch, 0)) for ch in CHANNELS)
                 + f" | {len(rows)} |\n")

    L.append("\nSpan-match gate (08b `localization_matches_events_<cell>.csv`, combined): ")
    for name, cfg in MODELS.items():
        for mode in CELLS:
            with open(cfg["results"] / f"localization_matches_events_{mode}.csv",
                      encoding="utf-8") as f:
                lm = {(r["clip_id"], r["event_id"]): r["matched"] == "True"
                      for r in csv.DictReader(f) if r["clip_kind"] == "combined"}
            miss = sorted(k for k, v in lm.items() if not v)
            gated_miss = [k for k in miss if data[name]["adeq"][mode].get(k)]
            L.append(f"{name} `{mode}` {sum(lm.values())}/{len(lm)} matched, unmatched "
                     f"{', '.join(c + '/' + e for c, e in miss) or 'none'}, of which Tier-1-gated "
                     f"{len(gated_miss)}; ")
    L.append("\n")

    stats = {}
    for (name, mode), rows in tier2.items():
        for label, col in SCORES:
            stats[(name, mode, label)] = score_stats(rows, col)

    def bh_family(keys):
        live = [k for k in keys if stats[k] and stats[k]["testable"]]
        adj = S18.benjamini_hochberg([stats[k]["p"] for k in live])
        return dict(zip(live, adj)), len(live), len(keys)

    fam_glm, n_glm, all_glm = bh_family([("GLM", m, s) for m in CELLS for s, _ in SCORES])
    fam_q4, n_q4, _ = bh_family([("Qwen", m, s) for m in ALL_MODES for s, _ in SCORES])
    fam_q2, n_q2, _ = bh_family([("Qwen", m, s) for m in CELLS for s, _ in SCORES])

    L.append("\n### Per-score results\n\n")
    L.append(f"BH families: GLM = its {n_glm} testable correlations over its 2 rated cells "
             f"(of {all_glm}); Qwen (paper) = {n_q4} testable over 4 cells; Qwen-2cell = "
             f"{n_q2} testable over the same 2 cells as GLM (post-hoc, like-for-like).\n\n")
    L.append("| score | cell | model | n | MAE | art. | rho (raw p) | art. rho | tau_b | BH p "
             "(own family) | BH p Qwen-2cell | +-.5 | det. | bias | model dist. | reference dist. |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for label, _ in SCORES:
        for mode in CELLS:
            for name in MODELS:
                s = stats[(name, mode, label)]
                if s is None:
                    L.append(f"| {label} | `{mode}` | {name} | 0 | -- |" + " -- |" * 11 + "\n")
                    continue
                key = (name, mode, label)
                if s["testable"]:
                    bh = fam_glm.get(key) if name == "GLM" else fam_q4.get(key)
                    bh2 = fam_q2.get(key) if name == "Qwen" else None
                    rho_txt = f"{s['rho']:+.2f} ({s['p']:.4f})"
                    tau_txt = f"{s['tau']:+.2f}"
                else:
                    bh = bh2 = None
                    rho_txt, tau_txt = "not testable", "--"
                L.append(f"| {label} | `{mode}` | {name} | {s['n']} | {s['mae']:.2f} | "
                         f"{s['art']:.2f} | {rho_txt} | {fmt(s['art_rho'], '{:+.2f}')} | "
                         f"{tau_txt} | {fmt(bh, '{:.4f}')} | "
                         f"{fmt(bh2, '{:.4f}')} | {s['half']}/{s['n']} | {s['det']:.0%} | "
                         f"{s['bias']:+.2f} | {dist(s['mdist'])} | {dist(s['hdist'])} |\n")
    surv = sorted(k for k, v in fam_glm.items() if v < 0.05)
    L.append(f"\nGLM correlations surviving BH 5%: "
             f"{', '.join(f'{m}/{s}' for _, m, s in surv) if surv else 'none'} "
             f"(raw p < .05: {sum(1 for k in fam_glm if stats[k]['p'] < 0.05)}).\n")
    surv2 = sorted(k for k, v in fam_q2.items() if v < 0.05)
    L.append(f"Qwen-2cell correlations surviving BH 5%: "
             f"{', '.join(f'{m}/{s}' for _, m, s in surv2) if surv2 else 'none'}.\n")

    L.append("\n### Common subset: items gated in both cells\n\n")
    for name in MODELS:
        ka = {r[0] for r in tier2[(name, 'grounded')]}
        kb = {r[0] for r in tier2[(name, 'ablate_channel')]}
        common = sorted(ka & kb)
        comp = Counter(channel_of[k] for k in common)
        cells = []
        for mode in CELLS:
            rows = [r for r in tier2[(name, mode)] if r[0] in set(common)]
            cells.append(f"`{mode}` " + ", ".join(
                f"{lab} {fmt((score_stats(rows, col) or {}).get('mae'))}" for lab, col in SCORES))
        L.append(f"- {name}: n = {len(common)} {dict(comp)}; MAE " + "; ".join(cells) + "\n")

    # --- 5. Fixed denominator -------------------------------------------------------------
    L.append("\n## 5. Fixed-denominator check (all 55 events) and joint success\n\n")
    L.append("| model | cell | score | ungated n | ungated MAE | gated n | gated MAE | gated +-.5 | "
             "joint success /55 |\n|---|---|---|---|---|---|---|---|---|\n")
    for name, d in data.items():
        for mode in CELLS:
            for label, col in SCORES:
                allp, gp = [], []
                for k in sorted(events):
                    o = d["outputs"][mode].get(k)
                    if o is None or k not in ref or col not in ref[k]:
                        continue
                    mv = o.get(col)
                    if not isinstance(mv, (int, float)):
                        continue
                    allp.append((ref[k][col], float(mv)))
                    if d["adeq"][mode].get(k):
                        gp.append((ref[k][col], float(mv)))
                within = sum(1 for r, m in gp if abs(r - m) <= 0.5)
                L.append(f"| {name} | `{mode}` | {label} | {len(allp)} | "
                         f"{statistics.mean(abs(r - m) for r, m in allp):.2f} | {len(gp)} | "
                         f"{fmt(statistics.mean(abs(r - m) for r, m in gp) if gp else None)} | "
                         f"{within}/{len(gp)} | {within}/{len(allp)} |\n")

    # --- 6. Ungated D1 by articulator ----------------------------------------------------
    L.append("\n## 6. Ungated model D1 by target articulator (no reference, no gate)\n\n")
    L.append("| model | target | n | reference D1 mean | D1 grounded | D1 -artic. | "
             "channel_judged -artic. |\n|---|---|---|---|---|---|---|\n")
    for name, d in data.items():
        for ch in CHANNELS:
            keys = [k for k in sorted(events) if channel_of[k] == ch]
            refm = statistics.mean(ref[k]["semantic_syntactic_load"] for k in keys)
            dg = Counter(d["outputs"]["grounded"][k]["semantic_syntactic_load"] for k in keys)
            dc = Counter(d["outputs"]["ablate_channel"][k]["semantic_syntactic_load"]
                         for k in keys)
            cj = Counter(d["outputs"]["ablate_channel"][k]["channel_judged"] for k in keys)
            L.append(f"| {name} | `{ch}` | {len(keys)} | {refm:.2f} | {dist(dg)} | {dist(dc)} | "
                     f"{dict(cj)} |\n")
    L.append("\nFace targets, paired per event (grounded -> -articulator D1): ")
    for name, d in data.items():
        keys = [k for k in sorted(events) if channel_of[k] == "face"]
        up = sum(1 for k in keys if d["outputs"]["ablate_channel"][k]["semantic_syntactic_load"]
                 > d["outputs"]["grounded"][k]["semantic_syntactic_load"])
        dn = sum(1 for k in keys if d["outputs"]["ablate_channel"][k]["semantic_syntactic_load"]
                 < d["outputs"]["grounded"][k]["semantic_syntactic_load"])
        g1 = sum(1 for k in keys if d["outputs"]["grounded"][k]["semantic_syntactic_load"] == 1)
        c2 = sum(1 for k in keys
                 if d["outputs"]["ablate_channel"][k]["semantic_syntactic_load"] == 2)
        man = sum(1 for k in keys if d["outputs"]["ablate_channel"][k]["channel_judged"]
                  == "manual")
        gj = Counter(d["outputs"]["grounded"][k]["channel_judged"] for k in keys)
        L.append(f"{name}: D1=1 named {g1}/15, D1=2 withheld {c2}/15, up {up} / down {dn} / "
                 f"same {15 - up - dn}, withheld self-selects manual {man}/15, grounded "
                 f"channel_judged {dict(gj)}; ")
    L.append("\n")

    L.append("\n## 7. Ungated `grounded`: reference mean / modal model value by target "
             "articulator (paper Table 6 analogue)\n\n")
    L.append("| model | score | manual (32) | face (15) | non-dom. (7) | outputs at their "
             "articulator's mode |\n|---|---|---|---|---|---|\n")
    for name, d in data.items():
        out = d["outputs"]["grounded"]
        for label, col in SCORES:
            by = defaultdict(list)
            for k in sorted(events):
                by[channel_of[k]].append((ref[k][col], out[k][col]))
            hits = sum(Counter(y for _, y in v).most_common(1)[0][1] for v in by.values())
            cells = [f"{statistics.mean(x for x, _ in by[ch]):.2f}/"
                     f"{Counter(y for _, y in by[ch]).most_common(1)[0][0]}"
                     for ch in ("manual", "face", "non_dominant_hand")]
            L.append(f"| {name} | {label} | " + " | ".join(cells) + f" | {hits}/55 |\n")

    OUT.write_text("".join(L), encoding="utf-8")
    log.info("Wrote %s", OUT)
    print("".join(L))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "split":
        split()
    else:
        report()


if __name__ == "__main__":
    main()
