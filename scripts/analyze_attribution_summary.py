#!/usr/bin/env python3
"""Roll 10b's four per-mode gated attribution CSVs into the tables the write-up needs, and add the
three things a per-mode CSV cannot show on its own.

10b_score_attribution_events.py reports each cue mode separately, gated on Tier 1 (span matched)
and Tier 2 (mean perception_score >= threshold). Those four numbers are the primary result, but
comparing them ACROSS modes is not valid as-is, because the Tier-2 gate admits a different -- and
differently sized -- subset of items in each mode (that is itself a finding, see section 1). So:

1. **Gate composition.** How many items survive each mode's gate, broken down by the manifest's
   `target_channel`. This is what makes the cross-mode comparison interpretable rather than
   misleading: if withholding the channel cue removes most `face` items from the gated set, then a
   change in that mode's D1 MAE is partly a change of sample, not of behaviour.

2. **Common subset.** The items that pass the gate in ALL FOUR modes, compared within-item across
   modes. Small by construction, but it is the only cue-ablation comparison here that holds the
   item set fixed.

3. **Trivial baselines.** MAE against the best constant guess (the reference mean, and the
   reference median) plus the number of distinct values the model actually used. Without these an
   MAE around 1 on a 0-3 scale looks like a moderate result when it may be no better than
   answering the same number every time.

Reads 10b's gated `attribution_scores_events_{mode}.csv` (NOT the `_ungated` files) plus
clip_manifest.csv for target_channel. Writes one markdown report.

Usage:
    python scripts/analyze_attribution_summary.py
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir, REPO_ROOT, parse_manifest_event
from lib.stats_utils import mean_absolute_error, spearman_correlation

log = get_logger("analyze_attribution_summary")

MODES = ["grounded", "ablate_translation", "ablate_channel", "ablate_both"]
CUE_LABEL = {"grounded": "translation + channel", "ablate_translation": "channel only",
             "ablate_channel": "translation only", "ablate_both": "neither (span only)"}
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]
SEMIOTIC = ["description", "indication", "depiction"]
DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"


def load_target_channels(manifest_path):
    out = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for index, event_id in ((1, "event1"), (2, "event2")):
                parsed = parse_manifest_event(row, index)
                if parsed is not None:
                    out[(row["clip_id"], event_id)] = parsed[2]
    return out


def load_scores(results_dir, mode, clip_kind="combined", suffix=""):
    path = results_dir / f"attribution_scores_events_{mode}{suffix}.csv"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["clip_kind"] == clip_kind]


def pair_stats(rows, human_col, model_col):
    pairs = [(float(r[human_col]), float(r[model_col])) for r in rows
             if r.get(human_col, "") != "" and r.get(model_col, "") != ""]
    if len(pairs) < 2:
        return None
    h, m = [x for x, _ in pairs], [y for _, y in pairs]
    rho, p = spearman_correlation(h, m)
    const_mean = mean_absolute_error(h, [statistics.mean(h)] * len(h))
    const_med = mean_absolute_error(h, [statistics.median(h)] * len(h))
    return {"n": len(pairs), "mae": mean_absolute_error(h, m), "rho": rho, "p": p,
            "best_const_mae": min(const_mean, const_med),
            # Mean signed error separates a constant offset (model systematically low/high but
            # tracking rank order) from unsystematic error. |bias| close to MAE means almost all
            # the error is one directional shift, which MAE alone hides.
            "bias": statistics.mean(y - x for x, y in pairs),
            # With two annotators on an integer 0-3 scale the reference mean is either an integer
            # (they agreed) or a .5 (they differed by one). |model - mean| <= 0.5 therefore means
            # "the model's score falls inside the interval the two experts bracketed" -- an exact
            # match when they agreed, and either of their two values when they did not. It is the
            # hit rate MAE cannot express, and it is scale-appropriate in a way MAE is not.
            "within_half": sum(1 for x, y in pairs if abs(y - x) <= 0.5),
            "model_distinct": len(set(m)), "model_mean": statistics.mean(m),
            "human_mean": statistics.mean(h)}


def fmt(stats, key, spec="{:.2f}"):
    return spec.format(stats[key]) if stats else "--"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--clip-kind", default="combined")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--suffix", default="",
                        help="suffix of the 10b outputs to read, e.g. `_checked` for the run "
                             "scored against the post-discussion reference")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    channels = load_target_channels(Path(args.manifest) if args.manifest else DEFAULT_MANIFEST)

    per_mode = {}
    for mode in MODES:
        rows = load_scores(results_dir, mode, args.clip_kind, args.suffix)
        if rows is None:
            log.warning("attribution_scores_events_%s.csv missing -- run 10b (gated) first.", mode)
            continue
        per_mode[mode] = rows
    if not per_mode:
        raise SystemExit("No gated attribution CSVs found. Run 10b_score_attribution_events.py "
                         "without --ignore-perception-gate first.")

    lines = ["# Tier-3 attribution: gated results, gate composition, and baselines\n\n",
             f"Clip kind: `{args.clip_kind}`. All rows are gated on Tier 1 (span matched) and "
             "Tier 2 (mean perception_score >= "
             f"{cfg['attribution_scoring']['perception_adequate_min_score']}).\n"]

    # --- 1. Gate composition ---------------------------------------------------------------
    lines.append("\n## 1. Gate composition: which items reach Tier 3\n")
    lines.append("\nThe gated n differs per mode, so the cross-mode numbers in section 2 are "
                 "partly a change of sample. This table is what makes that visible.\n\n")
    lines.append("| target_channel | items rated | " + " | ".join(f"`{m}`" for m in per_mode)
                 + " |\n|---|---|" + "---|" * len(per_mode) + "\n")
    total_by_channel = defaultdict(int)
    for key, ch in channels.items():
        total_by_channel[ch] += 1
    for ch in CHANNELS:
        if not total_by_channel[ch]:
            continue
        cells = []
        for mode, rows in per_mode.items():
            n = sum(1 for r in rows if channels.get((r["clip_id"], r["event_id"])) == ch)
            cells.append(str(n))
        lines.append(f"| `{ch}` | {total_by_channel[ch]} | " + " | ".join(cells) + " |\n")
    lines.append("| **total** | **" + str(sum(total_by_channel.values())) + "** | "
                 + " | ".join(f"**{len(r)}**" for r in per_mode.values()) + " |\n")

    # --- 2. Per-mode gated results ---------------------------------------------------------
    lines.append("\n## 2. Per-mode gated results\n")
    lines.append("\n`best const.` is the MAE of always answering the reference set's own mean or "
                 "median (whichever is better) -- an MAE at or above this is no better than a "
                 "constant guess. `distinct` is how many different values the model actually "
                 "used.\n\n")
    lines.append("| mode | cues | n | D1 MAE | D1 best const. | D1 bias | D1 rho (p) | "
                 "D1 within .5 | D1 distinct | D2 MAE | D2 best const. | D2 bias | D2 rho (p) | "
                 "D2 within .5 | D2 distinct |\n"
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for mode, rows in per_mode.items():
        d1 = pair_stats(rows, "human_d1", "model_d1")
        d2 = pair_stats(rows, "human_d2", "model_d2")
        lines.append(
            f"| `{mode}` | {CUE_LABEL[mode]} | {len(rows)} | {fmt(d1,'mae')} | "
            f"{fmt(d1,'best_const_mae')} | {fmt(d1,'bias','{:+.2f}')} | "
            f"{fmt(d1,'rho')} ({fmt(d1,'p','{:.3f}')}) | "
            f"{d1['within_half']}/{d1['n']} ({d1['within_half']/d1['n']:.0%}) | "
            f"{fmt(d1,'model_distinct','{:d}')} | {fmt(d2,'mae')} | {fmt(d2,'best_const_mae')} | "
            f"{fmt(d2,'bias','{:+.2f}')} | "
            f"{fmt(d2,'rho')} ({fmt(d2,'p','{:.3f}')}) | "
            f"{d2['within_half']}/{d2['n']} ({d2['within_half']/d2['n']:.0%}) | "
            f"{fmt(d2,'model_distinct','{:d}')} |\n")

    lines.append("\n### Directional error rates and semiotic modes\n\n")
    lines.append("| mode | n | D1 under-attrib. | D2 over-attrib. | "
                 + " | ".join(f"{m} MAE / within .5" for m in SEMIOTIC) + " |\n|---|---|---|---|"
                 + "---|" * len(SEMIOTIC) + "\n")
    for mode, rows in per_mode.items():
        under = sum(1 for r in rows if r["under_attribution_d1"] == "True") / len(rows)
        over = sum(1 for r in rows if r["over_attribution_d2"] == "True") / len(rows)
        cells = []
        for sm in SEMIOTIC:
            s = pair_stats(rows, f"human_{sm}", f"model_{sm}")
            cells.append(f"{fmt(s,'mae')} / {s['within_half']}/{s['n']}" if s else "--")
        lines.append(f"| `{mode}` | {len(rows)} | {under:.1%} | {over:.1%} | "
                     + " | ".join(cells) + " |\n")

    # --- 3. Common subset ------------------------------------------------------------------
    keysets = [{(r["clip_id"], r["event_id"]) for r in rows} for rows in per_mode.values()]
    common = set.intersection(*keysets) if keysets else set()
    lines.append(f"\n## 3. Common subset: items passing the gate in all {len(per_mode)} modes "
                 f"(n={len(common)})\n")
    if len(common) < 3:
        lines.append("\nToo few items pass every mode's gate for a within-item cross-mode "
                     "comparison; report the per-mode numbers with the composition caveat "
                     "instead.\n")
    else:
        ch_counts = defaultdict(int)
        for key in common:
            ch_counts[channels.get(key, "?")] += 1
        lines.append("\nChannel composition: "
                     + ", ".join(f"`{k}` {v}" for k, v in sorted(ch_counts.items())) + ".\n")
        lines.append("\nThe only cue comparison here that holds the item set fixed. Small n; read "
                     "as a consistency check on section 2, not as an independent result.\n\n")
        lines.append("| mode | D1 MAE | D1 rho | D2 MAE | D2 rho | "
                     + " | ".join(f"{m} MAE" for m in SEMIOTIC) + " |\n|---|---|---|---|---|"
                     + "---|" * len(SEMIOTIC) + "\n")
        for mode, rows in per_mode.items():
            sub = [r for r in rows if (r["clip_id"], r["event_id"]) in common]
            d1, d2 = pair_stats(sub, "human_d1", "model_d1"), pair_stats(sub, "human_d2", "model_d2")
            cells = [fmt(pair_stats(sub, f"human_{sm}", f"model_{sm}"), "mae") for sm in SEMIOTIC]
            lines.append(f"| `{mode}` | {fmt(d1,'mae')} | {fmt(d1,'rho')} | {fmt(d2,'mae')} | "
                         f"{fmt(d2,'rho')} | " + " | ".join(cells) + " |\n")

    # --- 4. Score distributions ------------------------------------------------------------
    lines.append("\n## 4. Score distributions (model vs reference), gated rows\n\n")
    for mode, rows in per_mode.items():
        lines.append(f"- `{mode}` (n={len(rows)}): ")
        parts = []
        for label, hcol, mcol in [("D1", "human_d1", "model_d1"), ("D2", "human_d2", "model_d2")]:
            mvals = sorted(float(r[mcol]) for r in rows)
            hvals = sorted(float(r[hcol]) for r in rows)
            mdist = {v: mvals.count(v) for v in sorted(set(mvals))}
            parts.append(f"{label} model mean {statistics.mean(mvals):.2f} "
                         f"{{{', '.join(f'{k:g}: {v}' for k, v in mdist.items())}}}, "
                         f"reference mean {statistics.mean(hvals):.2f}")
        lines.append("; ".join(parts) + "\n")

    # --- 5. Semiotic-mode distributions ----------------------------------------------------
    # A low MAE on a 0-3 scale can come from signal or from both sides sitting on the same floor.
    # depiction in particular is near-0 in the reference for most items, so its MAE must be read
    # against the distributions, not on its own. Also note the reference's OWN inter-annotator
    # agreement differs sharply across the three modes (see 12_compute_annotator_agreement.py):
    # a mode the annotators do not agree on cannot support a model-vs-reference claim either way.
    lines.append("\n## 5. Semiotic-mode distributions (gated rows)\n")
    lines.append("\nRead the MAEs in section 2 against these: when reference and model both sit "
                 "at the same end of the scale, a low MAE reflects a shared floor rather than "
                 "recovered signal.\n\n")
    lines.append("| mode | semiotic mode | reference distribution | model distribution | "
                 "rho (p) |\n|---|---|---|---|---|\n")
    for mode, rows in per_mode.items():
        for sm in SEMIOTIC:
            pairs = [(float(r[f"human_{sm}"]), float(r[f"model_{sm}"])) for r in rows
                     if r.get(f"human_{sm}", "") != "" and r.get(f"model_{sm}", "") != ""]
            if not pairs:
                continue
            h = [x for x, _ in pairs]
            m = [y for _, y in pairs]
            hd = {v: h.count(v) for v in sorted(set(h))}
            md = {v: m.count(v) for v in sorted(set(m))}
            rho, p = spearman_correlation(h, m) if len(set(m)) > 1 and len(set(h)) > 1 \
                else (float("nan"), float("nan"))
            rho_txt = "undefined (no variance)" if rho != rho else f"{rho:.2f} ({p:.3f})"
            lines.append(f"| `{mode}` | {sm} | "
                         f"{{{', '.join(f'{k:g}: {v}' for k, v in hd.items())}}} | "
                         f"{{{', '.join(f'{k:g}: {v}' for k, v in md.items())}}} | {rho_txt} |\n")

    # --- 6. Is the model's score anything more than the target channel restated? -------------
    # Section 2's Spearman rho can be produced entirely by BETWEEN-channel structure: if the
    # reference gives high D1 to manual targets and low D1 to face targets, and the model emits one
    # value per channel, the two rank-correlate without the model having read any individual clip.
    # Three things separate those accounts, none of which requires new annotation:
    #   (a) how much of the MODEL's own output is predictable from target_channel alone (a fact
    #       about the output; no reference involved, so no circularity);
    #   (b) the same for the REFERENCE, which bounds what a channel-only predictor could achieve;
    #   (c) a leave-one-out channel-mean predictor -- predict each item from the reference mean of
    #       the OTHER items sharing its channel -- scored on the same rows as the model. If the
    #       model does not beat this, it has added nothing to knowing which channel was named.
    lines.append("\n## 6. Channel-only baseline: does the model add anything to the channel label?\n")
    lines.append("\n`channel-determined` = fraction of rows whose value equals the modal value for "
                 "their target channel; for the model this is computed with no reference involved. "
                 "`LOO channel-mean` predicts each item from the reference mean of the *other* "
                 "gated items sharing its target channel, scored on the same rows as the model -- "
                 "so it measures what is achievable from the channel label alone.\n\n")
    lines.append("| mode | dim | model MAE | LOO channel-mean MAE | model rho | LOO channel-mean "
                 "rho | model channel-determined | reference channel-determined |\n"
                 "|---|---|---|---|---|---|---|---|\n")
    for mode, rows in per_mode.items():
        for dim, hcol, mcol in [("D1", "human_d1", "model_d1"), ("D2", "human_d2", "model_d2")]:
            items = [(channels.get((r["clip_id"], r["event_id"])), float(r[hcol]), float(r[mcol]))
                     for r in rows]
            by_ch = defaultdict(list)
            for ch, h, m in items:
                by_ch[ch].append((h, m))

            def modal_share(index):
                hits = 0
                for ch, vals in by_ch.items():
                    col = [v[index] for v in vals]
                    hits += col.count(max(set(col), key=col.count))
                return hits / len(items)

            loo = []
            for i, (ch, h, _) in enumerate(items):
                others = [x[1] for j, x in enumerate(items) if j != i and x[0] == ch]
                loo.append(statistics.mean(others) if others else statistics.mean(
                    [x[1] for j, x in enumerate(items) if j != i]))
            h_all = [x[1] for x in items]
            m_all = [x[2] for x in items]
            rho_m = spearman_correlation(h_all, m_all)[0] if len(set(m_all)) > 1 else float("nan")
            rho_l = spearman_correlation(h_all, loo)[0] if len(set(loo)) > 1 else float("nan")
            lines.append(
                f"| `{mode}` | {dim} | {mean_absolute_error(h_all, m_all):.2f} | "
                f"{mean_absolute_error(h_all, loo):.2f} | "
                f"{'--' if rho_m != rho_m else f'{rho_m:.2f}'} | "
                f"{'--' if rho_l != rho_l else f'{rho_l:.2f}'} | "
                f"{modal_share(1):.1%} | {modal_share(0):.1%} |\n")

    lines.append("\n### Model score by target channel, all 55 rated items per mode (ungated)\n")
    lines.append("\nUngated on purpose: this is a description of what the model emitted, not a "
                 "model-vs-reference accuracy claim, so the perception gate does not apply. The "
                 "same 55 videos and spans are used in every mode -- only the cue text differs.\n\n"
                 "Two provenance facts, because this is the one section reading a file that "
                 "`--suffix` does not select. (i) The `reference D1` column comes from whatever "
                 "reference 10b was last run against for the `_ungated` files, not necessarily "
                 "the one behind the gated sections above; both must be the adjudicated "
                 "`events_checked.csv`, and if the manual/face/non-dominant-hand reference D1 "
                 "below is not 3.00 / 1.13 / 2.14 they are stale and 10b needs re-running with "
                 "`--events-csv .../events_checked.csv`. (ii) `_ungated` lifts the perception "
                 "gate only -- 10b always applies the localization gate -- so a cell can hold "
                 "fewer than 55 rows where the model's self-reported span drifted off target. "
                 "Figures quoted over all 55 events regardless of span match come from "
                 "`18_analyze_robustness_checks.py` section 8, not from here.\n\n")
    lines.append("| target channel | items | reference D1 | "
                 + " | ".join(f"model D1 `{m}`" for m in per_mode) + " |\n|---|---|---|"
                 + "---|" * len(per_mode) + "\n")
    ungated = {}
    for mode in per_mode:
        path = results_dir / f"attribution_scores_events_{mode}_ungated.csv"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                ungated[mode] = [r for r in csv.DictReader(f) if r["clip_kind"] == args.clip_kind]
    if ungated:
        # The `items` and `reference D1` columns must come from the manifest and the reference, not
        # from whichever mode happens to be iterated last: Tier-1 recall on `combined` is 55/55 in
        # three modes but 53/55 in ablate_both, so a per-mode count would silently differ by row.
        # Each cell's own n is therefore printed with the distribution.
        ref_by_ch = defaultdict(list)
        for mode, rows in ungated.items():
            for r in rows:
                ref_by_ch[channels.get((r["clip_id"], r["event_id"]))].append(float(r["human_d1"]))
            break  # one mode is enough for the reference; it does not vary by mode
        for ch in CHANNELS:
            if not total_by_channel[ch]:
                continue
            cells = []
            for mode in per_mode:
                rows = [r for r in ungated.get(mode, [])
                        if channels.get((r["clip_id"], r["event_id"])) == ch]
                if not rows:
                    cells.append("--")
                    continue
                vals = [float(r["model_d1"]) for r in rows]
                dist = {v: vals.count(v) for v in sorted(set(vals))}
                suffix = f" (n={len(rows)})" if len(rows) != total_by_channel[ch] else ""
                cells.append("{" + ", ".join(f"{k:g}: {v}" for k, v in dist.items())
                             + "}" + suffix)
            ref_vals = ref_by_ch.get(ch, [])
            ref_txt = f"{statistics.mean(ref_vals):.2f}" if ref_vals else "--"
            lines.append(f"| `{ch}` | {total_by_channel[ch]} | {ref_txt} | "
                         + " | ".join(cells) + " |\n")
    else:
        lines.append("| _ungated CSVs not found -- run 10b with --ignore-perception-gate_ | | | "
                     + "| " * len(per_mode) + "\n")

    lines.append("\n## 7. Model score within $\\pm$0.5 of the expert mean\n")
    lines.append("\nWith two experts scoring integers 0--3, the reference mean is an integer when "
                 "they agreed and a .5 when they differed by one, so this counts the model landing "
                 "inside the interval the two experts bracketed: an exact match where they agreed, "
                 "either of their values where they did not.\n\n")
    lines.append("| mode | n | D1 | D2 | " + " | ".join(SEMIOTIC) + " |\n|---|---|---|---|"
                 + "---|" * len(SEMIOTIC) + "\n")
    for mode, rows in per_mode.items():
        cells = []
        for col in ["d1", "d2"] + SEMIOTIC:
            s = pair_stats(rows, f"human_{col}", f"model_{col}")
            cells.append(f"{s['within_half']}/{s['n']} ({s['within_half']/s['n']:.0%})" if s else "--")
        lines.append(f"| `{mode}` | {len(rows)} | " + " | ".join(cells) + " |\n")

    out_path = results_dir / f"attribution_summary_{args.clip_kind}{args.suffix}.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    log.info("Wrote %s", out_path)
    print("".join(lines))


if __name__ == "__main__":
    main()
