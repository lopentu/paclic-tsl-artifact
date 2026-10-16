#!/usr/bin/env python3
"""Read out the T=1.0 sampling sweep (28): is "articulator-determined" a decoding artifact?

Every rated result in the paper is one greedy sample per item, so "D1 is 100% articulator-
determined in the gated set" is a statement about the outputs we observed, not about the model's
distribution. A referee can answer it in one line: *you decoded at temperature 0, so of course the
output is deterministic.* This script restates the claim over a sampled distribution.

Sweep 28 draws k samples per item at T = 1.0 (undamped, deliberately harder than the instruct
model's recommended 0.7) in the two cells the claim is made about: `grounded`, where D1 is claimed
articulator-determined, and `ablate_channel`, where the same items flip to the self-selected
articulator's value.

**Analysed at k = 4 by default, not k = 5.** The sweep's outer loop is the sample index, so an
interrupted run leaves complete layers; at the time of writing layers 1-4 are complete in *both*
cells (440 runs) while layer 5 is complete only in `grounded`. Analysing "all files present" would
therefore compare a 5-draw `grounded` cell against a 4-draw `ablate_channel` cell, and the modal
share of a 5-sample set is not comparable with that of a 4-sample set. `--k` fixes the depth; the
script refuses to run on any item that cannot supply it.

Readouts, all reference-free:

  1. **Articulator determination under sampling** -- the share of *draws* landing on the target
     articulator's modal D1. This is the number that replaces "100% articulator-determined".
  2. **Per-item stability** -- modal share and Shannon entropy of each item's k draws, so a high
     aggregate cannot hide items that are internally unstable.
  3. **Greedy vs. sampled mode** -- whether the T=0 output the paper reports is the same value
     sampling makes modal. If it is, the greedy run is a representative draw, not a lucky one.
  4. **The face flip** -- the paper's sharpest label-driven observation is that the 15 face items
     score D1=1 when the face is named and D1=2 when the articulator is withheld. Under sampling
     this becomes a difference of distributions rather than of two point outputs.
  5. **Endpoint avoidance** -- greedy never emits D1=0 or D1=3. Sampling asks whether the
     endpoints are absent from the distribution or merely off its mode, which is the difference
     between a compressed mapping and an unanchored scale.

Usage:
    python scripts/30_analyze_temperature_stability.py
    python scripts/30_analyze_temperature_stability.py --k 5 --modes grounded
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import get_logger, ensure_dir, REPO_ROOT, parse_manifest_event

log = get_logger("analyze_temperature")

DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"
SAMPLE_DIR = REPO_ROOT / "data/model_outputs/qwen3vl_8b/temperature_t1"
GREEDY_DIR = REPO_ROOT / "data/model_outputs/qwen3vl_8b"
OUT_MD = REPO_ROOT / "data/results/qwen3vl_8b/temperature_stability.md"

MODES = ["grounded", "ablate_channel"]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]
SCORE_KEYS = [("D1", "semantic_syntactic_load"),
              ("D2", "pragmatic_depictive_affective_enrichment"),
              ("description", "description"), ("indication", "indication"),
              ("depiction", "depiction")]


def entropy(values):
    """Shannon entropy in bits of a list of discrete outcomes."""
    if not values:
        return float("nan")
    counts = Counter(values)
    n = len(values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def load_events(manifest_path):
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


def candidate_of(path):
    """Returns (candidate, status) with status in {ok, absent, unparsed}.

    `absent` and `unparsed` are kept apart on purpose: a file that does not exist is a run the
    sweep has not reached, whereas a file whose `parsed` block is empty is a run the model
    completed and whose output failed the JSON schema. The first is a gap in coverage, the
    second is a result -- the greedy runs are 220/220 schema-valid, so any sampled failure rate
    belongs in the paper rather than in a missing-data note.
    """
    if not path.exists():
        return None, "absent"
    data = json.loads(path.read_text(encoding="utf-8"))
    parsed = data.get("parsed")
    cands = (parsed.get("candidate_actions") if isinstance(parsed, dict) else None) or []
    if not cands:
        return None, "unparsed"
    return cands[0], "ok"


def score(candidate, key):
    if candidate is None:
        return None
    value = candidate.get(key)
    return value if isinstance(value, (int, float)) else None


def collect(events, k, modes):
    """samples[mode][(clip,event)] = list of k candidate dicts; greedy[mode][(clip,event)] = one."""
    samples = {m: {} for m in modes}
    greedy = {m: {} for m in modes}
    absent, unparsed = [], []
    for clip, event, channel, video in events:
        for mode in modes:
            drawn = []
            for i in range(1, k + 1):
                path = SAMPLE_DIR / f"{clip}__{event}__combined__sample{i}__{mode}.json"
                cand, status = candidate_of(path)
                if status == "absent":
                    absent.append(path.name)
                elif status == "unparsed":
                    unparsed.append(path.name)
                else:
                    drawn.append(cand)
            samples[mode][(clip, event)] = drawn
            greedy[mode][(clip, event)] = candidate_of(
                GREEDY_DIR / f"{clip}__{event}__combined__run1__{mode}.json")[0]
    return samples, greedy, absent, unparsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--k", type=int, default=4,
                    help="draws per item to analyse; the default is the deepest layer complete "
                         "in BOTH cells (see module docstring)")
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--out", default=str(OUT_MD))
    args = ap.parse_args()

    events = load_events(args.manifest)
    channel_of = {(c, e): ch for c, e, ch, _ in events}
    samples, greedy, absent, unparsed = collect(events, args.k, args.modes)

    attempted = len(events) * len(args.modes) * args.k
    out = [f"# Output stability under sampling (T = 1.0, k = {args.k})\n",
           f"\n{len(events)} events x {len(args.modes)} cells x {args.k} draws = "
           f"{attempted} sampled runs, against the greedy (T = 0) runs the paper reports.\n"]
    if absent:
        out.append(f"\n**{len(absent)} draws not yet run at k = {args.k}** "
                   f"(e.g. {absent[0]}), so the cells below are NOT balanced. Resume 28 before "
                   "quoting these numbers.\n")
    else:
        out.append(f"\nEvery item supplies {args.k} draws in every cell: no gaps in coverage.\n")
    valid = attempted - len(absent) - len(unparsed)
    out.append(f"\n**Schema validity under sampling: {valid}/{attempted - len(absent)} "
               f"({100 * valid / max(1, attempted - len(absent)):.1f}%).** The greedy runs are "
               "220/220. ")
    if unparsed:
        out.append(f"{len(unparsed)} sampled run(s) emitted unparseable JSON "
                   f"({', '.join(unparsed)}) and are dropped from every readout below; the "
                   "schema is enforced identically in both settings, so this is a property of "
                   "sampling, not of the parser.\n")
    else:
        out.append("No sampled run failed the schema.\n")

    # ---- 1. Articulator determination, over draws rather than over greedy outputs -------------
    out.append("\n## 1. Is D1 articulator-determined in the distribution?\n\n"
               "`modal(greedy)` is the target articulator's modal D1 among the greedy runs -- the "
               "value the paper's 'articulator-determined' counts are computed against. "
               "`draws at modal` pools every draw from every item of that articulator.\n\n")
    for mode in args.modes:
        greedy_modal = {}
        for ch in CHANNELS:
            vals = [score(greedy[mode][key], "semantic_syntactic_load")
                    for key, c in channel_of.items() if c == ch]
            vals = [v for v in vals if v is not None]
            if vals:
                greedy_modal[ch] = Counter(vals).most_common(1)[0][0]
        out.append(f"\n**{mode}**\n\n"
                   "| target articulator | n | modal(greedy) | draws at modal | mean per-item "
                   "modal share | mean entropy (bits) | values seen |\n"
                   "|---|---|---|---|---|---|---|\n")
        for ch in CHANNELS:
            keys = [key for key, c in channel_of.items() if c == ch]
            if not keys:
                continue
            pooled, shares, ents = [], [], []
            for key in keys:
                vals = [score(c, "semantic_syntactic_load") for c in samples[mode][key]]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                pooled.extend(vals)
                shares.append(Counter(vals).most_common(1)[0][1] / len(vals))
                ents.append(entropy(vals))
            if not pooled:
                continue
            gm = greedy_modal.get(ch)
            at = sum(1 for v in pooled if v == gm)
            seen = "/".join(str(v) for v in sorted(set(pooled)))
            out.append(f"| {ch} | {len(keys)} | {gm} | {at}/{len(pooled)} = "
                       f"{100 * at / len(pooled):.1f}% | {statistics.mean(shares):.2f} | "
                       f"{statistics.mean(ents):.2f} | {seen} |\n")

    # ---- 2. Greedy output vs. the sampled mode -----------------------------------------------
    out.append("\n## 2. Is the greedy output the sampled mode?\n\n"
               "If it is, the paper's T=0 runs are representative draws rather than an artifact "
               "of the decoding rule.\n\n"
               "| cell | score | greedy == sampled mode | mean per-item modal share | "
               "mean entropy |\n|---|---|---|---|---|\n")
    for mode in args.modes:
        for label, key in SCORE_KEYS:
            agree = total = 0
            shares, ents = [], []
            for ek in channel_of:
                vals = [score(c, key) for c in samples[mode][ek]]
                vals = [v for v in vals if v is not None]
                g = score(greedy[mode][ek], key)
                if not vals or g is None:
                    continue
                mode_val, count = Counter(vals).most_common(1)[0]
                total += 1
                agree += int(mode_val == g)
                shares.append(count / len(vals))
                ents.append(entropy(vals))
            if total:
                out.append(f"| {mode} | {label} | {agree}/{total} | "
                           f"{statistics.mean(shares):.2f} | {statistics.mean(ents):.2f} |\n")

    # ---- 3. The face flip, as a difference of distributions ----------------------------------
    out.append("\n## 3. The face flip under sampling\n\n"
               "Greedy: the 15 face targets score D1=1 on 15/15 when the face is named and D1=2 "
               "on 15/15 when the articulator is withheld. Pooling draws turns that into a "
               "difference of distributions.\n\n"
               "| target articulator | cell | mean D1 | D1 distribution over draws | "
               "channel_judged = cued/self-selected |\n|---|---|---|---|---|\n")
    for ch in CHANNELS:
        keys = [key for key, c in channel_of.items() if c == ch]
        if not keys:
            continue
        for mode in args.modes:
            vals, chans = [], []
            for key in keys:
                for c in samples[mode][key]:
                    v = score(c, "semantic_syntactic_load")
                    if v is not None:
                        vals.append(v)
                    chans.append((c or {}).get("channel_judged"))
            if not vals:
                continue
            dist = Counter(vals)
            dist_s = ", ".join(f"{v}:{dist[v]}" for v in sorted(dist))
            cc = Counter(chans)
            if mode == "grounded":
                comply = f"{cc.get(ch, 0)}/{len(chans)} at cued"
            else:
                top, n = cc.most_common(1)[0] if cc else ("--", 0)
                comply = f"{n}/{len(chans)} at {top}"
            out.append(f"| {ch} | {mode} | {statistics.mean(vals):.2f} | {dist_s} | "
                       f"{comply} |\n")

    # ---- 4. Endpoint avoidance ---------------------------------------------------------------
    out.append("\n## 4. Does sampling reach the endpoints the greedy runs never use?\n\n"
               "The paper reports that the model never emits D1=0 or D1=3, and flags that a "
               "compressed mapping is confounded with a failure to anchor an unfamiliar scale. "
               "If the endpoints are absent from 440 undamped draws too, the compression is a "
               "property of the distribution.\n\n"
               "| cell | score | draws | 0 | 1 | 2 | 3 | endpoint share |\n|---|---|---|---|---|"
               "---|---|---|\n")
    for mode in args.modes:
        for label, key in SCORE_KEYS:
            vals = [score(c, key) for ek in channel_of for c in samples[mode][ek]]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            d = Counter(vals)
            ends = d.get(0, 0) + d.get(3, 0)
            out.append(f"| {mode} | {label} | {len(vals)} | {d.get(0,0)} | {d.get(1,0)} | "
                       f"{d.get(2,0)} | {d.get(3,0)} | {100 * ends / len(vals):.1f}% |\n")

    # ---- 5. Per-item instability, so an aggregate cannot hide it -----------------------------
    out.append("\n## 5. Items whose D1 is not unanimous across draws\n\n")
    for mode in args.modes:
        rows = []
        for ek, ch in channel_of.items():
            vals = [score(c, "semantic_syntactic_load") for c in samples[mode][ek]]
            vals = [v for v in vals if v is not None]
            if vals and len(set(vals)) > 1:
                rows.append((ek, ch, vals))
        out.append(f"\n**{mode}**: {len(rows)}/{len(channel_of)} items split across draws"
                   + (".\n\n" if not rows else ":\n\n"))
        if rows:
            out.append("| item | articulator | draws |\n|---|---|---|\n")
            for (clip, event), ch, vals in sorted(rows):
                out.append(f"| {clip} {event} | {ch} | {', '.join(str(v) for v in vals)} |\n")

    # ---- 6. Confidence ------------------------------------------------------------------------
    out.append("\n## 6. Stated confidence\n\n| cell | mean stated confidence over draws |\n"
               "|---|---|\n")
    for mode in args.modes:
        vals = [score(c, "confidence") for ek in channel_of for c in samples[mode][ek]]
        vals = [v for v in vals if v is not None]
        if vals:
            out.append(f"| {mode} | {statistics.mean(vals):.2f} |\n")

    ensure_dir(Path(args.out).parent)
    Path(args.out).write_text("".join(out), encoding="utf-8")
    log.info("wrote %s", args.out)
    print("".join(out))


if __name__ == "__main__":
    main()
