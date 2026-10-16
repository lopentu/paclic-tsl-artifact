#!/usr/bin/env python3
"""Read out the visual-null baselines: which of the model's outputs survive losing the video?

Companion to 19_run_visual_null_baselines.py. Compares each degraded visual condition against the
REAL main-dataset run for the same item, prompt and cue cell, so every row differs from its
baseline in pixel content alone.

What each comparison licenses:

  * If the FUNCTIONAL scores (D1, D2, the semiotic modes) are largely unchanged with no video, they
    were not being read off the video. This is the behavioural version of the channel-only
    baseline in the paper's Tier 3 -- stronger, because the channel-only predictor is a statistical
    artefact of the reference annotation whereas this is the model's own behaviour.
  * If the FORM DESCRIPTIONS collapse while the functional scores do not, that is the dissociation
    the paper claims, demonstrated directly rather than inferred: form description depends on the
    pixels, functional attribution does not.
  * `static` and `shuffled` localise it further. Both keep the signer's appearance; `static`
    removes motion, `shuffled` keeps every frame and destroys only temporal order. A movement
    description that survives shuffling was not built from a trajectory.

Refusals are counted, not discarded. Under `none` the prompt still refers to "this clip" (it is
held byte-identical -- see 19's docstring), so a hedge or a refusal is a legitimate outcome and one
of the more informative ones: a model that declines to score an unseen clip is behaving better
than one that scores it confidently.

Usage:
    python scripts/analyze_visual_null.py --config scripts/config/pipeline_config.qwen3vl_8b.yaml
    python scripts/analyze_visual_null.py --config ... --mode grounded
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir, REPO_ROOT, parse_manifest_event
from lib.stats_utils import mean_absolute_error

log = get_logger("analyze_visual_null")

DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"
VISUAL_CONDITIONS = ["none", "black", "static", "shuffled", "real"]
MODES = ["grounded", "ablate_translation", "ablate_channel", "ablate_both"]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]
SCORES = [("D1", "semantic_syntactic_load"),
          ("D2", "pragmatic_depictive_affective_enrichment"),
          ("indication", "indication")]
FORM_FIELDS = ["hands", "handshape", "movement", "location", "nonmanual", "non_dominant_hand"]

STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "then", "with", "is",
             "are", "as", "it", "its", "s", "signer", "signer's", "hand", "hands"}

# A hedge or refusal under `none` is a result, so it needs detecting rather than dropping.
HEDGE_RE = re.compile(
    r"\bno video\b|\bnot (?:provided|available|shown|attached|visible)\b|\bcannot (?:see|view|"
    r"analyze|evaluate|determine)\b|\bunable to\b|\bI do not have\b|\bwas not (?:given|provided)\b"
    r"|\bplease provide\b|\bmissing video\b", re.IGNORECASE)


def tokens(text):
    return {t for t in re.findall(r"[a-z']+", (text or "").lower()) if t not in STOPWORDS}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_manifest_events(manifest_path):
    out = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("status") or "").strip().lower() == "excluded":
                continue
            for index in (1, 2):
                parsed = parse_manifest_event(row, index)
                if parsed is not None:
                    out.append((row["clip_id"], f"event{index}", parsed[2]))
    return out


def load_json_run(path):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    cands = (data.get("parsed") or {}).get("candidate_actions") or []
    return {"candidate": cands[0] if cands else None,
            "schema_valid": bool(data.get("schema_valid")),
            "raw": data.get("raw_text") or ""}


def load_reference(events_csv):
    ref = defaultdict(dict)
    with open(events_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["clip_id"], row["event_id"])
            ref[key][row["annotator_id"]] = {
                col: float(row[col]) for col in
                ["semantic_syntactic_load", "pragmatic_depictive_affective_enrichment",
                 "indication"] if (row.get(col) or "").strip()
            }
    return ref


def fmt(value, spec="{:.3f}"):
    return spec.format(value) if isinstance(value, (int, float)) else "--"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--subdir", default="visual_null")
    parser.add_argument("--mode", default=None, help="restrict to one cue cell")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--reference-csv", default=None,
                        help="post-discussion reference for the MAE column (default: "
                             "events_checked.csv beside paths.human_annotations_csv)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    main_dir = cfg["paths"]["model_outputs_dir"]
    null_dir = main_dir / args.subdir
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    events = load_manifest_events(Path(args.manifest) if args.manifest else DEFAULT_MANIFEST)
    channel_of = {(c, e): ch for c, e, ch in events}

    ref_csv = Path(args.reference_csv) if args.reference_csv else \
        cfg["paths"]["human_annotations_csv"].with_name("events_checked.csv")
    ref = load_reference(ref_csv) if ref_csv.exists() else {}

    modes = [args.mode] if args.mode else MODES

    # real = the main dataset's own run for the same item/mode/clip kind, which is the baseline
    # every degraded condition is measured against.
    real = {}
    for mode in modes:
        for clip_id, event_id, _ in events:
            run = load_json_run(main_dir / f"{clip_id}__{event_id}__combined__run1__{mode}.json")
            if run:
                real[(clip_id, event_id, mode)] = run

    degraded = defaultdict(dict)
    for path in sorted(null_dir.glob("*__combined__*__run1__*.json")):
        m = re.match(r"(?P<clip>[^_]+)__(?P<event>event\d)__combined__(?P<visual>[a-z]+)__"
                     r"run1__(?P<mode>\w+)\.json", path.name)
        if not m or m.group("mode") not in modes:
            continue
        run = load_json_run(path)
        if run:
            degraded[m.group("visual")][(m.group("clip"), m.group("event"),
                                         m.group("mode"))] = run

    if not degraded:
        log.error("no visual-null outputs found under %s", null_dir)
        return
    present = [v for v in VISUAL_CONDITIONS if v in degraded]

    # Restrict every condition, INCLUDING the real baseline, to the items all of them have. A
    # partially finished condition would otherwise be compared against a `real` row averaged over
    # more items, so the two columns would not be measuring the same events -- which is precisely
    # the comparison this whole file rests on.
    common = set.intersection(*(set(degraded[v]) for v in present)) & set(real)
    n_before = {v: len(degraded[v]) for v in present}
    for v in present:
        degraded[v] = {k: r for k, r in degraded[v].items() if k in common}
    real = {k: r for k, r in real.items() if k in common}
    log.info("conditions found: %s -> restricted to %d common items", n_before, len(common))
    if not common:
        log.error("no item is present in every condition; nothing comparable to report")
        return

    out = ["# Visual-null baselines: what survives losing the video?\n",
           f"\nModel outputs `{main_dir.name}`, clip kind `combined`, cue cell(s) "
           f"{', '.join('`' + m + '`' for m in modes)}. Each degraded condition is compared "
           f"against the main dataset's own run for the SAME item and cue cell, so the only "
           f"difference is pixel content -- the prompt is byte-identical throughout, including "
           f"under `none`, where it still refers to \"this clip\". Generated by "
           f"`scripts/analyze_visual_null.py`.\n",
           f"\nAll conditions, the real baseline included, are restricted to the "
           f"**{len(common)} items present in every condition**, so no two columns are averaged "
           f"over different events.\n"]

    # --- 1. did it answer at all -------------------------------------------------------------
    out.append("\n## 1. Did the model answer, and did it notice?\n")
    out.append("\nA refusal or hedge under `none` is a result, not a failed run: a model that "
               "declines to score a clip it was not shown is behaving better than one that scores "
               "it confidently. Both are counted.\n\n")
    out.append("| visual | mode | n | schema-valid | hedges/refuses | mean confidence |\n"
               + "|---" * 6 + "|\n")
    for visual in present:
        for mode in modes:
            rows = [(k, r) for k, r in degraded[visual].items() if k[2] == mode]
            if not rows:
                continue
            confs = [c for _, r in rows
                     if isinstance((c := (r["candidate"] or {}).get("confidence")), (int, float))]
            out.append(f"| `{visual}` | `{mode}` | {len(rows)} | "
                       f"{sum(1 for _, r in rows if r['schema_valid'])}/{len(rows)} | "
                       f"{sum(1 for _, r in rows if HEDGE_RE.search(r['raw']))}/{len(rows)} | "
                       f"{fmt(statistics.mean(confs), '{:.2f}') if confs else '--'} |\n")

    # --- 2. do the functional scores move ----------------------------------------------------
    out.append("\n## 2. Do the functional scores move when the video does not inform them?\n")
    out.append("\n**identical** = share of items whose score is exactly what the same item got "
               "with the real video. A high value means the score did not come from the pixels. "
               "**MAE vs ref.** is against the post-discussion expert reference, on all items "
               "(ungated), with the real run's own MAE for comparison.\n\n")
    out.append("| score | visual | mode | n | identical to real | mean (real) | MAE vs ref. |\n"
               + "|---" * 7 + "|\n")
    for label, col in SCORES:
        for visual in present + ["real"]:
            for mode in modes:
                if visual == "real":
                    pairs = [(k, r) for k, r in real.items() if k[2] == mode]
                else:
                    pairs = [(k, r) for k, r in degraded[visual].items() if k[2] == mode]
                vals, same, refs, mvals = [], 0, [], []
                for k, r in pairs:
                    v = (r["candidate"] or {}).get(col)
                    if not isinstance(v, (int, float)):
                        continue
                    vals.append(float(v))
                    rv = (real.get(k, {}).get("candidate") or {}).get(col)
                    if isinstance(rv, (int, float)) and float(rv) == float(v):
                        same += 1
                    key = (k[0], k[1])
                    if key in ref and ref[key]:
                        per_rater = [d[col] for d in ref[key].values() if col in d]
                        if per_rater:
                            refs.append(statistics.mean(per_rater))
                            mvals.append(float(v))
                if not vals:
                    continue
                mae = mean_absolute_error(refs, mvals) if refs else None
                out.append(f"| {label} | `{visual}` | `{mode}` | {len(vals)} | "
                           f"{'--' if visual == 'real' else f'{same}/{len(vals)}'} | "
                           f"{statistics.mean(vals):.2f} | {fmt(mae, '{:.2f}')} |\n")

    # --- 3. do the form descriptions collapse ------------------------------------------------
    out.append("\n## 3. Do the form descriptions collapse?\n")
    out.append("\nToken Jaccard between each condition's form description and the real run's, over "
               "all six content fields. Low = the description changed, i.e.\\ it depended on the "
               "pixels. Read against section 2: descriptions collapsing while scores hold is the "
               "dissociation stated behaviourally.\n\n")
    out.append("| visual | mode | n | J vs real (all fields) | J vs real (movement only) |\n"
               + "|---" * 5 + "|\n")
    for visual in present:
        for mode in modes:
            allf, mvf = [], []
            for k, r in degraded[visual].items():
                if k[2] != mode or k not in real:
                    continue
                d = (r["candidate"] or {}).get("form_description") or {}
                b = (real[k]["candidate"] or {}).get("form_description") or {}
                allf.append(jaccard(tokens(" ".join((d.get(f) or "") for f in FORM_FIELDS)),
                                    tokens(" ".join((b.get(f) or "") for f in FORM_FIELDS))))
                mvf.append(jaccard(tokens(d.get("movement") or ""),
                                   tokens(b.get("movement") or "")))
            if not allf:
                continue
            out.append(f"| `{visual}` | `{mode}` | {len(allf)} | "
                       f"{statistics.mean(allf):.3f} | {statistics.mean(mvf):.3f} |\n")

    # --- 4. self-selected articulator --------------------------------------------------------
    out.append("\n## 4. Which articulator does it name when it cannot see one?\n")
    out.append("\nArticulator-withheld cells only, where `channel_judged` is a measurement rather "
               "than a compliance check. If the distribution with no video matches the "
               "distribution with the real video, the choice was not being made from the video.\n\n")
    withheld = [m for m in modes if m in ("ablate_channel", "ablate_both")]
    if withheld:
        out.append("| visual | mode | n | channel_judged distribution | matches target | "
                   "matches real run |\n" + "|---" * 6 + "|\n")
        for visual in present + ["real"]:
            for mode in withheld:
                src = real if visual == "real" else degraded[visual]
                rows = [(k, r) for k, r in src.items() if k[2] == mode]
                judged = {k: ((r["candidate"] or {}).get("channel_judged") or "").strip()
                          for k, r in rows}
                judged = {k: v for k, v in judged.items() if v}
                if not judged:
                    continue
                hits = sum(1 for k, v in judged.items() if v == channel_of[(k[0], k[1])])
                same = sum(1 for k, v in judged.items()
                           if v == ((real.get(k, {}).get("candidate") or {})
                                    .get("channel_judged") or "").strip())
                out.append(f"| `{visual}` | `{mode}` | {len(judged)} | "
                           f"{dict(sorted(Counter(judged.values()).items()))} | "
                           f"{hits}/{len(judged)} | "
                           f"{'--' if visual == 'real' else f'{same}/{len(judged)}'} |\n")
    else:
        out.append("\nNot run for an articulator-withheld cell, so this section is empty.\n")

    # --- 5. per-articulator D1, the the paper's degraded-video table under each condition ---------------------
    out.append("\n## 5. D1 by target articulator, per visual condition\n")
    out.append("\nThe paper's articulator->D1 mapping, recomputed with the video degraded. If the "
               "mapping is unchanged with no video at all, it is a prior over the label, not a "
               "judgement about an action.\n\n")
    out.append("| visual | mode | " + " | ".join(f"`{c}`" for c in CHANNELS) + " |\n"
               + "|---" * (len(CHANNELS) + 2) + "|\n")
    for visual in present + ["real"]:
        for mode in modes:
            src = real if visual == "real" else degraded[visual]
            rows = [(k, r) for k, r in src.items() if k[2] == mode]
            if not rows:
                continue
            by_channel = defaultdict(list)
            for k, r in rows:
                v = (r["candidate"] or {}).get("semantic_syntactic_load")
                if isinstance(v, (int, float)):
                    by_channel[channel_of[(k[0], k[1])]].append(int(v))
            out.append(f"| `{visual}` | `{mode}` | " + " | ".join(
                str(dict(sorted(Counter(by_channel[c]).items()))) if by_channel.get(c) else "--"
                for c in CHANNELS) + " |\n")

    out_path = results_dir / "visual_null.md"
    out_path.write_text("".join(out), encoding="utf-8")
    log.info("Wrote %s", out_path)
    print("".join(out))


if __name__ == "__main__":
    main()
