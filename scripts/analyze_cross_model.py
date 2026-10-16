#!/usr/bin/env python3
"""Cross-model comparison on the analyses that need NO human rating.

These analyses need only model output and clip_manifest.csv, so any further model can be
compared on them without new expert rating.

The three analyses, each computable from model output plus clip_manifest.csv alone:

  1. SCHEMA COMPLIANCE. Reported first because it gates the rest and because it is itself a
     finding: the paper's claims presuppose a model that can be *asked* these questions. A model
     that cannot emit the schema has not failed at sign linguistics, it has failed at instruction
     following, and the two must not be reported as one number.

  2. ARTICULATOR SELF-SELECTION. Under the articulator-withheld cells, per-class recall and the
     non-dominant-hand rate on the positive trials. The paper's headline negative result. If a
     second model also never self-selects the non-dominant hand, the finding stops being a
     property of one checkpoint.

  3. ARTICULATOR-DETERMINISM OF D1. The share of each model's own D1 outputs equal to the modal
     value for their target articulator -- computed from model output only, with no reference
     involved, which is what makes it portable across models. Plus the D1 flip: does D1 change
     wholesale for the same events when only the articulator cue changes?

  4. THE RELABELLING TEST. J(X,A) vs J(X,B) from the relabelling analysis, recomputed per model.

CLIP KIND IS HELD FIXED. The paper compares models on `combined` (window clip and target-span clip
in one call), so the Qwen column is the main run itself. The script refuses a clip kind that a
model directory lacks rather than silently comparing different input formats.

Usage:
    python scripts/analyze_cross_model.py --models qwen3vl_8b glm46v_flash gemma4_e4b mimo_vl_7b \
        --clip-kind combined --out data/results/crossmodel/cross_model_all4_combined.md
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

log = get_logger("analyze_cross_model")

DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"
MODES = ["grounded", "ablate_translation", "ablate_channel", "ablate_both"]
CHANNEL_WITHHELD_MODES = ["ablate_channel", "ablate_both"]
CUE_PAIRS = [("grounded", "ablate_channel"), ("ablate_translation", "ablate_both")]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]
ACTION_FIELDS = ["handshape", "movement", "location"]
EVENT_CLIP_KIND = {"event1": "ev1", "event2": "ev2"}

STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "then", "with", "is",
             "are", "as", "it", "its", "s", "signer", "signer's", "hand", "hands"}


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


def resolve_kind(clip_kind, event_id):
    """`event` means "whichever tight event cut carries this event" -- ev1 for event1, ev2 for
    event2. Written as one option so a caller cannot accidentally compare event1 rows against
    event2 rows, or drop half the events by asking for `ev1` alone."""
    return EVENT_CLIP_KIND[event_id] if clip_kind == "event" else clip_kind


def load_model_runs(model_dir, events, clip_kind):
    """{mode: {(clip_id, event_id): {"candidate": .., "schema_valid": bool, "raw": str}}}"""
    runs = {}
    for mode in MODES:
        runs[mode] = {}
        for clip_id, event_id, _ in events:
            kind = resolve_kind(clip_kind, event_id)
            path = model_dir / f"{clip_id}__{event_id}__{kind}__run1__{mode}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            cands = (data.get("parsed") or {}).get("candidate_actions") or []
            runs[mode][(clip_id, event_id)] = {
                "candidate": cands[0] if cands else None,
                "schema_valid": bool(data.get("schema_valid")),
                "raw": data.get("raw_text") or "",
                "model_version": data.get("model_version"),
            }
    return runs


def fmt(value, spec="{:.3f}"):
    return spec.format(value) if isinstance(value, (int, float)) else "--"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True,
                        help="model_outputs_dir tags to compare, e.g. qwen3vl_8b glm46v_flash")
    parser.add_argument("--outputs-root", default=None,
                        help="parent of the model tags (default: data/model_outputs)")
    parser.add_argument("--clip-kind", default="event",
                        choices=["event", "ev1", "ev2", "window", "combined"],
                        help="`event` (default) uses each event's own tight cut (ev1/ev2). "
                             "`combined` is the paper's format (window and target-span clip in one call).")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--no-common-items", action="store_true",
                        help="do NOT restrict to items every model has. Only for inspecting one "
                             "model's coverage -- with two or more models this makes the rate "
                             "columns non-comparable.")
    parser.add_argument("--out", default=None,
                        help="output path (default: data/results/cross_model_<clip_kind>.md)")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root) if args.outputs_root \
        else (REPO_ROOT / "data" / "model_outputs")
    events = load_manifest_events(Path(args.manifest) if args.manifest else DEFAULT_MANIFEST)
    channel_of = {(c, e): ch for c, e, ch in events}

    per_model = {}
    for tag in args.models:
        model_dir = outputs_root / tag
        if not model_dir.exists():
            log.error("no such model output dir: %s -- skipping", model_dir)
            continue
        runs = load_model_runs(model_dir, events, args.clip_kind)
        found = sum(len(v) for v in runs.values())
        if found == 0:
            log.error("%s has no runs for clip kind '%s'. Refusing to substitute a different "
                      "clip kind: the comparison is only meaningful if every model saw the same "
                      "input format.", tag, args.clip_kind)
            continue
        per_model[tag] = runs
        log.info("%s: %s runs per mode", tag, {m: len(runs[m]) for m in MODES})

    if not per_model:
        log.error("nothing to compare")
        return

    # Restrict every model to the items ALL of them have, per cue cell. Without this, a partially
    # completed run on one model would be compared against a complete run on another and the
    # per-class rates would be computed over different item sets -- the n columns would differ but
    # the rates would silently not be like-for-like, which is the one thing this whole script
    # exists to guarantee.
    if len(per_model) > 1 and not args.no_common_items:
        dropped = 0
        for mode in MODES:
            common = set.intersection(*(set(runs[mode]) for runs in per_model.values()))
            for runs in per_model.values():
                for key in list(runs[mode]):
                    if key not in common:
                        del runs[mode][key]
                        dropped += 1
        if dropped:
            log.warning("restricted to items common to all models: dropped %d model-item rows. "
                        "Per-mode common item counts: %s", dropped,
                        {m: len(next(iter(per_model.values()))[m]) for m in MODES})

    versions = {}
    for tag, runs in per_model.items():
        vs = {r["model_version"] for mode in MODES for r in runs[mode].values()
              if r.get("model_version")}
        versions[tag] = ", ".join(sorted(vs)) or "unrecorded"

    out = ["# Cross-model comparison, reference-free analyses only\n",
           f"\nClip kind `{args.clip_kind}`"
           + (" (each event's own tight cut: ev1 for event1, ev2 for event2)"
              if args.clip_kind == "event" else "") +
           f", {len(events)} events, one greedy run per item. Generated by "
           f"`scripts/analyze_cross_model.py`.\n",
           "\nNo human rating enters any number below, which is what makes these three analyses "
           "portable to a new checkpoint at zero expert cost. "
           + ("This is the `combined` kind the paper's main tables use, so the Qwen column is the "
              "main run itself.\n" if args.clip_kind == "combined" else
              "Note the clip kind: it is NOT the `combined` kind the paper's main tables use "
              "(this kind sends one video per call). Every column is recomputed on this "
              "kind so they are like-for-like; they are the same analyses as the main tables, not "
              "the same numbers.\n"),
           ("\nEvery model is restricted to the items all of them have, per cue cell, so the rate "
            "columns are computed over one item set.\n" if len(per_model) > 1
            and not args.no_common_items else
            "\nItem sets are NOT intersected across models (`--no-common-items`), so rate columns "
            "may be computed over different items.\n"),
           "\n| tag | checkpoint |\n|---|---|\n"]
    for tag in per_model:
        out.append(f"| `{tag}` | {versions[tag]} |\n")

    # --- 1. schema compliance ---------------------------------------------------------------
    out.append("\n## 1. Can the model be asked the question at all?\n")
    out.append("\nSchema-valid share of runs, per cue cell. This gates everything below: a model "
               "that cannot emit the output schema has failed at instruction following, not at "
               "sign linguistics, and reporting the two together would credit the wrong "
               "explanation.\n\n")
    out.append("| model | " + " | ".join(f"`{m}`" for m in MODES) + " | overall |\n"
               + "|---" * (len(MODES) + 2) + "|\n")
    for tag, runs in per_model.items():
        cells, total_ok, total_n = [], 0, 0
        for mode in MODES:
            rs = list(runs[mode].values())
            ok = sum(1 for r in rs if r["schema_valid"])
            total_ok += ok
            total_n += len(rs)
            cells.append(f"{ok}/{len(rs)}" if rs else "--")
        out.append(f"| `{tag}` | " + " | ".join(cells) +
                   f" | {total_ok}/{total_n} = {total_ok / total_n:.3f} |\n"
                   if total_n else f"| `{tag}` | " + " | ".join(cells) + " | -- |\n")

    # --- 2. articulator self-selection ------------------------------------------------------
    out.append("\n## 2. Articulator self-selection, articulator-withheld cells\n")
    out.append("\nPer-class recall of the manifest's `target_channel`, and the non-dominant-hand "
               "rate on positive trials only (7 events x 2 withheld cells = 14). Runs whose "
               "output did not parse are excluded and counted separately -- scoring an unparseable "
               "run as a wrong articulator would let an instruction-following failure masquerade "
               "as a perceptual one.\n\n")
    out.append("| model | cell | scored | " + " | ".join(f"{c} recall" for c in CHANNELS) +
               " | balanced acc. |\n" + "|---" * (len(CHANNELS) + 4) + "|\n")
    ndh_summary = {}
    for tag, runs in per_model.items():
        ndh_hits, ndh_total = 0, 0
        for mode in CHANNEL_WITHHELD_MODES:
            scored = {k: (r["candidate"] or {}).get("channel_judged")
                      for k, r in runs[mode].items() if r["candidate"]}
            scored = {k: (v or "").strip() for k, v in scored.items() if v}
            if not scored:
                out.append(f"| `{tag}` | `{mode}` | 0 | " + " | ".join("--" for _ in CHANNELS) +
                           " | -- |\n")
                continue
            recalls, cells = [], []
            for channel in CHANNELS:
                keys = [k for k in scored if channel_of[k] == channel]
                if not keys:
                    cells.append("--")
                    continue
                hits = sum(1 for k in keys if scored[k] == channel)
                recalls.append(hits / len(keys))
                cells.append(f"{hits}/{len(keys)}")
                if channel == "non_dominant_hand":
                    ndh_hits += hits
                    ndh_total += len(keys)
            out.append(f"| `{tag}` | `{mode}` | {len(scored)}/{len(runs[mode])} | " +
                       " | ".join(cells) +
                       f" | {statistics.mean(recalls):.3f} |\n")
        ndh_summary[tag] = (ndh_hits, ndh_total)

    out.append("\n**Non-dominant-hand recall on positive trials:** " + "; ".join(
        f"`{tag}` {h}/{n}" for tag, (h, n) in ndh_summary.items() if n) + ".\n")

    out.append("\nWhere the non-dominant-hand misses go, per model:\n\n")
    out.append("| item | " + " | ".join(f"`{t}` / {m}" for t in per_model
                                        for m in CHANNEL_WITHHELD_MODES) + " |\n"
               + "|---" * (len(per_model) * len(CHANNEL_WITHHELD_MODES) + 1) + "|\n")
    for k in sorted(k for k in channel_of if channel_of[k] == "non_dominant_hand"):
        cells = []
        for tag, runs in per_model.items():
            for mode in CHANNEL_WITHHELD_MODES:
                cand = (runs[mode].get(k) or {}).get("candidate")
                cells.append((cand or {}).get("channel_judged") or "--")
        out.append(f"| {k[0]}/{k[1]} | " + " | ".join(str(c) for c in cells) + " |\n")

    # --- 3. articulator-determinism of D1 ---------------------------------------------------
    out.append("\n## 3. Is D1 determined by the target articulator?\n")
    out.append("\nShare of a model's own D1 outputs equal to the modal D1 for their target "
               "articulator. Computed from model output only -- no reference scores are involved "
               "-- which is exactly why it transfers to a new checkpoint for free. 100% means the "
               "model emitted one value per articulator and no item-level variation survived.\n\n")
    out.append("| model | cell | n | determinism | D1 by articulator |\n" + "|---" * 5 + "|\n")
    for tag, runs in per_model.items():
        for mode in MODES:
            by_channel = defaultdict(list)
            for k, r in runs[mode].items():
                d1 = (r["candidate"] or {}).get("semantic_syntactic_load")
                if isinstance(d1, (int, float)):
                    by_channel[channel_of[k]].append(int(d1))
            n = sum(len(v) for v in by_channel.values())
            if not n:
                out.append(f"| `{tag}` | `{mode}` | 0 | -- | -- |\n")
                continue
            modal_hits = sum(Counter(v).most_common(1)[0][1] for v in by_channel.values())
            spread = "; ".join(f"{ch} {dict(sorted(Counter(by_channel[ch]).items()))}"
                               for ch in CHANNELS if by_channel.get(ch))
            out.append(f"| `{tag}` | `{mode}` | {n} | {modal_hits / n:.3f} | {spread} |\n")

    out.append("\n### The D1 flip: same events, articulator named vs withheld\n")
    out.append("\nThe paper's per-articulator table. Per articulator, the model's D1 distribution with the cue "
               "given against the same events with it withheld.\n\n")
    out.append("| model | articulator | n | named | withheld |\n" + "|---" * 5 + "|\n")
    for tag, runs in per_model.items():
        for channel in CHANNELS:
            keys = [k for k in sorted(runs["grounded"]) if channel_of[k] == channel
                    and k in runs["ablate_channel"]]
            if not keys:
                continue
            def dist(mode):
                vals = [(runs[mode][k]["candidate"] or {}).get("semantic_syntactic_load")
                        for k in keys]
                return dict(sorted(Counter(v for v in vals
                                           if isinstance(v, (int, float))).items()))
            out.append(f"| `{tag}` | `{channel}` | {len(keys)} | {dist('grounded')} | "
                       f"{dist('ablate_channel')} |\n")

    # --- 4. relabelling test ----------------------------------------------------------------
    out.append("\n## 4. Does naming the articulator redirect the model, or relabel what it saw?\n")
    out.append("\nthe relabelling analysis, per model. **X** = the target-action text with the articulator "
               "named; **A** = the same event with it withheld (the articulator the model chose "
               "itself); **B** = that same withheld run's own report of the non-dominant hand. "
               "`manual` is the no-switch baseline (the cue names what the model picks anyway) and "
               "`face` is the positive control (the cue can redirect). The question is which of "
               "those two the non-dominant hand resembles.\n\n")
    out.append("\n$J(X,B)$ requires the withheld run to have actually filled the "
               "`non_dominant_hand` description field. A model that omits it would score "
               "$J(X,B)=0$ on every item and pass the $A{>}B$ test trivially, so items whose "
               "withheld run lacks the field are EXCLUDED from the $B$ columns and counted "
               "separately, rather than being scored as a maximal difference.\n\n")
    out.append("| model | cue pair | articulator | n | J(X,A) | n(B) | J(X,B) | A>B |\n"
               + "|---" * 8 + "|\n")
    for tag, runs in per_model.items():
        for given, withheld in CUE_PAIRS:
            for channel in CHANNELS:
                ja, pairs_ab = [], []
                for k in sorted(runs[given]):
                    if channel_of[k] != channel or k not in runs[withheld]:
                        continue
                    g = runs[given][k]["candidate"]
                    w = runs[withheld][k]["candidate"]
                    if not g or not w:
                        continue
                    gf = g.get("form_description") or {}
                    wf = w.get("form_description") or {}
                    x = tokens(" ".join((gf.get(f) or "") for f in ACTION_FIELDS))
                    a = tokens(" ".join((wf.get(f) or "") for f in ACTION_FIELDS))
                    ja.append(jaccard(x, a))
                    ndh_text = wf.get("non_dominant_hand")
                    if ndh_text is None or not str(ndh_text).strip():
                        continue
                    pairs_ab.append((jaccard(x, a), jaccard(x, tokens(str(ndh_text)))))
                if not ja:
                    continue
                if pairs_ab:
                    b_cols = (f"{len(pairs_ab)} | "
                              f"{statistics.mean([p[1] for p in pairs_ab]):.3f} | "
                              f"{sum(1 for p, q in pairs_ab if p > q)}/{len(pairs_ab)}")
                else:
                    b_cols = "0 | field absent | n/a"
                out.append(f"| `{tag}` | `{given}`/`{withheld}` | `{channel}` | {len(ja)} | "
                           f"{statistics.mean(ja):.3f} | {b_cols} |\n")

    out_path = Path(args.out) if args.out else \
        ensure_dir(REPO_ROOT / "data" / "results") / f"cross_model_{args.clip_kind}.md"
    out_path.write_text("".join(out), encoding="utf-8")
    log.info("Wrote %s", out_path)
    print("".join(out))


if __name__ == "__main__":
    main()
