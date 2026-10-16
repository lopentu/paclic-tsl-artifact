#!/usr/bin/env python3
"""Read out the mis-cue condition (26) and the description->D1 mediation probe (27).

Both conditions exist to answer the same question with a manipulation rather than a correlation:
when the model's answer changes with the articulator cue, is it changing what it SAW or what it
CALLED what it saw?

## Mis-cue (26)

Every target is cued with the wrong articulator, across the manual/nonmanual boundary. The two
accounts make opposite predictions and the readouts are all reference-free:

  1. **Label compliance** -- does `channel_judged` echo the false label? Under the true label it
     is 55/55, so anything less is the model resisting.
  2. **Field routing** -- the behavioural signature of genuine redirection, already established
     on real face targets: with the face named, the model marks the hand fields not applicable
     and puts the substantive description in `nonmanual` (15/15). If a mis-cue redirects, a
     manual target cued as face should show that same signature. This is the decisive readout,
     because it does not depend on any similarity measure.
  3. **Description similarity** -- how close the mis-cued description is to the same item's
     true-cued description (G) and to its uncued, self-selected description (W).
  4. **Functional scores** -- whether D1 follows the FALSE label's modal value rather than the
     item's true one. This tests the paper's label-driven claim directly, on all 55 events.
  5. **Objections** -- a model that notices the mismatch and says so is a result, not a failure.

## Mediation (27)

D1 asked for from the model's own form description as text, with and without the articulator
label, compared against the D1 that same description's run produced from video.

Usage:
    python scripts/29_analyze_miscue_and_mediation.py
    python scripts/29_analyze_miscue_and_mediation.py --skip-mediation
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir, REPO_ROOT, parse_manifest_event

log = get_logger("analyze_miscue")

DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"
ACTION_FIELDS = ["handshape", "movement", "location"]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]
SCORE_KEYS = [("D1", "semantic_syntactic_load"),
              ("D2", "pragmatic_depictive_affective_enrichment"),
              ("description", "description"), ("indication", "indication"),
              ("depiction", "depiction")]
MISCUE_OF = {"manual": "face", "non_dominant_hand": "face", "head_torso": "face",
             "face": "manual"}

STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "then", "with", "is",
             "are", "as", "it", "its", "s", "signer", "signer's", "hand", "hands"}

# "This articulator is not the one acting" -- the routing signal. Matched against the whole
# field value, not searched within it, so a description that merely mentions stillness in
# passing does not count as a vacated field.
VACANT_RE = re.compile(r"^\s*(not applicable|not visible|none|no visible[^.]*|n/?a|-+)\s*\.?\s*$",
                       re.IGNORECASE)
# A model that notices it has been handed the wrong articulator.
OBJECTION_RE = re.compile(
    r"\bno (?:visible |significant )?(?:facial|manual|hand)\b|\bnot carried by\b|"
    r"\bdoes not appear to be\b|\bcontrary to\b|\bhowever[, ]|\bactually\b|"
    r"\bappears to be carried by\b|\bmismatch\b|\bincorrect\b", re.IGNORECASE)


def tokens(text):
    return {t for t in re.findall(r"[a-z']+", (text or "").lower()) if t not in STOPWORDS}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


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
    if not path.exists():
        return None, ""
    data = json.loads(path.read_text(encoding="utf-8"))
    cands = (data.get("parsed") or {}).get("candidate_actions") or []
    return (cands[0] if cands else None), (data.get("raw_text") or "")


def field(candidate, name):
    return ((candidate or {}).get("form_description", {}) or {}).get(name, "") or ""


def vacant(candidate, name):
    return bool(VACANT_RE.match(field(candidate, name).strip()))


def action_text(candidate):
    return " ".join(field(candidate, k) for k in ACTION_FIELDS)


# --------------------------------------------------------------------------------------
# mis-cue
# --------------------------------------------------------------------------------------

def section_miscue(out, main_dir, events, mode="grounded"):
    miscue_dir = main_dir / "miscue"
    rows = []
    for clip_id, event_id, channel, video_id in events:
        m, m_raw = candidate_of(miscue_dir / f"{clip_id}__{event_id}__combined__run1__"
                                             f"miscue_{mode}.json")
        g, _ = candidate_of(main_dir / f"{clip_id}__{event_id}__combined__run1__{mode}.json")
        w, _ = candidate_of(main_dir / f"{clip_id}__{event_id}__combined__run1__ablate_channel.json")
        if m is None or g is None:
            continue
        rows.append({"clip_id": clip_id, "event_id": event_id, "true": channel,
                     "cued": MISCUE_OF[channel], "video_id": video_id,
                     "m": m, "g": g, "w": w, "raw": m_raw})
    if not rows:
        out.append("\n_(no mis-cue outputs found)_\n")
        return rows

    out.append(f"\n## Mis-cue condition ({len(rows)} events, cue cell `{mode}`)\n")
    out.append("\nEvery target cued with the wrong articulator, across the manual/nonmanual "
               "boundary. Prompt byte-identical to the true-cue run apart from that one "
               "phrase.\n")

    # --- 1. does the label get echoed --------------------------------------------------
    out.append("\n### 1. Does `channel_judged` follow the false label?\n\n")
    out.append("| true articulator | n | cued as | judged = FALSE label | judged = TRUE "
               "articulator | other |\n" + "|---" * 6 + "|\n")
    for channel in CHANNELS:
        group = [r for r in rows if r["true"] == channel]
        if not group:
            continue
        follows = sum(1 for r in group if r["m"].get("channel_judged") == r["cued"])
        resists = sum(1 for r in group if r["m"].get("channel_judged") == r["true"])
        out.append(f"| `{channel}` | {len(group)} | `{MISCUE_OF[channel]}` | "
                   f"{follows}/{len(group)} | {resists}/{len(group)} | "
                   f"{len(group) - follows - resists}/{len(group)} |\n")
    total_follows = sum(1 for r in rows if r["m"].get("channel_judged") == r["cued"])
    out.append(f"\nOverall {total_follows}/{len(rows)} echo the false label, against 55/55 "
               f"compliance when the label is correct.\n")

    # --- 2. field routing: the decisive readout ----------------------------------------
    out.append("\n### 2. Does the DESCRIPTION move to the named articulator?\n")
    out.append("\nThe established signature of genuine redirection: with the face named on a "
               "real face target, the model vacates the hand fields (`not applicable` / `not "
               "visible`) and puts substance in `nonmanual`. Reference column is that same "
               "signature measured on the true-cue runs.\n\n")
    out.append("| true articulator | cued as | n | hand fields vacated (mis-cue) | "
               "(true cue) | `nonmanual` substantive (mis-cue) | (true cue) |\n"
               + "|---" * 7 + "|\n")
    for channel in CHANNELS:
        group = [r for r in rows if r["true"] == channel]
        if not group:
            continue
        def frac(key, fn):
            return f"{sum(1 for r in group if fn(r[key]))}/{len(group)}"
        out.append(
            f"| `{channel}` | `{MISCUE_OF[channel]}` | {len(group)} | "
            f"{frac('m', lambda c: vacant(c, 'handshape'))} | "
            f"{frac('g', lambda c: vacant(c, 'handshape'))} | "
            f"{frac('m', lambda c: not vacant(c, 'nonmanual') and field(c, 'nonmanual').strip() != '')} | "
            f"{frac('g', lambda c: not vacant(c, 'nonmanual') and field(c, 'nonmanual').strip() != '')} |\n")

    # --- 3. similarity ------------------------------------------------------------------
    out.append("\n### 3. What is the mis-cued description close to?\n")
    out.append("\n`M` the mis-cued description, `G` the same item's true-cued description, `W` "
               "its uncued self-selected description. High J(M,G) means the false label did not "
               "move the content.\n\n")
    out.append("| true articulator | cued as | n | J(M,G) | C(M,G) | J(M,W) | "
               "identical `movement` string to G |\n" + "|---" * 7 + "|\n")
    for channel in CHANNELS:
        group = [r for r in rows if r["true"] == channel]
        if not group:
            continue
        jmg, cmg, jmw, same = [], [], [], 0
        for r in group:
            tm, tg = tokens(action_text(r["m"])), tokens(action_text(r["g"]))
            jmg.append(jaccard(tm, tg))
            cmg.append(containment(tm, tg))
            if r["w"] is not None:
                jmw.append(jaccard(tm, tokens(action_text(r["w"]))))
            if field(r["m"], "movement").strip().lower() == \
               field(r["g"], "movement").strip().lower():
                same += 1
        out.append(f"| `{channel}` | `{MISCUE_OF[channel]}` | {len(group)} | "
                   f"{statistics.mean(jmg):.3f} | {statistics.mean(cmg):.3f} | "
                   f"{statistics.mean(jmw):.3f} | {same}/{len(group)} |\n")

    # --- 3b. is the "redirected" content newly computed, or promoted? --------------------
    # A manual target cued as FACE that vacates its hand fields has apparently redirected. But
    # the true-cue run ALREADY filled its own `nonmanual` field, as background detail, on nearly
    # every item. So the cue may not be redirecting attention at all -- it may be promoting
    # content the model had already produced to target status. The test: compare the mis-cued
    # run's target description against the TRUE-cue run's own field for the falsely named
    # articulator. High similarity = promotion of already-computed content, not new looking.
    out.append("\n### 3b. Is the redirected content new, or the true-cue run's own background "
               "field promoted?\n")
    out.append("\nFor each mis-cued run, its target description is compared against the "
               "**true-cue run's own field for the articulator that was falsely named** -- the "
               "`nonmanual` field for targets cued as face, the action fields for the face "
               "targets cued as manual. If the cue redirected attention, the mis-cued "
               "description should contain something the true-cue run did not already say; if it "
               "promoted, the two should coincide.\n\n")
    out.append("| true articulator | cued as | n | J(M-target, G-field-of-cued) | "
               "C(M-target, G-field-of-cued) | identical string |\n" + "|---" * 6 + "|\n")
    for channel in CHANNELS:
        group = [r for r in rows if r["true"] == channel]
        if not group:
            continue
        js, cs, same = [], [], 0
        for r in group:
            if r["cued"] == "face":
                m_txt, g_txt = field(r["m"], "nonmanual"), field(r["g"], "nonmanual")
            else:
                m_txt, g_txt = action_text(r["m"]), action_text(r["g"])
            tm, tg = tokens(m_txt), tokens(g_txt)
            js.append(jaccard(tm, tg))
            cs.append(containment(tm, tg))
            if m_txt.strip().lower() == g_txt.strip().lower():
                same += 1
        out.append(f"| `{channel}` | `{MISCUE_OF[channel]}` | {len(group)} | "
                   f"{statistics.mean(js):.3f} | {statistics.mean(cs):.3f} | "
                   f"{same}/{len(group)} |\n")

    # --- 4. do the functional scores follow the false label? ----------------------------
    # The label-driven prediction: D1 under mis-cue should match the modal D1 the TRUE-cue runs
    # give to the articulator that was falsely named, not the one the item actually has.
    modal_by_channel = {}
    for label, key in SCORE_KEYS:
        modal_by_channel[key] = {}
        for channel in CHANNELS:
            vals = [r["g"].get(key) for r in rows
                    if r["true"] == channel and isinstance(r["g"].get(key), (int, float))]
            if vals:
                modal_by_channel[key][channel] = Counter(vals).most_common(1)[0][0]

    out.append("\n### 4. Do the functional scores follow the false label?\n")
    out.append("\n`modal(true)` and `modal(cued)` are the true-cue runs' modal values for the "
               "item's real articulator and for the one falsely named. If the score is driven by "
               "the label, mis-cued output should sit at `modal(cued)`.\n\n")
    out.append("| score | true articulator | n | mean (true cue) | mean (mis-cue) | "
               "modal(true) | modal(cued) | at modal(cued) | at modal(true) |\n"
               + "|---" * 9 + "|\n")
    for label, key in SCORE_KEYS:
        for channel in CHANNELS:
            group = [r for r in rows if r["true"] == channel
                     and isinstance(r["m"].get(key), (int, float))
                     and isinstance(r["g"].get(key), (int, float))]
            if not group:
                continue
            mt = modal_by_channel[key].get(channel)
            mc = modal_by_channel[key].get(MISCUE_OF[channel])
            out.append(
                f"| {label} | `{channel}` | {len(group)} | "
                f"{statistics.mean(r['g'][key] for r in group):.2f} | "
                f"{statistics.mean(r['m'][key] for r in group):.2f} | "
                f"{mt} | {mc} | "
                f"{sum(1 for r in group if r['m'][key] == mc)}/{len(group)} | "
                f"{sum(1 for r in group if r['m'][key] == mt)}/{len(group)} |\n")
        out.append("|" + " |" * 9 + "\n")

    # --- 5. objections and confidence ---------------------------------------------------
    out.append("\n### 5. Does the model object, and how confident is it?\n\n")
    out.append("| true articulator | n | mean stated confidence (mis-cue) | (true cue) | "
               "output flags a mismatch |\n" + "|---" * 5 + "|\n")
    for channel in CHANNELS:
        group = [r for r in rows if r["true"] == channel]
        if not group:
            continue
        cm = [r["m"]["confidence"] for r in group
              if isinstance(r["m"].get("confidence"), (int, float))]
        cg = [r["g"]["confidence"] for r in group
              if isinstance(r["g"].get("confidence"), (int, float))]
        obj = sum(1 for r in group
                  if OBJECTION_RE.search(r["m"].get("observable_evidence", "") or ""))
        out.append(f"| `{channel}` | {len(group)} | "
                   f"{statistics.mean(cm):.2f} | {statistics.mean(cg):.2f} | "
                   f"{obj}/{len(group)} |\n")

    out.append("\n### Per-item detail\n\n")
    out.append("| item | true | cued | judged | D1 true | D1 mis | J(M,G) | hands vacated |\n"
               + "|---" * 8 + "|\n")
    for r in rows:
        out.append(f"| {r['clip_id']}/{r['event_id']} | {r['true']} | {r['cued']} | "
                   f"{r['m'].get('channel_judged')} | "
                   f"{r['g'].get('semantic_syntactic_load')} | "
                   f"{r['m'].get('semantic_syntactic_load')} | "
                   f"{jaccard(tokens(action_text(r['m'])), tokens(action_text(r['g']))):.2f} | "
                   f"{'yes' if vacant(r['m'], 'handshape') else 'no'} |\n")
    return rows


# --------------------------------------------------------------------------------------
# mediation
# --------------------------------------------------------------------------------------

def section_mediation(out, main_dir, events, mode="grounded", subdir="mediation"):
    med_dir = main_dir / subdir
    rows = []
    for clip_id, event_id, channel, video_id in events:
        video, _ = candidate_of(main_dir / f"{clip_id}__{event_id}__combined__run1__{mode}.json")
        if video is None:
            continue
        arms = {}
        for arm in ("nolabel", "withlabel", "mislabel"):
            path = med_dir / f"{clip_id}__{event_id}__{arm}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            arms[arm] = data.get("parsed") or {}
        # mislabel is optional: it was added after the first two arms had already run, so the
        # report has to stay readable on the 110-run state as well as the 165-run one.
        if "nolabel" not in arms or "withlabel" not in arms:
            continue
        rows.append({"clip_id": clip_id, "event_id": event_id, "channel": channel,
                     "video": video, **arms})
    if not rows:
        out.append("\n_(no mediation outputs found)_\n")
        return rows

    has_mis = [r for r in rows if "mislabel" in r]
    out.append(f"\n## Description -> D1 mediation probe ({len(rows)} events, `{subdir}/`)\n")
    out.append("\nThe model's own form description handed back as text, no video, with no "
               "articulator label, with the true one, and -- in the third arm -- with a FALSE "
               "one. Agreement with the video run's score means that score was recoverable from "
               "the description alone.\n")
    if has_mis:
        out.append(f"\nThe `mislabel` arm ({len(has_mis)}/{len(rows)} events) is the sharpest "
                   "test in the study: a true description and a false label are put in conflict "
                   "with no video in the room at all, so whichever the score follows is the one "
                   "driving it. There is no visual confound left to appeal to.\n")
    else:
        out.append("\n_(the `mislabel` arm has not been run; columns for it are omitted)_\n")
    out.append("\n")

    for label, key in SCORE_KEYS:
        out.append(f"\n**{label}**\n\n")
        head = ["articulator", "n", "video mean", "nolabel mean", "withlabel mean"]
        if has_mis:
            head.append("mislabel mean")
        head += ["nolabel = video", "withlabel = video", "nolabel = withlabel"]
        out.append("| " + " | ".join(head) + " |\n" + "|---" * len(head) + "|\n")
        for channel in CHANNELS + ["ALL"]:
            group = [r for r in rows if channel == "ALL" or r["channel"] == channel]
            group = [r for r in group
                     if isinstance(r["video"].get(key), (int, float))
                     and isinstance(r["nolabel"].get(key), (int, float))
                     and isinstance(r["withlabel"].get(key), (int, float))]
            if not group:
                continue
            cells = [f"`{channel}`", str(len(group)),
                     f"{statistics.mean(r['video'][key] for r in group):.2f}",
                     f"{statistics.mean(r['nolabel'][key] for r in group):.2f}",
                     f"{statistics.mean(r['withlabel'][key] for r in group):.2f}"]
            if has_mis:
                mis = [r['mislabel'][key] for r in group
                       if isinstance(r.get("mislabel", {}).get(key), (int, float))]
                cells.append(f"{statistics.mean(mis):.2f}" if mis else "--")
            cells += [
                f"{sum(1 for r in group if r['nolabel'][key] == r['video'][key])}/{len(group)}",
                f"{sum(1 for r in group if r['withlabel'][key] == r['video'][key])}/{len(group)}",
                f"{sum(1 for r in group if r['nolabel'][key] == r['withlabel'][key])}/{len(group)}"]
            out.append("| " + " | ".join(cells) + " |\n")

    if has_mis:
        section_mislabel(out, has_mis)
    return rows


def section_mislabel(out, rows):
    """The false-label arm: a true description, a false articulator label, and no video.

    `modal(true)` / `modal(cued)` are taken from the WITHLABEL arm rather than from the video
    runs, so the comparison is text-arm against text-arm and no modality difference is folded in.
    """
    out.append("\n### Does a text-only score follow a FALSE articulator label?\n")
    out.append("\nThe description handed to the model is unchanged and true of the item; only the "
               "label attached to it is false, and it is false in the same across-the-boundary "
               "direction as the video mis-cue. `modal(true)` and `modal(cued)` are the "
               "`withlabel` arm's modal values for the item's real articulator and for the one "
               "falsely named. If the label is doing the work, the mislabel arm should sit at "
               "`modal(cued)`.\n\n")

    modal = {}
    for _, key in SCORE_KEYS:
        modal[key] = {}
        for channel in CHANNELS:
            vals = [r["withlabel"][key] for r in rows if r["channel"] == channel
                    and isinstance(r["withlabel"].get(key), (int, float))]
            if vals:
                modal[key][channel] = Counter(vals).most_common(1)[0][0]

    out.append("| score | true articulator | n | cued as | withlabel mean | mislabel mean | "
               "shift | modal(true) | modal(cued) | at modal(cued) | at modal(true) |\n"
               + "|---" * 11 + "|\n")
    for label, key in SCORE_KEYS:
        for channel in CHANNELS:
            group = [r for r in rows if r["channel"] == channel
                     and isinstance(r["withlabel"].get(key), (int, float))
                     and isinstance(r["mislabel"].get(key), (int, float))]
            if not group:
                continue
            wm = statistics.mean(r["withlabel"][key] for r in group)
            mm = statistics.mean(r["mislabel"][key] for r in group)
            mt, mc = modal[key].get(channel), modal[key].get(MISCUE_OF[channel])
            out.append(
                f"| {label} | `{channel}` | {len(group)} | `{MISCUE_OF[channel]}` | "
                f"{wm:.2f} | {mm:.2f} | {mm - wm:+.2f} | {mt} | {mc} | "
                f"{sum(1 for r in group if r['mislabel'][key] == mc)}/{len(group)} | "
                f"{sum(1 for r in group if r['mislabel'][key] == mt)}/{len(group)} |\n")
        out.append("|" + " |" * 11 + "\n")

    out.append("\n**Item-level movement under the false label**\n\n")
    out.append("| score | n | unchanged from `withlabel` | moved toward `modal(cued)` | "
               "moved away | mean abs(shift) |\n" + "|---" * 6 + "|\n")
    for label, key in SCORE_KEYS:
        group = [r for r in rows
                 if isinstance(r["withlabel"].get(key), (int, float))
                 and isinstance(r["mislabel"].get(key), (int, float))
                 and modal[key].get(MISCUE_OF[r["channel"]]) is not None]
        if not group:
            continue
        same = toward = away = 0
        shifts = []
        for r in group:
            w, m = r["withlabel"][key], r["mislabel"][key]
            target = modal[key][MISCUE_OF[r["channel"]]]
            shifts.append(abs(m - w))
            if m == w:
                same += 1
            elif abs(m - target) < abs(w - target):
                toward += 1
            else:
                away += 1
        out.append(f"| {label} | {len(group)} | {same}/{len(group)} | {toward}/{len(group)} | "
                   f"{away}/{len(group)} | {statistics.mean(shifts):.2f} |\n")

    out.append("\n**Per-item detail (D1)**\n\n")
    out.append("| item | true | cued as | video | nolabel | withlabel | mislabel |\n"
               + "|---" * 7 + "|\n")
    key = "semantic_syntactic_load"
    for r in sorted(rows, key=lambda r: (r["channel"], r["clip_id"], r["event_id"])):
        out.append(f"| {r['clip_id']}/{r['event_id']} | {r['channel']} | "
                   f"`{MISCUE_OF[r['channel']]}` | {r['video'].get(key)} | "
                   f"{r['nolabel'].get(key)} | {r['withlabel'].get(key)} | "
                   f"{r['mislabel'].get(key)} |\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--mode", default="grounded")
    parser.add_argument("--mediation-subdir", default="mediation",
                        help="read the mediation arms from this subdir; all three arms must come\nfrom a single run of one prompt version, because a prompt edit makes\nnew runs incomparable to old ones")
    parser.add_argument("--skip-miscue", action="store_true")
    parser.add_argument("--skip-mediation", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    main_dir = cfg["paths"]["model_outputs_dir"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    events = load_events(Path(args.manifest) if args.manifest else DEFAULT_MANIFEST)

    out = ["# Mis-cue condition and description->D1 mediation probe\n",
           f"\nModel outputs `{main_dir.name}`, clip kind `combined`, {len(events)} events. "
           "Generated by `scripts/29_analyze_miscue_and_mediation.py`.\n"]
    if not args.skip_miscue:
        section_miscue(out, main_dir, events, args.mode)
    if not args.skip_mediation:
        section_mediation(out, main_dir, events, args.mode, args.mediation_subdir)

    suffix = "" if args.mediation_subdir == "mediation" else f"_{args.mediation_subdir}"
    path = results_dir / f"miscue_and_mediation{suffix}.md"
    path.write_text("".join(out), encoding="utf-8")
    log.info("Wrote %s", path)
    print("".join(out))


if __name__ == "__main__":
    main()
