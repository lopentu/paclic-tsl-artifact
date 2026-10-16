#!/usr/bin/env python3
"""Does the target-channel cue REDIRECT the model to a different articulator, or only RELABEL the
action it had already picked out?

Motivating observation (annotator, 2026-07-30): when the target channel is `non_dominant_hand` and
the non-dominant hand is comparatively still, the model appears to describe the *moving dominant
hand* and simply call it the non-dominant hand -- i.e. it does not first establish handedness from
the clip and then locate the other hand. If so, "channel compliance" (`channel_judged` ==
`target_channel`, 55/55 under CHANNEL_GIVEN modes) overstates what the cue actually achieved: the
model would be echoing the requested label onto an unchanged percept.

This is testable without new annotation, because the same event was run under a mode where the
channel was given and a mode where it was withheld, holding everything else constant:

    grounded            (translation + channel)  vs  ablate_channel (translation only)
    ablate_translation  (channel only)           vs  ablate_both    (neither)

Within each pair the ONLY difference is the channel cue. So compare the two runs' description of
WHICH PHYSICAL ACTION was seen -- the `handshape` / `movement` / `location` fields, which pick out
an action independently of what it is called -- and stratify by the manifest's `target_channel`:

    target_channel == manual             the cue names the articulator the model self-selects
                                         anyway, so high similarity is expected and is the
                                         no-switch-demanded BASELINE.
    target_channel == face               the cue names a channel the model did not self-select, and
                                         one that is not a hand, so low similarity is the POSITIVE
                                         CONTROL that the cue can redirect the description at all.
    target_channel == non_dominant_hand  the cue names the OTHER HAND. If similarity here sits at
                                         the `manual` baseline rather than at the `face` level, the
                                         cue relabelled rather than redirected -- the annotator's
                                         hypothesis.

Two secondary measures, same files:

  * **NDH-field collapse.** Under a `non_dominant_hand` target with the channel given, does the
    `non_dominant_hand` field just restate that run's own `handshape`/`movement` (one action wearing
    two labels), where under CHANNEL_WITHHELD the same field independently reports a resting or
    supporting hand? Reported as token overlap between the NDH field and handshape+movement, plus a
    count of rest/support vocabulary, both stratified by target channel and cue condition.

  * **`hands` field label flip.** What the model says is doing the action (`dominant hand` /
    `non-dominant hand` / `both hands`) in each cue condition, per target channel.

Everything is read straight from the model output JSON plus clip_manifest.csv -- no human rating is
involved, so this analysis is independent of Tier 2/Tier 3 and of the perception gate.

Usage:
    python scripts/analyze_channel_relabeling.py
    python scripts/analyze_channel_relabeling.py --clip-kind combined
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir, REPO_ROOT, parse_manifest_event

log = get_logger("analyze_channel_relabeling")

DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"

# (channel-given mode, channel-withheld mode, what is held constant across the pair)
CUE_PAIRS = [("grounded", "ablate_channel", "translation given in both"),
             ("ablate_translation", "ablate_both", "translation withheld in both")]

# The fields that identify WHICH action was described, as opposed to what it was called.
ACTION_FIELDS = ["handshape", "movement", "location"]

CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]

# Vocabulary for "this hand is not the action" -- a resting, holding, or supporting articulator.
REST_PATTERN = re.compile(
    r"\brest(?:s|ing|ed)?\b|\bnot (?:visible|involved|moving|active)\b|\bstationary\b|"
    r"\bstill\b|\bsteady\b|\bstable base\b|\bnot applicable\b|\bn/?a\b|\bsupport",
    re.IGNORECASE)

STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "then", "with", "is",
             "are", "as", "it", "its", "s", "signer", "signer's", "hand", "hands"}


def tokens(text):
    """Content tokens: lowercased words, stopwords and the word `hand` itself removed.

    `hand`/`hands`/`signer` are dropped on purpose. They are exactly the words that carry the
    LABEL rather than the described action, and leaving them in would let a pure relabelling
    register as a difference in what was seen.
    """
    return {t for t in re.findall(r"[a-z']+", (text or "").lower()) if t not in STOPWORDS}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_run(model_outputs_dir, clip_id, event_id, clip_kind, mode):
    path = model_outputs_dir / f"{clip_id}__{event_id}__{clip_kind}__run1__{mode}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("parsed"):
        return None
    candidates = data["parsed"].get("candidate_actions") or []
    return candidates[0] if candidates else None


def load_manifest_events(manifest_path):
    """[(clip_id, event_id, target_channel)] in manifest order."""
    out = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for index, event_id in ((1, "event1"), (2, "event2")):
                parsed = parse_manifest_event(row, index)
                if parsed is not None:
                    out.append((row["clip_id"], event_id, parsed[2]))
    return out


def fmt_mean(values, fmt="{:.3f}"):
    return fmt.format(statistics.mean(values)) if values else "--"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--clip-kind", default="combined",
                        choices=["window", "ev1", "ev2", "combined"])
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_outputs_dir = cfg["paths"]["model_outputs_dir"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
    events = load_manifest_events(manifest_path)

    lines = ["# Does the channel cue redirect the model, or only relabel what it already saw?\n\n",
             f"Clip kind: `{args.clip_kind}`. Model outputs: `{model_outputs_dir.name}`. "
             f"No human rating is used, so this is independent of Tier 2/3 and of the "
             f"perception gate.\n"]

    # --- 1. Action-description similarity across the channel cue --------------------------
    lines.append("\n## 1. Similarity of the described ACTION with vs without the channel cue\n")
    lines.append("\nToken-Jaccard and character-level ratio over `handshape`+`movement`+`location` "
                 "(the fields that identify which action was seen). The words `hand`/`hands`/"
                 "`signer` are excluded from the token measure so that a pure change of LABEL "
                 "cannot register as a change in what was described.\n")

    per_pair_rows = {}
    for given_mode, withheld_mode, held in CUE_PAIRS:
        by_channel = defaultdict(list)
        detail = []
        for clip_id, event_id, channel in events:
            g = load_run(model_outputs_dir, clip_id, event_id, args.clip_kind, given_mode)
            w = load_run(model_outputs_dir, clip_id, event_id, args.clip_kind, withheld_mode)
            if g is None or w is None:
                continue
            gf, wf = g.get("form_description", {}), w.get("form_description", {})
            g_txt = " ".join(gf.get(k, "") or "" for k in ACTION_FIELDS)
            w_txt = " ".join(wf.get(k, "") or "" for k in ACTION_FIELDS)
            jac = jaccard(tokens(g_txt), tokens(w_txt))
            ratio = SequenceMatcher(None, g_txt.lower(), w_txt.lower()).ratio()
            exact_movement = (gf.get("movement", "") or "").strip().lower() == \
                             (wf.get("movement", "") or "").strip().lower()
            by_channel[channel].append((jac, ratio, exact_movement))
            detail.append((clip_id, event_id, channel, jac, ratio, exact_movement,
                           g.get("channel_judged"), w.get("channel_judged"),
                           gf.get("hands", ""), wf.get("hands", "")))
        per_pair_rows[(given_mode, withheld_mode)] = detail

        lines.append(f"\n### `{given_mode}` vs `{withheld_mode}` ({held})\n\n")
        lines.append("| target_channel | n | mean token-Jaccard | mean char ratio | "
                     "identical `movement` string |\n|---|---|---|---|---|\n")
        for channel in CHANNELS:
            vals = by_channel.get(channel)
            if not vals:
                continue
            lines.append(f"| `{channel}` | {len(vals)} | {fmt_mean([v[0] for v in vals])} | "
                         f"{fmt_mean([v[1] for v in vals])} | "
                         f"{sum(1 for v in vals if v[2])}/{len(vals)} |\n")

    # --- 2. Per-item detail for the non_dominant_hand targets -----------------------------
    lines.append("\n## 2. Item-level detail, `non_dominant_hand` targets\n")
    for (given_mode, withheld_mode), detail in per_pair_rows.items():
        ndh = [d for d in detail if d[2] == "non_dominant_hand"]
        if not ndh:
            continue
        lines.append(f"\n### `{given_mode}` vs `{withheld_mode}`\n\n")
        lines.append(f"| item | Jaccard | identical `movement` | `hands` ({given_mode}) | "
                     f"`hands` ({withheld_mode}) | `channel_judged` given/withheld |\n"
                     "|---|---|---|---|---|---|\n")
        for clip_id, event_id, _, jac, _, exact_mv, g_cj, w_cj, g_hands, w_hands in ndh:
            lines.append(f"| {clip_id}/{event_id} | {jac:.2f} | {'yes' if exact_mv else 'no'} | "
                         f"{g_hands} | {w_hands} | {g_cj} / {w_cj} |\n")

    # --- 2b. The decisive within-run test --------------------------------------------------
    # Section 1's aggregate Jaccard has a confound: under CHANNEL_WITHHELD the model often reports
    # "both hands", so partial overlap with a non-dominant-hand-only description would be expected
    # even from correct behaviour. This removes it by comparing the cued description against TWO
    # texts from the SAME withheld run:
    #   A = that run's target-action fields -> the articulator the model self-selected (dominant /
    #                                          both hands)
    #   B = that run's own non_dominant_hand field -> what it says the NON-dominant hand is doing
    #                                          when the non-dominant hand is not the target
    # If the cued description matches A more than B, then what the cue produced is the
    # self-selected (dominant) percept wearing the requested label -- not a switch to the hand the
    # model itself had identified as non-dominant.
    lines.append("\n## 2b. Decisive test: does the cued description match the withheld run's "
                 "self-selected percept, or its own non-dominant-hand report?\n")
    lines.append("\nFor each event: **X** = the channel-given run's target-action text "
                 "(`handshape`+`movement`+`location`); **A** = the channel-withheld run's "
                 "target-action text (the articulator it self-selected); **B** = the same withheld "
                 "run's own `non_dominant_hand` field. `J(X,A) > J(X,B)` means the cued description "
                 "tracks the self-selected percept rather than the model's own report of the "
                 "non-dominant hand. A and B come from the same run, which removes the "
                 "\"withheld run described both hands\" confound in section 1.\n")
    for given_mode, withheld_mode, held in CUE_PAIRS:
        rows, by_channel = [], defaultdict(list)
        for clip_id, event_id, channel in events:
            g = load_run(model_outputs_dir, clip_id, event_id, args.clip_kind, given_mode)
            w = load_run(model_outputs_dir, clip_id, event_id, args.clip_kind, withheld_mode)
            if g is None or w is None:
                continue
            gf, wf = g.get("form_description", {}), w.get("form_description", {})
            x = tokens(" ".join(gf.get(k, "") or "" for k in ACTION_FIELDS))
            a = tokens(" ".join(wf.get(k, "") or "" for k in ACTION_FIELDS))
            b = tokens(wf.get("non_dominant_hand", "") or "")
            j_a, j_b = jaccard(x, a), jaccard(x, b)
            by_channel[channel].append((j_a, j_b))
            if channel == "non_dominant_hand":
                rows.append((clip_id, event_id, j_a, j_b))
        lines.append(f"\n### `{given_mode}` vs `{withheld_mode}` ({held})\n\n")
        lines.append("| target_channel | n | mean J(X,A) self-selected percept | "
                     "mean J(X,B) own NDH report | J(X,A) > J(X,B) |\n|---|---|---|---|---|\n")
        for channel in CHANNELS:
            vals = by_channel.get(channel)
            if not vals:
                continue
            lines.append(f"| `{channel}` | {len(vals)} | {fmt_mean([v[0] for v in vals])} | "
                         f"{fmt_mean([v[1] for v in vals])} | "
                         f"{sum(1 for v in vals if v[0] > v[1])}/{len(vals)} |\n")
        if rows:
            lines.append("\nPer item, `non_dominant_hand` targets:\n\n")
            lines.append("| item | J(X,A) | J(X,B) |\n|---|---|---|\n")
            for clip_id, event_id, j_a, j_b in rows:
                lines.append(f"| {clip_id}/{event_id} | {j_a:.2f} | {j_b:.2f} |\n")

    # --- 3. The non_dominant_hand FIELD itself --------------------------------------------
    lines.append("\n## 3. The `non_dominant_hand` field: independent report, or restatement of "
                 "the target action?\n")
    lines.append("\nToken-Jaccard between the run's own `non_dominant_hand` field and its "
                 "`handshape`+`movement` (high = the field restates the target action rather than "
                 "reporting a second articulator), and how often that field uses "
                 "resting/supporting vocabulary.\n")
    for mode in ["grounded", "ablate_translation", "ablate_channel", "ablate_both"]:
        by_channel = defaultdict(list)
        for clip_id, event_id, channel in events:
            run = load_run(model_outputs_dir, clip_id, event_id, args.clip_kind, mode)
            if run is None:
                continue
            fd = run.get("form_description", {})
            ndh_txt = fd.get("non_dominant_hand", "") or ""
            action_txt = " ".join(fd.get(k, "") or "" for k in ["handshape", "movement"])
            by_channel[channel].append((jaccard(tokens(ndh_txt), tokens(action_txt)),
                                        bool(REST_PATTERN.search(ndh_txt))))
        lines.append(f"\n### `{mode}`\n\n")
        lines.append("| target_channel | n | NDH-field vs target-action Jaccard | "
                     "NDH field says resting/supporting |\n|---|---|---|---|\n")
        for channel in CHANNELS:
            vals = by_channel.get(channel)
            if not vals:
                continue
            lines.append(f"| `{channel}` | {len(vals)} | {fmt_mean([v[0] for v in vals])} | "
                         f"{sum(1 for v in vals if v[1])}/{len(vals)} |\n")

    # --- 4. What the `hands` field says is acting ----------------------------------------
    lines.append("\n## 4. `hands` field (what the model says is performing the action)\n")
    for mode in ["grounded", "ablate_translation", "ablate_channel", "ablate_both"]:
        counts = defaultdict(lambda: defaultdict(int))
        for clip_id, event_id, channel in events:
            run = load_run(model_outputs_dir, clip_id, event_id, args.clip_kind, mode)
            if run is None:
                continue
            raw = (run.get("form_description", {}).get("hands", "") or "").strip().lower()
            if "non-dominant" in raw or "non dominant" in raw:
                bucket = "non-dominant hand"
            elif "both" in raw or ("dominant" in raw and "non-dominant" in raw):
                bucket = "both hands"
            elif "dominant" in raw or "right hand" in raw or "left hand" in raw:
                bucket = "dominant hand"
            elif "not visible" in raw or raw in {"", "n/a", "not applicable"}:
                bucket = "not visible / N/A"
            else:
                bucket = raw[:28]
            counts[channel][bucket] += 1
        lines.append(f"\n- `{mode}`: " + "; ".join(
            f"**{ch}** -> " + ", ".join(f"{b} {n}" for b, n in sorted(counts[ch].items()))
            for ch in CHANNELS if counts[ch]) + "\n")

    out_path = results_dir / f"channel_relabeling_{args.clip_kind}.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    log.info("Wrote %s", out_path)
    print("".join(lines))


if __name__ == "__main__":
    main()
