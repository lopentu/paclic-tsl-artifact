#!/usr/bin/env python3
"""Read out the three follow-up conditions run on 2026-08-15: resting-articulator mis-cue (26
--scheme resting_ndh), wrong-clip (31), and rubric anchoring (32).

All three exist to close a specific objection to the cue-ablation result, and each is designed so
that the decisive readout is reference-free -- it compares the model against its OWN earlier
output, not against our annotation. "You labelled it wrong" is not an available rebuttal to any of
them.

## 1. Resting-articulator mis-cue

Every target is cued as `non_dominant_hand`, and the strata come from the model's own grounded
`form_description.hands` field. For the ~32 items the model itself described as one-handed or with
no visible non-dominant hand, the cue names an articulator the model has already testified is doing
nothing. The remaining ~23 are internal controls where the cue happens to be true of the video.

  - If the model is LOCATING, a cue pointing at a resting articulator should produce a vacated
    description, a low functional score, or an objection.
  - If the model is NAMING, it should produce a fluent description of an articulator it just said
    was still, and score it like any other non-dominant-hand target.

## 2. Wrong-clip

The whole prompt is kept -- translation, target span, articulator cue -- and only the video is
swapped, for a real clip of a different item with the same articulator. This answers the standing
objection to the degraded ladder: that black/static/shuffled video is out of distribution, so the
model's insensitivity to it says nothing about ordinary inputs. Here the input is a perfectly
ordinary sign clip; it is just the wrong one.

The decisive readout is whether the description resembles the clip the model SAW (partner B's own
grounded description) or the prompt it was GIVEN (item A's). Both comparison texts are the model's
own prior output.

## 3. Rubric anchoring

D1 is given worked illustrations of what its endpoints mean; D2 is deliberately left unanchored in
the same prompt, as a within-prompt control. There is no instruction to use the full range -- that
would measure instruction compliance rather than whether the scale was anchored. So:

  - D1 spreads and D2 does not  -> the compression was an unanchored-scale artefact, specific to
    the dimension that got anchors.
  - both spread                 -> the model is reacting to being given a longer rubric, and
                                   anchoring is not the explanation.
  - neither spreads             -> the compression survives anchoring and is a property of the
                                   judgments, which is what the paper currently claims.

Usage:
    python scripts/33_analyze_followup_conditions.py
    python scripts/33_analyze_followup_conditions.py --skip-wrongclip
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir, REPO_ROOT, parse_manifest_event

log = get_logger("analyze_followup")

DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"
ACTION_FIELDS = ["handshape", "movement", "location"]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]
SCORE_KEYS = [("D1", "semantic_syntactic_load"),
              ("D2", "pragmatic_depictive_affective_enrichment"),
              ("description", "description"), ("indication", "indication"),
              ("depiction", "depiction")]
D1 = "semantic_syntactic_load"
D2 = "pragmatic_depictive_affective_enrichment"

STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "then", "with", "is",
             "are", "as", "it", "its", "s", "signer", "signer's", "hand", "hands"}

# Same two regexes as 29, deliberately: the readouts are meant to be comparable across the two
# mis-cue schemes, and a different vacancy rule would make them incommensurable.
VACANT_RE = re.compile(r"^\s*(not applicable|not visible|none|no visible[^.]*|n/?a|-+)\s*\.?\s*$",
                       re.IGNORECASE)
OBJECTION_RE = re.compile(
    r"\bno (?:visible |significant )?(?:facial|manual|hand)\b|\bnot carried by\b|"
    r"\bdoes not appear to be\b|\bcontrary to\b|\bhowever[, ]|\bactually\b|"
    r"\bappears to be carried by\b|\bmismatch\b|\bincorrect\b", re.IGNORECASE)


# --------------------------------------------------------------------------------------
# shared helpers (duplicated from 29 on purpose: each analyser stays runnable standalone)
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


def record_of(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def field(candidate, name):
    return ((candidate or {}).get("form_description", {}) or {}).get(name, "") or ""


def vacant(candidate, name):
    return bool(VACANT_RE.match(field(candidate, name).strip()))


def action_text(candidate):
    return " ".join(field(candidate, k) for k in ACTION_FIELDS)


def num(candidate, key):
    v = (candidate or {}).get(key)
    return v if isinstance(v, (int, float)) else None


def mean_or_dash(values, fmt="{:.2f}"):
    vals = [v for v in values if v is not None]
    return fmt.format(statistics.mean(vals)) if vals else "--"


def sd_or_dash(values, fmt="{:.2f}"):
    vals = [v for v in values if v is not None]
    return fmt.format(statistics.pstdev(vals)) if len(vals) > 1 else "--"


def unusable_note(paths):
    """Name WHY a run was dropped, not just that it was. The stored `schema_errors` already say
    it; surfacing them keeps a reader from reading ordinary attrition as a capability failure."""
    out = []
    for path in paths:
        rec = record_of(path) or {}
        if ((rec.get("parsed") or {}).get("candidate_actions")):
            continue
        errs = rec.get("schema_errors") or ["no candidate_actions"]
        out.append(f"`{path.name}` ({errs[0]})")
    return out


# --------------------------------------------------------------------------------------
# 1. resting-articulator mis-cue
# --------------------------------------------------------------------------------------

def ndh_status(hands_text):
    """Classify the model's own `hands` testimony into whether the NON-DOMINANT hand is acting.

    Order matters: 'non-dominant hand' contains 'dominant hand' as a substring, so the
    non-dominant test has to come first or every one-handed item would be misread as two-handed.
    """
    t = (hands_text or "").strip().lower()
    if not t:
        return "unknown"
    if "both" in t:
        return "active"
    if "non-dominant" in t or "non dominant" in t or "nondominant" in t:
        return "active"
    if "not visible" in t or "not actively" in t or "no visible" in t:
        return "resting"
    if "dominant" in t or "one hand" in t or "one-handed" in t:
        return "resting"
    return "unknown"


def section_resting_miscue(out, main_dir, events, mode="grounded"):
    rest_dir = main_dir / "miscue_resting"
    rows = []
    for clip_id, event_id, channel, video_id in events:
        m, m_raw = candidate_of(rest_dir / f"{clip_id}__{event_id}__combined__run1__"
                                           f"miscue_{mode}.json")
        g, _ = candidate_of(main_dir / f"{clip_id}__{event_id}__combined__run1__{mode}.json")
        if m is None or g is None:
            continue
        rows.append({"clip_id": clip_id, "event_id": event_id, "true": channel,
                     "video_id": video_id, "m": m, "g": g, "raw": m_raw,
                     "hands": field(g, "hands"),
                     "stratum": ndh_status(field(g, "hands")),
                     "g_ndh_vacant": vacant(g, "non_dominant_hand")})
    if not rows:
        out.append("\n## 1. Resting-articulator mis-cue\n\n_(no outputs in `miscue_resting/`)_\n")
        return rows

    strata = ["resting", "active", "unknown"]
    out.append(f"\n## 1. Resting-articulator mis-cue ({len(rows)} events, cue cell `{mode}`)\n")
    out.append("\nEvery target cued as `non_dominant_hand`, whatever it actually is. Strata are "
               "the model's own grounded `form_description.hands` field, so the classification is "
               "the model's testimony about the video, not ours: **resting** = it reported one "
               "hand, or no visible non-dominant hand; **active** = it reported both hands or the "
               "non-dominant hand. On the resting stratum the cue names an articulator the model "
               "has already said is doing nothing; the active stratum is an internal control where "
               "the cue happens to be true.\n")

    out.append("\n### 1.0 Strata, from the model's own grounded description\n\n")
    out.append("| model's own `hands` value | n | stratum | grounded `non_dominant_hand` field "
               "vacated |\n" + "|---" * 4 + "|\n")
    for hands_val, n in Counter(r["hands"] for r in rows).most_common():
        group = [r for r in rows if r["hands"] == hands_val]
        vac = sum(1 for r in group if r["g_ndh_vacant"])
        out.append(f"| `{hands_val or '(empty)'}` | {n} | {ndh_status(hands_val)} | "
                   f"{vac}/{n} |\n")
    counts = Counter(r["stratum"] for r in rows)
    out.append(f"\nTotals: resting {counts['resting']}, active {counts['active']}, "
               f"unknown {counts['unknown']}.\n")

    # --- 1.1 label compliance ----------------------------------------------------------
    out.append("\n### 1.1 Does `channel_judged` follow the cue onto a resting articulator?\n")
    out.append("\nThe buckets are mutually exclusive. Items whose real articulator IS "
               "`non_dominant_hand` are counted only as echoing the cue, never also as resisting "
               "it -- for them the cue is true, so the two would be the same event; they are "
               "reported separately in the last column.\n\n")
    out.append("| stratum | n | echoes the cue (`non_dominant_hand`) | names the item's true "
               "articulator instead | other | of which cue was already correct |\n"
               + "|---" * 6 + "|\n")
    for st in strata:
        group = [r for r in rows if r["stratum"] == st]
        if not group:
            continue
        follows = resists = other = 0
        for r in group:
            judged = r["m"].get("channel_judged")
            if judged == "non_dominant_hand":
                follows += 1
            elif judged == r["true"]:
                resists += 1
            else:
                other += 1
        correct_cue = sum(1 for r in group if r["true"] == "non_dominant_hand")
        out.append(f"| {st} | {len(group)} | {follows}/{len(group)} | {resists}/{len(group)} | "
                   f"{other}/{len(group)} | {correct_cue} |\n")

    # --- 1.2 the sharpest readout: does it now describe what it said was still? ---------
    out.append("\n### 1.2 Does the model now describe activity it had already denied?\n")
    out.append("\nRestricted to items whose grounded run **vacated** its own `non_dominant_hand` "
               "field (`not applicable` / `not visible` / `none`). If the mis-cued run fills that "
               "same field with substance, the model has produced a description of an articulator "
               "it just testified was not participating.\n\n")
    denied = [r for r in rows if r["g_ndh_vacant"]]
    if not denied:
        out.append("_(no item vacated its own `non_dominant_hand` field in the grounded run)_\n")
    else:
        filled = [r for r in denied if not vacant(r["m"], "non_dominant_hand")
                  and field(r["m"], "non_dominant_hand").strip() != ""]
        out.append(f"| items whose grounded run vacated `non_dominant_hand` | "
                   f"mis-cued run fills it | mis-cued run also vacates it |\n"
                   + "|---" * 3 + "|\n")
        out.append(f"| {len(denied)} | {len(filled)}/{len(denied)} | "
                   f"{len(denied) - len(filled)}/{len(denied)} |\n")
        if filled:
            out.append("\nExamples of a field that was empty under the true cue and substantive "
                       "under the resting cue:\n\n")
            for r in filled[:6]:
                out.append(f"- **{r['clip_id']}/{r['event_id']}** (`hands` = "
                           f"`{r['hands']}`) -> "
                           f"`{field(r['m'], 'non_dominant_hand').strip()[:160]}`\n")

    # --- 1.3 do the hand-action fields survive the cue? --------------------------------
    out.append("\n### 1.3 Where does the description go?\n\n")
    out.append("| stratum | n | action fields vacated (mis-cue) | (true cue) | "
               "`non_dominant_hand` substantive (mis-cue) | (true cue) | J(M,G) on action "
               "fields |\n" + "|---" * 7 + "|\n")
    for st in strata:
        group = [r for r in rows if r["stratum"] == st]
        if not group:
            continue
        def frac(key, fn):
            return f"{sum(1 for r in group if fn(r[key]))}/{len(group)}"
        js = [jaccard(tokens(action_text(r["m"])), tokens(action_text(r["g"]))) for r in group]
        subst = lambda c: (not vacant(c, "non_dominant_hand")
                           and field(c, "non_dominant_hand").strip() != "")
        out.append(f"| {st} | {len(group)} | "
                   f"{frac('m', lambda c: vacant(c, 'handshape'))} | "
                   f"{frac('g', lambda c: vacant(c, 'handshape'))} | "
                   f"{frac('m', subst)} | {frac('g', subst)} | "
                   f"{statistics.mean(js):.3f} |\n")

    # --- 1.4 functional scores ---------------------------------------------------------
    # The label-driven prediction: an item cued as non_dominant_hand should be scored like the
    # true non_dominant_hand items are, regardless of whether anything is moving.
    ndh_modal = {}
    for _, key in SCORE_KEYS:
        vals = [num(r["g"], key) for r in rows if r["true"] == "non_dominant_hand"]
        vals = [v for v in vals if v is not None]
        ndh_modal[key] = Counter(vals).most_common(1)[0][0] if vals else None

    out.append("\n### 1.4 Do the functional scores follow the cue onto a resting articulator?\n")
    out.append("\n`modal(NDH)` is the modal value the true-cue runs give to real "
               "`non_dominant_hand` targets. If the score is driven by the label, the resting "
               "stratum should sit there too.\n\n")
    out.append("| score | stratum | n | mean (true cue) | mean (resting cue) | shift | "
               "modal(NDH) | at modal(NDH) |\n" + "|---" * 8 + "|\n")
    for label, key in SCORE_KEYS:
        for st in strata:
            group = [r for r in rows if r["stratum"] == st
                     and num(r["m"], key) is not None and num(r["g"], key) is not None]
            if not group:
                continue
            gm = statistics.mean(num(r["g"], key) for r in group)
            mm = statistics.mean(num(r["m"], key) for r in group)
            at = sum(1 for r in group if num(r["m"], key) == ndh_modal[key])
            out.append(f"| {label} | {st} | {len(group)} | {gm:.2f} | {mm:.2f} | "
                       f"{mm - gm:+.2f} | {ndh_modal[key]} | {at}/{len(group)} |\n")
        out.append("|" + " |" * 8 + "\n")

    # --- 1.5 objections and confidence -------------------------------------------------
    out.append("\n### 1.5 Does the model object, and how confident is it?\n\n")
    out.append("| stratum | n | mean confidence (resting cue) | (true cue) | "
               "output flags a mismatch |\n" + "|---" * 5 + "|\n")
    for st in strata:
        group = [r for r in rows if r["stratum"] == st]
        if not group:
            continue
        obj = sum(1 for r in group
                  if OBJECTION_RE.search(r["m"].get("observable_evidence", "") or ""))
        out.append(f"| {st} | {len(group)} | "
                   f"{mean_or_dash([num(r['m'], 'confidence') for r in group])} | "
                   f"{mean_or_dash([num(r['g'], 'confidence') for r in group])} | "
                   f"{obj}/{len(group)} |\n")

    out.append("\n### Per-item detail\n\n")
    out.append("| item | true | model's `hands` | stratum | judged | D1 true | D1 cued | "
               "NDH field filled |\n" + "|---" * 8 + "|\n")
    for r in sorted(rows, key=lambda r: (r["stratum"], r["clip_id"], r["event_id"])):
        subst = (not vacant(r["m"], "non_dominant_hand")
                 and field(r["m"], "non_dominant_hand").strip() != "")
        out.append(f"| {r['clip_id']}/{r['event_id']} | {r['true']} | `{r['hands']}` | "
                   f"{r['stratum']} | {r['m'].get('channel_judged')} | "
                   f"{num(r['g'], D1)} | {num(r['m'], D1)} | "
                   f"{'yes' if subst else 'no'} |\n")
    return rows


# --------------------------------------------------------------------------------------
# 2. wrong-clip
# --------------------------------------------------------------------------------------

def section_wrongclip(out, main_dir, mode="grounded"):
    wc_dir = main_dir / "wrongclip"
    paths = sorted(p for p in wc_dir.glob(f"*__wrongclip_{mode}.json")
                   if "manifest" not in p.name) if wc_dir.exists() else []
    rows = []
    for path in paths:
        rec = record_of(path)
        if rec is None:
            continue
        cands = (rec.get("parsed") or {}).get("candidate_actions") or []
        wc = cands[0] if cands else None
        pf, vf = rec.get("prompt_from") or {}, rec.get("video_from") or {}
        a, _ = candidate_of(main_dir / f"{pf.get('clip_id')}__{pf.get('event_id')}__combined__"
                                       f"run1__{mode}.json")
        b, _ = candidate_of(main_dir / f"{vf.get('clip_id')}__{vf.get('event_id')}__combined__"
                                       f"run1__{mode}.json")
        if wc is None or a is None or b is None:
            continue
        rows.append({"wc": wc, "a": a, "b": b, "rec": rec,
                     "a_id": f"{pf.get('clip_id')}/{pf.get('event_id')}",
                     "b_id": f"{vf.get('clip_id')}/{vf.get('event_id')}",
                     "channel": rec.get("target_channel"),
                     "clamped": bool(rec.get("span_clamped"))})
    if not rows:
        out.append("\n## 2. Wrong-clip condition\n\n_(no outputs in `wrongclip/`)_\n")
        return rows

    out.append(f"\n## 2. Wrong-clip condition ({len(rows)} events, cue cell `{mode}`)\n")
    out.append("\nPrompt from item **A** -- translation, target span, articulator cue, all of it "
               "-- paired with the video of item **B**, a different item with the same "
               "articulator. Unlike the degraded ladder, the input here is an ordinary, "
               "in-distribution sign clip; it is simply not the one the prompt is about. Both "
               "comparison descriptions (A's and B's) are the model's own true-cue output for "
               "those items, so nothing here depends on our annotation.\n")
    unusable = unusable_note(paths)
    if unusable or len(rows) < len(paths):
        out.append(f"\n{len(paths)} runs completed; **{len(rows)} usable**"
                   + (f". Dropped for unparseable output: {', '.join(unusable)}" if unusable
                      else " (the rest lack a comparison run)") + ".\n")
    clamped_n = sum(1 for r in rows if r["clamped"])
    out.append(f"\n{clamped_n}/{len(rows)} runs had A's requested span clamped to fit B's "
               "duration; §2.5 splits on that.\n")

    # --- 2.1 the decisive readout ------------------------------------------------------
    out.append("\n### 2.1 Does the description follow the video seen, or the prompt given?\n")
    out.append("\n`WC` the wrong-clip description, `A` the description the model gave for the "
               "item the PROMPT is about, `B` the one it gave for the clip it actually SAW. A "
               "model that is looking should score higher against B; a model generating from the "
               "translation should score higher against A.\n\n")
    out.append("| articulator | n | J(WC,B) *saw* | J(WC,A) *was told* | J(WC,B) - J(WC,A) | "
               "C(WC,B) | C(WC,A) | closer to B |\n" + "|---" * 8 + "|\n")
    for channel in CHANNELS + ["ALL"]:
        group = [r for r in rows if channel == "ALL" or r["channel"] == channel]
        if not group:
            continue
        jb, ja, cb, ca, closer = [], [], [], [], 0
        for r in group:
            tw = tokens(action_text(r["wc"]))
            tb, ta = tokens(action_text(r["b"])), tokens(action_text(r["a"]))
            jb.append(jaccard(tw, tb)); ja.append(jaccard(tw, ta))
            cb.append(containment(tw, tb)); ca.append(containment(tw, ta))
            if jb[-1] > ja[-1]:
                closer += 1
        out.append(f"| `{channel}` | {len(group)} | {statistics.mean(jb):.3f} | "
                   f"{statistics.mean(ja):.3f} | "
                   f"{statistics.mean(jb) - statistics.mean(ja):+.3f} | "
                   f"{statistics.mean(cb):.3f} | {statistics.mean(ca):.3f} | "
                   f"{closer}/{len(group)} |\n")

    # --- 2.2 scores --------------------------------------------------------------------
    out.append("\n### 2.2 Do the functional scores follow the video seen or the prompt given?\n\n")
    out.append("| score | n | mean WC | mean A *told* | mean B *saw* | mean abs(WC-A) | "
               "mean abs(WC-B) | WC = A | WC = B |\n" + "|---" * 9 + "|\n")
    for label, key in SCORE_KEYS:
        group = [r for r in rows if num(r["wc"], key) is not None
                 and num(r["a"], key) is not None and num(r["b"], key) is not None]
        if not group:
            continue
        da = [abs(num(r["wc"], key) - num(r["a"], key)) for r in group]
        db = [abs(num(r["wc"], key) - num(r["b"], key)) for r in group]
        out.append(f"| {label} | {len(group)} | "
                   f"{statistics.mean(num(r['wc'], key) for r in group):.2f} | "
                   f"{statistics.mean(num(r['a'], key) for r in group):.2f} | "
                   f"{statistics.mean(num(r['b'], key) for r in group):.2f} | "
                   f"{statistics.mean(da):.2f} | {statistics.mean(db):.2f} | "
                   f"{sum(1 for r in group if num(r['wc'], key) == num(r['a'], key))}"
                   f"/{len(group)} | "
                   f"{sum(1 for r in group if num(r['wc'], key) == num(r['b'], key))}"
                   f"/{len(group)} |\n")

    # --- 2.3 does it notice --------------------------------------------------------------
    out.append("\n### 2.3 Does the model notice the clip does not match the prompt?\n\n")
    out.append("| articulator | n | mean confidence (wrong clip) | (A's own true run) | "
               "flags a mismatch | `caption_leakage_risk` raised |\n" + "|---" * 6 + "|\n")
    for channel in CHANNELS + ["ALL"]:
        group = [r for r in rows if channel == "ALL" or r["channel"] == channel]
        if not group:
            continue
        obj = sum(1 for r in group
                  if OBJECTION_RE.search(r["wc"].get("observable_evidence", "") or ""))
        leak = sum(1 for r in group
                   if str(((r["rec"].get("parsed") or {}).get("caption_leakage_risk")
                           or "")).strip().lower() in {"high", "medium", "true", "yes"})
        out.append(f"| `{channel}` | {len(group)} | "
                   f"{mean_or_dash([num(r['wc'], 'confidence') for r in group])} | "
                   f"{mean_or_dash([num(r['a'], 'confidence') for r in group])} | "
                   f"{obj}/{len(group)} | {leak}/{len(group)} |\n")

    # --- 2.4 channel compliance (null check) ---------------------------------------------
    follows = sum(1 for r in rows if r["wc"].get("channel_judged") == r["channel"])
    out.append(f"\n### 2.4 Cue compliance (null check)\n\n"
               f"A and B share an articulator by construction, so the cue is true of the video "
               f"shown and `channel_judged` should be unaffected: {follows}/{len(rows)} echo the "
               f"cue. A drop here would mean the swap disturbed something other than the content.\n")

    # --- 2.5 clamping control -------------------------------------------------------------
    out.append("\n### 2.5 Span-clamping control\n")
    out.append("\nWhere A's requested span did not fit B's duration it was clamped. If the "
               "headline effect were an artefact of clamping, the two rows would differ.\n\n")
    out.append("| span clamped | n | J(WC,B) | J(WC,A) | closer to B |\n" + "|---" * 5 + "|\n")
    for flag in (False, True):
        group = [r for r in rows if r["clamped"] == flag]
        if not group:
            continue
        jb = [jaccard(tokens(action_text(r["wc"])), tokens(action_text(r["b"]))) for r in group]
        ja = [jaccard(tokens(action_text(r["wc"])), tokens(action_text(r["a"]))) for r in group]
        closer = sum(1 for x, y in zip(jb, ja) if x > y)
        out.append(f"| {'yes' if flag else 'no'} | {len(group)} | {statistics.mean(jb):.3f} | "
                   f"{statistics.mean(ja):.3f} | {closer}/{len(group)} |\n")

    out.append("\n### Per-item detail\n\n")
    out.append("| prompt from A | video from B | articulator | J(WC,B) | J(WC,A) | D1 A | D1 B | "
               "D1 WC | flags |\n" + "|---" * 9 + "|\n")
    for r in rows:
        tw = tokens(action_text(r["wc"]))
        out.append(f"| {r['a_id']} | {r['b_id']} | {r['channel']} | "
                   f"{jaccard(tw, tokens(action_text(r['b']))):.2f} | "
                   f"{jaccard(tw, tokens(action_text(r['a']))):.2f} | "
                   f"{num(r['a'], D1)} | {num(r['b'], D1)} | {num(r['wc'], D1)} | "
                   f"{'clamped ' if r['clamped'] else ''}"
                   f"{'objects' if OBJECTION_RE.search(r['wc'].get('observable_evidence','') or '') else ''} |\n")
    return rows


# --------------------------------------------------------------------------------------
# 3. rubric anchoring
# --------------------------------------------------------------------------------------

def dist_row(name, values, scale_lo=0, scale_hi=3):
    """All five scores in Prompt C are on the SAME 0-3 scale (see prompt_c step 3-5), so the
    endpoints are 0 and 3, not 1 and 5. Counting `at an endpoint` against the wrong bounds would
    silently understate exactly the compression the anchoring condition is testing."""
    vals = [v for v in values if v is not None]
    if not vals:
        return f"| {name} | 0 | -- | -- | -- | -- | -- | -- |\n"
    at_lo = sum(1 for v in vals if v <= scale_lo)
    at_hi = sum(1 for v in vals if v >= scale_hi)
    return (f"| {name} | {len(vals)} | {statistics.mean(vals):.2f} | "
            f"{statistics.pstdev(vals):.2f} | {min(vals)}-{max(vals)} | "
            f"{len(set(vals))} | {entropy(vals):.2f} | "
            f"{at_lo + at_hi}/{len(vals)} |\n")


def section_anchored(out, main_dir, events, mode="grounded"):
    anc_dir = main_dir / "anchored"
    rows = []
    for clip_id, event_id, channel, video_id in events:
        a, _ = candidate_of(anc_dir / f"{clip_id}__{event_id}__combined__run1__anchored_{mode}.json")
        g, _ = candidate_of(main_dir / f"{clip_id}__{event_id}__combined__run1__{mode}.json")
        if a is None or g is None:
            continue
        rows.append({"clip_id": clip_id, "event_id": event_id, "channel": channel,
                     "a": a, "g": g})
    present = sorted(anc_dir.glob(f"*__anchored_{mode}.json")) if anc_dir.exists() else []
    unusable = unusable_note(present)
    if not rows:
        out.append("\n## 3. Rubric anchoring\n\n_(no outputs in `anchored/`)_\n")
        return rows

    out.append(f"\n## 3. Rubric anchoring ({len(rows)} events, cue cell `{mode}`)\n")
    if unusable:
        out.append(f"\n{len(present)} runs completed; **{len(rows)} usable**. Dropped for "
                   f"unparseable output: {', '.join(unusable)}.\n")
    out.append("\n**D1 is anchored; D2 and the three sub-scores are not.** They travel in the "
               "same prompt, so they are a within-prompt control for the model simply reacting to "
               "a longer rubric. The prompt contains no instruction to use the full range: that "
               "would measure compliance, not calibration.\n")

    # --- 3.1 distributions ---------------------------------------------------------
    out.append("\n### 3.1 Does the anchored dimension spread, and do the unanchored ones "
               "stay put?\n\n")
    out.append("| dimension | condition | n | mean | SD | range | distinct values | entropy "
               "(bits) | at an endpoint |\n" + "|---" * 9 + "|\n")
    for label, key in SCORE_KEYS:
        tag = "**anchored**" if key == D1 else "control"
        for cond, src in (("true cue", "g"), ("anchored prompt", "a")):
            row = dist_row(f"{label} ({tag}) | {cond}",
                           [num(r[src], key) for r in rows])
            out.append("| " + row[2:])
        out.append("|" + " |" * 9 + "\n")

    # --- 3.2 does the articulator effect survive ------------------------------------------
    out.append("\n### 3.2 Does the articulator effect on D1 survive anchoring?\n")
    out.append("\nThe paper's central quantity is the D1 gap between articulators. If that gap "
               "were an artefact of an uncalibrated scale, giving the scale worked endpoints "
               "should shrink it.\n\n")
    out.append("| articulator | n | D1 mean (true cue) | D1 mean (anchored) | shift | "
               "D1 SD (true cue) | D1 SD (anchored) |\n" + "|---" * 7 + "|\n")
    means, sizes = {}, {}
    for channel in CHANNELS:
        group = [r for r in rows if r["channel"] == channel]
        gv = [num(r["g"], D1) for r in group]
        av = [num(r["a"], D1) for r in group]
        gv, av = [v for v in gv if v is not None], [v for v in av if v is not None]
        if not gv or not av:
            continue
        means[channel] = (statistics.mean(gv), statistics.mean(av))
        sizes[channel] = len(group)
        out.append(f"| `{channel}` | {len(group)} | {statistics.mean(gv):.2f} | "
                   f"{statistics.mean(av):.2f} | "
                   f"{statistics.mean(av) - statistics.mean(gv):+.2f} | "
                   f"{sd_or_dash(gv)} | {sd_or_dash(av)} |\n")
    # A max-min over per-articulator means is dominated by whichever cell is smallest, and
    # head_torso has n=1: one item moving one scale point would read as the gap doubling. So the
    # headline number is computed over cells with n>1, with the all-cells figure kept alongside
    # rather than dropped.
    if len(means) > 1:
        def spread(idx, src):
            vals = [v[idx] for k, v in src.items()]
            return max(vals) - min(vals)
        big = {k: v for k, v in means.items() if sizes.get(k, 0) > 1}
        small = [f"`{k}` (n={sizes[k]})" for k in means if sizes.get(k, 0) <= 1]
        if len(big) > 1:
            out.append(f"\nMax-min D1 gap across articulators, **cells with n>1 only**: "
                       f"**{spread(0, big):.2f}** under the true cue, **{spread(1, big):.2f}** "
                       f"with D1 anchored ({spread(1, big) - spread(0, big):+.2f}).\n")
        if small:
            out.append(f"\nExcluded from that figure: {', '.join(small)} -- a single item cannot "
                       f"support a group mean. Over all cells including them the gap reads "
                       f"{spread(0, means):.2f} -> {spread(1, means):.2f}, which is an artefact "
                       f"of the singleton cell and should not be quoted.\n")

    # --- 3.3 item-level movement ----------------------------------------------------------
    out.append("\n### 3.3 Item-level movement\n\n")
    out.append("| dimension | n | unchanged | moved up | moved down | mean signed shift | "
               "mean abs(shift) |\n" + "|---" * 7 + "|\n")
    for label, key in SCORE_KEYS:
        group = [r for r in rows
                 if num(r["a"], key) is not None and num(r["g"], key) is not None]
        if not group:
            continue
        deltas = [num(r["a"], key) - num(r["g"], key) for r in group]
        tag = " (anchored)" if key == D1 else ""
        out.append(f"| {label}{tag} | {len(group)} | "
                   f"{sum(1 for d in deltas if d == 0)}/{len(group)} | "
                   f"{sum(1 for d in deltas if d > 0)}/{len(group)} | "
                   f"{sum(1 for d in deltas if d < 0)}/{len(group)} | "
                   f"{statistics.mean(deltas):+.2f} | "
                   f"{statistics.mean(abs(d) for d in deltas):.2f} |\n")

    # --- 3.4 histogram --------------------------------------------------------------------
    out.append("\n### 3.4 D1 histogram\n\n")
    all_vals = sorted({v for r in rows for v in (num(r["g"], D1), num(r["a"], D1))
                       if v is not None})
    out.append("| condition | " + " | ".join(str(v) for v in all_vals) + " |\n"
               + "|---" * (len(all_vals) + 1) + "|\n")
    for cond, src in (("true cue", "g"), ("anchored", "a")):
        counts = Counter(num(r[src], D1) for r in rows if num(r[src], D1) is not None)
        out.append(f"| {cond} | " + " | ".join(str(counts.get(v, 0)) for v in all_vals) + " |\n")

    out.append("\n### Per-item detail\n\n")
    out.append("| item | articulator | D1 true | D1 anchored | D1 shift | D2 true | "
               "D2 anchored | D2 shift |\n" + "|---" * 8 + "|\n")
    for r in rows:
        def shift(key):
            x, y = num(r["g"], key), num(r["a"], key)
            return f"{y - x:+d}" if isinstance(x, int) and isinstance(y, int) else "--"
        out.append(f"| {r['clip_id']}/{r['event_id']} | {r['channel']} | "
                   f"{num(r['g'], D1)} | {num(r['a'], D1)} | {shift(D1)} | "
                   f"{num(r['g'], D2)} | {num(r['a'], D2)} | {shift(D2)} |\n")
    return rows


# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--mode", default="grounded")
    parser.add_argument("--skip-resting", action="store_true")
    parser.add_argument("--skip-wrongclip", action="store_true")
    parser.add_argument("--skip-anchored", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    main_dir = cfg["paths"]["model_outputs_dir"]
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    events = load_events(Path(args.manifest) if args.manifest else DEFAULT_MANIFEST)

    out = ["# Follow-up conditions: resting-articulator mis-cue, wrong-clip, rubric anchoring\n",
           f"\nModel outputs `{main_dir.name}`, clip kind `combined`, {len(events)} events, cue "
           f"cell `{args.mode}`. Generated by `scripts/33_analyze_followup_conditions.py`.\n",
           "\nEach section compares the model against its own earlier output rather than against "
           "our annotation, so none of the three readouts can be answered with \"your reference "
           "labels are wrong\".\n"]
    if not args.skip_resting:
        section_resting_miscue(out, main_dir, events, args.mode)
    if not args.skip_wrongclip:
        section_wrongclip(out, main_dir, args.mode)
    if not args.skip_anchored:
        section_anchored(out, main_dir, events, args.mode)

    path = results_dir / "followup_conditions.md"
    path.write_text("".join(out), encoding="utf-8")
    log.info("Wrote %s", path)
    print("".join(out))


if __name__ == "__main__":
    main()
