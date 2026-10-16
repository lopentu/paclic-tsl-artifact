#!/usr/bin/env python3
"""Description -> D1 mediation probe: is the functional score a function of the label, or of the
model's own description of what it saw?

The no-video condition shows D1 surviving the loss of the video on 46/55 items. It cannot say
WHY, because that prompt still carries the articulator label: a D1 that survives could be
following the label, or following whatever the model would have described anyway. The two paths
are dissociated here by handing the model back its OWN form description, as text, with no video,
in two arms that differ in one sentence:

    nolabel     the description alone
    withlabel   the description plus "The action is carried by <articulator>."
    mislabel    the description plus the same sentence naming the WRONG articulator

    D1(nolabel)  == D1(video run)                        ->  score is a function of the DESCRIPTION
    D1(nolabel)  != D1(video run), D1(withlabel) == it   ->  score is a function of the LABEL

`mislabel` (added 2026-08-15) is what makes the three arms a test rather than a comparison: with
`withlabel` alone the label and the description agree, so a score matching both is unattributable.
Putting them in conflict -- true description, false label, no video -- is the one place in the
study where naming and seeing point at different answers with nothing visual left to confound it.

Descriptions are taken verbatim from each event's `grounded` run -- the same run whose D1 is the
comparison target -- rendered as `field: value` lines with empty and `not applicable` /
`not visible` fields dropped, since those carry no content and their presence differs
systematically between face and manual targets.

Text-only, so this costs a fraction of a video run. Outputs land in their own namespace
(`--out-subdir`, default `mediation`) and can never pool with the main 220.

Usage:
    python scripts/27_run_mediation_probe.py --dry-run
    python scripts/27_run_mediation_probe.py
    python scripts/27_run_mediation_probe.py --arm nolabel
"""

import argparse
import csv
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir, REPO_ROOT, parse_manifest_event
from lib.prompt_utils import render_blocks, CHANNEL_PHRASES
from lib.schema_utils import extract_json_object, validate_output

log = get_logger("run_mediation")

ARMS = ["nolabel", "withlabel", "mislabel"]
LABEL_ARMS = {"withlabel", "mislabel"}
PROMPT_FILE = "prompts/prompt_m_mediation.md"
DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"
SOURCE_MODE = "grounded"

# Third arm (2026-08-15). `withlabel` adds the TRUE articulator; `mislabel` adds a FALSE one, using
# the same map as 26_run_miscue_condition.py's cross_boundary scheme (asserted below, not trusted).
#
# This is the sharpest form of the paper's central claim available anywhere in the study, because
# it is the only place the two candidate paths are separated with no video in the room at all. The
# model is handed its OWN description of what it saw -- verbatim, true, and the very text whose D1
# is the comparison target -- and then told the action was carried by an articulator that
# description does not mention:
#
#     D1(mislabel) == D1(nolabel)     the score follows the DESCRIPTION; the label is inert once
#                                     the content is fixed, and the video runs' label effect must
#                                     be operating on perception rather than on scoring
#     D1(mislabel) tracks the LABEL   the score follows the name even when the description in front
#                                     of it says otherwise -- naming, with nothing left to see
#
# The `hands` field is already excluded from the rendered description (see FORM_FIELDS), so the
# true articulator is not smuggled back in as content for the false label to contradict.
MISCUE_OF = {"manual": "face", "non_dominant_hand": "face", "head_torso": "face",
             "face": "manual"}


def assert_miscue_map_matches_video_condition(main_dir):
    """The false labels here must be the ones 26 actually used, or the arms are not comparable.

    26 records its map in the run manifest, so this is checked against what ran rather than
    against a copy of a constant that could have drifted since.
    """
    mpath = main_dir / "miscue" / "run_manifest_miscue.json"
    if not mpath.exists():
        log.warning("no %s: cannot verify the mislabel map against the video mis-cue run", mpath)
        return
    history = json.loads(mpath.read_text(encoding="utf-8"))
    used = [e["miscue_map"] for e in history
            if e.get("scheme", "cross_boundary") == "cross_boundary" and "miscue_map" in e]
    if used and used[-1] != MISCUE_OF:
        raise SystemExit(f"mislabel map {MISCUE_OF} != the map 26 ran with {used[-1]}; the arms "
                         f"would not be comparable. Reconcile before running.")

# Ordered so the rendered description reads like the schema's own field order. `hands` is
# EXCLUDED from both arms: its value is "dominant hand" / "non-dominant hand" / "not visible",
# i.e. it states the articulator rather than describing the action, so leaving it in would put
# the label back into the arm that is supposed to be missing it. This is the same exclusion the
# paper's description-similarity measure already makes (drop `hand(s)`/`signer` so a label change
# cannot register as a content change) -- applied here for the same reason.
FORM_FIELDS = ["handshape", "movement", "location", "nonmanual", "non_dominant_hand",
               "temporal_alignment"]
EMPTY_RE = re.compile(r"^\s*(not applicable|not visible|none|n/?a|-+)\s*\.?\s*$", re.IGNORECASE)

SCORE_KEYS = ["semantic_syntactic_load", "pragmatic_depictive_affective_enrichment",
              "description", "indication", "depiction"]
SCHEMA = {
    "type": "object",
    "required": SCORE_KEYS,
    "properties": {**{k: {"type": "integer", "minimum": 0, "maximum": 3} for k in SCORE_KEYS},
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                   "reasoning": {"type": "string"}},
}


def load_mediation_prompt(path):
    """Body between the header rule and the researcher-facing NOTE ON USE section."""
    text = Path(path).read_text(encoding="utf-8")
    body = text.split("\n---\n", 1)[1]
    body = body.split("## NOTE ON USE", 1)[0]
    found = {m for m in re.findall(r"\[\[([A-Z_]+)\]\]", body)}
    expected = {"MEANING_GIVEN", "LABEL_GIVEN"}
    if found != expected:
        raise ValueError(f"{path}: expected blocks {sorted(expected)}, found {sorted(found)}")
    return body.strip() + "\n"


def render_description(form_description):
    """`field: value` lines, dropping empties and not-applicable placeholders."""
    lines = []
    for field in FORM_FIELDS:
        value = (form_description.get(field) or "").strip()
        if not value or EMPTY_RE.match(value):
            continue
        lines.append(f"- {field.replace('_', ' ')}: {value}")
    return "\n".join(lines)


def label_channel_for(arm, true_channel):
    """Which articulator this arm states, or None if it states none."""
    if arm == "withlabel":
        return true_channel
    if arm == "mislabel":
        return MISCUE_OF[true_channel]
    return None


def fill_prompt(prompt_text, arm, translation, channel_code, description):
    keep = {"MEANING_GIVEN"} | ({"LABEL_GIVEN"} if arm in LABEL_ARMS else set())
    text = render_blocks(prompt_text, keep)
    if not translation.strip():
        raise ValueError("the manifest's translation column is empty (see README)")
    text = text.replace("__UTTERANCE_MEANING__", translation)
    if arm in LABEL_ARMS:
        text = text.replace("__TARGET_CHANNEL__", CHANNEL_PHRASES[channel_code])
    text = text.replace("__FORM_DESCRIPTION__", description)
    leftover = re.findall(r"__[A-Z_]+__", text)
    if leftover:
        raise ValueError(f"unsubstituted placeholder(s) {leftover} in arm {arm}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--arm", default=",".join(ARMS),
                        help="comma-separated: " + ", ".join(ARMS) + " (default: both)")
    parser.add_argument("--out-subdir", default="mediation")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arm.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        parser.error(f"--arm: {bad} not in {ARMS}")

    cfg = load_config(args.config)
    main_dir = cfg["paths"]["model_outputs_dir"]
    out_dir = ensure_dir(main_dir / args.out_subdir)
    if "mislabel" in arms:
        assert_miscue_map_matches_video_condition(main_dir)
    backend_name = args.backend or cfg["model"]["backend"]
    prompt_text = load_mediation_prompt(Path(__file__).resolve().parent / "config" / PROMPT_FILE)

    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
    planned = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("status") or "").strip().lower() == "excluded":
                continue
            clip_id = (row.get("clip_id") or "").strip()
            translation = (row.get("translation") or "").strip()
            for index in (1, 2):
                parsed_ev = parse_manifest_event(row, index)
                if parsed_ev is None:
                    continue
                event_id = f"event{index}"
                src = main_dir / f"{clip_id}__{event_id}__combined__run1__{SOURCE_MODE}.json"
                if not src.exists():
                    log.warning("%s/%s: no %s run to take a description from", clip_id, event_id,
                                SOURCE_MODE)
                    continue
                data = json.loads(src.read_text(encoding="utf-8"))
                cands = (data.get("parsed") or {}).get("candidate_actions") or []
                if not cands:
                    continue
                candidate = cands[0]
                description = render_description(candidate.get("form_description", {}))
                if not description:
                    log.warning("%s/%s: empty description after filtering", clip_id, event_id)
                    continue
                for arm in arms:
                    out_path = out_dir / f"{clip_id}__{event_id}__{arm}.json"
                    if out_path.exists() and not args.overwrite:
                        continue
                    planned.append({
                        "clip_id": clip_id, "event_id": event_id, "arm": arm,
                        "out_path": out_path, "translation": translation,
                        "target_channel": parsed_ev[2], "description": description,
                        "label_channel": label_channel_for(arm, parsed_ev[2]),
                        "source_run": src.name, "video_id": row.get("video_id", ""),
                        "source_scores": {k: candidate.get(k) for k in SCORE_KEYS},
                    })

    planned.sort(key=lambda j: (arms.index(j["arm"]), j["clip_id"], j["event_id"]))
    log.info("Planned %d text-only call(s) across arms=%s", len(planned), arms)

    if args.dry_run:
        for job in planned[:2]:
            print(f"\n===== [dry-run] {job['clip_id']}/{job['event_id']} arm={job['arm']} "
                  f"(true channel {job['target_channel']}, source D1 "
                  f"{job['source_scores']['semantic_syntactic_load']}) =====")
            print(fill_prompt(prompt_text, job["arm"], job["translation"],
                              job["label_channel"], job["description"]))
        for job in planned:
            fill_prompt(prompt_text, job["arm"], job["translation"], job["label_channel"],
                        job["description"])
        print(f"\n[dry-run] all {len(planned)} prompts render without unsubstituted fields")
        return

    from lib.model_backends import get_backend, build_backend_kwargs
    backend_kwargs = build_backend_kwargs(backend_name, cfg["model"])
    backend = get_backend(backend_name, **backend_kwargs)

    for done, job in enumerate(planned, start=1):
        label = f"{job['clip_id']}/{job['event_id']}"
        log.info("[%d/%d] %s arm=%s ...", done, len(planned), label, job["arm"])
        row_prompt = fill_prompt(prompt_text, job["arm"], job["translation"],
                                 job["label_channel"], job["description"])
        raw_text = backend.generate(
            video_path=None,
            prompt_text=row_prompt,
            temperature=cfg["model"]["temperature"],
            do_sample=cfg["model"]["do_sample"],
            max_new_tokens=cfg["model"]["max_new_tokens"],
        )
        parsed, schema_valid, schema_errors = None, False, ["not parsed"]
        try:
            parsed = extract_json_object(raw_text)
            schema_valid, schema_errors = validate_output(parsed, SCHEMA)
        except (ValueError, json.JSONDecodeError) as exc:
            schema_errors = [f"JSON parse error: {exc}"]
        if not schema_valid:
            log.warning("%s [%s]: schema validation failed: %s", label, job["arm"], schema_errors)

        job["out_path"].write_text(json.dumps({
            "clip_id": job["clip_id"],
            "event_id": job["event_id"],
            "condition": "mediation",
            "arm": job["arm"],
            "target_channel": job["target_channel"],
            "label_channel_sent": job["label_channel"],
            "label_is_correct": job["label_channel"] == job["target_channel"],
            "video_id": job["video_id"],
            "source_run": job["source_run"],
            "source_scores": job["source_scores"],
            "description_sent": job["description"],
            "prompt_file": PROMPT_FILE,
            "backend": backend_name,
            "model_name": backend.model_name,
            "model_version": backend.model_version,
            "timestamp": datetime.datetime.now().isoformat(),
            "raw_text": raw_text,
            "parsed": parsed,
            "schema_valid": schema_valid,
            "schema_errors": schema_errors,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
