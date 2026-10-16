#!/usr/bin/env python3
"""Mis-cue condition: give the model the WRONG target articulator and see what follows the label.

The paper's central claim is that naming an articulator can change what an answer is *called*
without changing what was *seen*. Everything supporting it so far is either correlational (the
label-only predictor) or rests on 7 non-dominant-hand events. This condition tests it directly
on all 55, and it is the one manipulation that makes the two accounts predict opposite things:

    the cue SELECTS a percept   ->  a manual target cued as "facial activity" comes back with a
                                    description of the FACE (and the hand fields marked N/A)
    the cue RELABELS a percept  ->  the description is of the same action as ever, only
                                    `channel_judged` and the functional scores move to match the
                                    label that was supplied

Note what this buys that no existing cell does. In `grounded` vs `ablate_channel` the cue's
naming and its target-resolution are confounded (see the paper's "Scope of claims"): the correct
label both names a body part and identifies which co-occurring action is meant, so an adequacy
gain cannot be attributed to naming alone. Under a mis-cue the two point at DIFFERENT answers --
resolution still points at the annotated target, naming points elsewhere -- so whichever the
output follows is identified. The confound is dissolved by the design rather than hedged in
prose.

**Which wrong label.** Every target is mis-cued ACROSS the manual/nonmanual boundary:

    manual (32)             -> face
    non_dominant_hand (7)   -> face
    head_torso (1)          -> face
    face (15)               -> manual

That boundary is chosen because it is the one place the study has already shown the cue CAN
redirect content: naming the face on a real face target moves the substantive description into
the `nonmanual` field and marks the hand field not applicable on 15/15 items. So a null result
here is interpretable -- the cue is being asked to do something it demonstrably can do, on a
model that demonstrably complies with the label 55/55 when the label is correct. Both directions
are run because "manual is a strong default" would otherwise explain a one-directional result.

The prompt is BYTE-IDENTICAL to the `grounded` prompt except for the phrase substituted into
__TARGET_CHANNEL__. Nothing is added to warn, hedge, or draw attention to the articulator; a
model that notices the mismatch and says so is producing a result, not a failed run, and its
output is recorded verbatim like any other.

Outputs land in their own namespace (`--out-subdir`, default `miscue`) so they can never be
pooled with the main 220 runs. Analyse with 29_analyze_miscue_and_mediation.py.

**Two schemes** (`--scheme`, see SCHEMES below for the full rationale):
    cross_boundary  the manual/nonmanual swap described above -- the paper's mis-cue condition
    resting_ndh     cue every target as the non-dominant hand, so the cue is held constant while
                    its truth varies. Asks a question cross_boundary cannot: will the model rate
                    an articulator it has itself just described as doing nothing?

Usage:
    python scripts/26_run_miscue_condition.py --dry-run
    python scripts/26_run_miscue_condition.py
    python scripts/26_run_miscue_condition.py --mode ablate_translation --channel face
    python scripts/26_run_miscue_condition.py --scheme resting_ndh --out-subdir miscue_resting
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import (load_config, get_logger, ensure_dir, REPO_ROOT, MODE_CUES,
                        parse_manifest_event, kind_clip_bounds)
from lib.prompt_utils import load_prompt_template, render_blocks, substitute_prompt_fields
from lib.schema_utils import extract_json_object, validate_output, load_schema

log = get_logger("run_miscue")

# Only the two CHANNEL_GIVEN modes can be mis-cued: the withheld modes have no label to falsify.
MODES = ["grounded", "ablate_translation"]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]

# The mis-cue map. See the module docstring for why it crosses the manual/nonmanual boundary in
# both directions rather than, say, sending non_dominant_hand -> manual (which is its own
# supertype and would make a "correct" reading of the mis-cue possible).
MISCUE_OF = {"manual": "face", "non_dominant_hand": "face", "head_torso": "face",
             "face": "manual"}

# Second scheme (2026-08-15): cue EVERY target as the non-dominant hand, whatever it really is.
#
# This is deliberately NOT uniformly a mis-cue, and that is what it buys. `cross_boundary` asks
# whether a false label redirects the answer; it cannot ask whether the model will rate an
# articulator that is *doing nothing*, because face and hands are both plausibly active in signing.
# Here the cue is held constant while its truth varies, and the strata come from the model's OWN
# `form_description.hands` in the grounded run rather than from our annotation:
#
#     hands = "non-dominant hand"  (7)   cue is CORRECT          -- internal control
#     hands = "both hands"/"both"  (16)  cue is CORRECT          -- internal control
#     hands = "dominant hand"      (16)  cue names a RESTING part, with an active hand alongside
#     hands = "not visible"        (15)  cue names a RESTING part, with no manual activity at all
#     hands = "not visible or not actively signing" (1)          -- as above
#
# The last two strata are the condition proper: a confident functional rating there is a rating of
# an articulator the model itself has just testified was not doing anything. The split between them
# matters -- with an active hand alongside, the model can relabel a percept it did form; with no
# manual activity at all, there is nothing to relabel and a rating has to be generated outright.
#
# Because the criterion is the model's own description, "you annotated it wrong" is not available
# as a reading of the result. Stratification happens in analysis; the runner is uniform.
RESTING_NDH_OF = {c: "non_dominant_hand" for c in CHANNELS}

SCHEMES = {"cross_boundary": MISCUE_OF, "resting_ndh": RESTING_NDH_OF}

PROMPT_FILE = "prompts/prompt_c_grounded_attribution.md"
SCHEMA_FILE = "prompts/prompt_c_output.schema.json"
DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"

# Identical to 07b/19's `combined` clip kind, which is what the paper's results are computed on.
EVENT_CLIP_KIND = {"event1": "ev1", "event2": "ev2"}
COMBINED_VIDEO_LABEL = ("The following clip is a close-up cut of the target action, padded "
                        "slightly on each side. Describe and score THIS clip's target action:")
COMBINED_CONTEXT_LABEL = ("The following clip is the full analysis window containing the target "
                          "action, shown first for surrounding context (signing before/after, "
                          "discourse scope) -- it is the same passage as the close-up clip shown "
                          "after it, not separate or preceding discourse:")
COMBINED_CONTEXT_TEXT = ("Two clips are given above. The first is broader context -- use it to "
                         "judge how the target action's timing relates to the surrounding "
                         "signing. The second is a close-up of the target action itself: that is "
                         "the clip \"this clip\" refers to below, and the one to describe and "
                         "score.")


def fill_prompt(prompt_text, mode, item_label, translation, chan_code, rel_start, rel_end):
    cues = MODE_CUES[mode]
    keep = {"MEANING_GIVEN" if cues["translation"] else "MEANING_WITHHELD",
            "CHANNEL_GIVEN" if cues["channel"] else "CHANNEL_WITHHELD"}
    return substitute_prompt_fields(
        render_blocks(prompt_text, keep), item_label=item_label, mode=mode,
        translation=translation if cues["translation"] else None,
        channel_code=chan_code if cues["channel"] else None,
        rel_start=rel_start, rel_end=rel_end,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--mode", default="grounded",
                        help="one or more of " + ", ".join(MODES) + " (comma-separated; all "
                             "share one model load)")
    parser.add_argument("--scheme", choices=sorted(SCHEMES), default="cross_boundary",
                        help="which articulator to substitute: cross_boundary (default; the "
                             "manual/nonmanual swap the paper reports) or resting_ndh (cue every "
                             "target as the non-dominant hand). ALWAYS pair a non-default scheme "
                             "with its own --out-subdir; filenames do not encode the scheme.")
    parser.add_argument("--channel", choices=CHANNELS, default=None,
                        help="restrict to events whose TRUE target_channel is this")
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--pad-sec", type=float, default=0.3,
                        help="MUST match what 03b_cut_event_clips.py used (default 0.3)")
    parser.add_argument("--out-subdir", default="miscue",
                        help="subdirectory of model_outputs_dir for these runs (default: "
                             "miscue) -- keeps them out of the main 220-run set")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        parser.error(f"--mode: {bad} not in {MODES}")

    cue_map = SCHEMES[args.scheme]
    # Output filenames carry the mode but not the scheme, so two schemes sharing an --out-subdir
    # would silently overwrite each other and leave a directory whose files cannot be told apart.
    # Refuse rather than rely on the caller remembering.
    if args.scheme != "cross_boundary" and args.out_subdir == parser.get_default("out_subdir"):
        parser.error(f"--scheme {args.scheme} needs its own --out-subdir (default "
                     f"'{args.out_subdir}' holds the cross_boundary runs)")

    cfg = load_config(args.config)
    c_clips_dir = cfg["paths"]["c_clips_dir"]
    out_dir = ensure_dir(cfg["paths"]["model_outputs_dir"] / args.out_subdir)

    backend_name = args.backend or cfg["model"]["backend"]
    n_runs = args.n_runs or cfg["model"]["n_runs_per_item"]

    scripts_dir = Path(__file__).resolve().parent
    prompt_text = load_prompt_template(scripts_dir / "config" / PROMPT_FILE)
    schema = load_schema(scripts_dir / "config" / SCHEMA_FILE)

    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    planned = []
    for row in rows:
        clip_id = (row.get("clip_id") or "").strip()
        if (row.get("status") or "").strip().lower() == "excluded":
            continue
        if args.clip_id and clip_id != args.clip_id:
            continue
        start_raw = (row.get("clip_start_sec") or "").strip()
        end_raw = (row.get("clip_end_sec") or "").strip()
        if not start_raw or not end_raw:
            continue
        clip_start_sec, clip_end_sec = float(start_raw), float(end_raw)
        translation = (row.get("translation") or "").strip()

        for index in (1, 2):
            parsed = parse_manifest_event(row, index)
            if parsed is None:
                continue
            ev_start, ev_end, true_channel = parsed
            if args.channel and true_channel != args.channel:
                continue
            event_id = f"event{index}"
            event_kind = EVENT_CLIP_KIND[event_id]
            kind_start, _ = kind_clip_bounds(event_kind, clip_start_sec, clip_end_sec,
                                             ev_start, ev_end, args.pad_sec)
            for mode in modes:
                for run_idx in range(1, n_runs + 1):
                    out_path = out_dir / (f"{clip_id}__{event_id}__combined__run{run_idx}__"
                                          f"miscue_{mode}.json")
                    if out_path.exists() and not args.overwrite:
                        continue
                    planned.append({
                        "clip_id": clip_id, "event_id": event_id, "run_idx": run_idx,
                        "mode": mode, "out_path": out_path, "translation": translation,
                        "true_channel": true_channel,
                        "cued_channel": cue_map[true_channel],
                        "rel_start": ev_start - kind_start, "rel_end": ev_end - kind_start,
                        "target_clip": c_clips_dir / f"{clip_id}_{event_kind}_model.mp4",
                        "context_clip": c_clips_dir / f"{clip_id}_window_model.mp4",
                        "video_id": row.get("video_id", ""),
                    })

    planned.sort(key=lambda j: (modes.index(j["mode"]), j["clip_id"], j["event_id"]))
    log.info("Planned %d call(s) across mode=%s, backend=%s (one model load)",
             len(planned), modes, backend_name)
    by_pair = {}
    for job in planned:
        by_pair[(job["true_channel"], job["cued_channel"])] = \
            by_pair.get((job["true_channel"], job["cued_channel"]), 0) + 1
    for (true_c, cued_c), n in sorted(by_pair.items()):
        log.info("  %-18s cued as %-18s %d", true_c, cued_c, n)

    if args.dry_run:
        for job in planned:
            label = f"{job['clip_id']}/{job['event_id']}"
            try:
                fill_prompt(prompt_text, job["mode"], label, job["translation"],
                            job["cued_channel"], job["rel_start"], job["rel_end"])
                status = "ok"
            except ValueError as exc:
                status = f"WOULD FAIL: {exc}"
            print(f"[dry-run] {label} true={job['true_channel']} -> cued={job['cued_channel']} "
                  f"target=[{job['rel_start']:.1f},{job['rel_end']:.1f}]s "
                  f"-> {job['out_path'].name}  [{status}]")
        return

    from lib.model_backends import get_backend, build_backend_kwargs
    backend_kwargs = build_backend_kwargs(backend_name, cfg["model"])
    backend = get_backend(backend_name, **backend_kwargs)

    manifest_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "script": "26_run_miscue_condition.py",
        "condition": "miscue",
        "scheme": args.scheme,
        "miscue_map": cue_map,
        "prompt_identical_to_main_run_except_target_channel": True,
        "backend": backend_name,
        "model_name": backend.model_name,
        "model_version": backend.model_version,
        "prompt_modes": modes,
        "prompt_file": PROMPT_FILE,
        "clip_kind": "combined",
        "channel_filter": args.channel,
        "pad_sec": args.pad_sec,
        "backend_kwargs": backend_kwargs,
        "temperature": cfg["model"]["temperature"],
        "do_sample": cfg["model"]["do_sample"],
        "max_new_tokens": cfg["model"]["max_new_tokens"],
        "n_planned": len(planned),
    }
    mpath = out_dir / "run_manifest_miscue.json"
    history = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else []
    history.append(manifest_entry)
    mpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    for done, job in enumerate(planned, start=1):
        label = f"{job['clip_id']}/{job['event_id']}"
        if not job["target_clip"].exists() or not job["context_clip"].exists():
            log.warning("%s: missing source clip(s), skipping", label)
            continue

        log.info("[%d/%d] %s %s cued as %s (mode=%s) ...", done, len(planned), label,
                 job["true_channel"], job["cued_channel"], job["mode"])
        row_prompt = fill_prompt(prompt_text, job["mode"], label, job["translation"],
                                 job["cued_channel"], job["rel_start"], job["rel_end"])
        raw_text = backend.generate(
            video_path=job["target_clip"],
            prompt_text=row_prompt,
            temperature=cfg["model"]["temperature"],
            do_sample=cfg["model"]["do_sample"],
            max_new_tokens=cfg["model"]["max_new_tokens"],
            context_video_path=job["context_clip"],
            context_text=COMBINED_CONTEXT_TEXT,
            video_label=COMBINED_VIDEO_LABEL,
            context_video_label=COMBINED_CONTEXT_LABEL,
        )

        parsed, schema_valid, schema_errors = None, False, ["not parsed"]
        try:
            parsed = extract_json_object(raw_text)
            schema_valid, schema_errors = validate_output(parsed, schema)
        except (ValueError, json.JSONDecodeError) as exc:
            schema_errors = [f"JSON parse error: {exc}"]
        if not schema_valid:
            # A model that objects to the mis-cue, or refuses, is a RESULT. Log and keep going;
            # never retry or reword, which would make the condition non-comparable.
            log.warning("%s [miscue]: schema validation failed: %s", label, schema_errors)

        job["out_path"].write_text(json.dumps({
            "clip_id": job["clip_id"],
            "event_id": job["event_id"],
            "clip_kind": "combined",
            "condition": "miscue",
            "scheme": args.scheme,
            "target_channel": job["true_channel"],
            "target_channel_cued": job["cued_channel"],
            "cue_is_correct": job["cued_channel"] == job["true_channel"],
            "video_path": str(job["target_clip"]),
            "context_clip_path": str(job["context_clip"]),
            "video_id": job["video_id"],
            "mode": job["mode"],
            "prompt_file": PROMPT_FILE,
            "prompt_identical_to_main_run_except_target_channel": True,
            "cues": MODE_CUES[job["mode"]],
            "pad_sec": args.pad_sec,
            "run_index": job["run_idx"],
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
