#!/usr/bin/env python3
"""Text-only metalinguistic probe: does the model hold the knowledge the paper says it holds?

The study frames its object as *metalinguistic knowledge* and reports a
dissociation between that knowledge and visual grounding -- but it never measures the knowledge in
isolation. Everything it observes is measured through a video task, so "knows the categories but
cannot anchor them" is an inference from the model using the vocabulary fluently, not a result.

This closes that gap with no video at all. The model is asked, in plain text, the questions whose
answers its video behaviour presupposes:

  * what functions the non-dominant hand can carry in signed discourse, and how much
    semantic/syntactic load a non-dominant-hand hold typically carries -- the exact judgement it
    scores at the nonmanual level (D1 ~ 1) on video while the reference puts it high (2.14);
  * whether a non-dominant-hand action counts as a manual action -- which bears directly on the
    inventory-overlap objection: if the model itself says
    NON-DOMINANT HAND is a subtype of MANUAL, then answering `manual` on a non-dominant-hand
    target is a defensible superset answer and the 0/14 recall figure needs restating;
  * how a grammatical nonmanual is told apart from an affective one, and whether facial activity
    can carry semantic/syntactic load -- the FACE D1 = 1 mapping;
  * how handedness is established for a signer at all -- the mechanism the paper says is missing.

Two answer formats per question, run separately:

    prose    free text. What the model volunteers, in its own framing. This is the format the
             claim "it uses the vocabulary fluently" is about.
    scored   the same question forced onto the study's own 0-3 D1 scale, so the answer is directly
             comparable to what the model output on video for the same articulator. This is the
             comparison that makes the dissociation a measurement rather than an impression: the
             model asserting in text that a non-dominant-hand hold carries D1 = 3, having scored
             it 1 on video, is a contradiction internal to one model.

No video, no clip manifest, no reference annotation, no human rating -- this is cheap enough to
re-run for any model under comparison, which is the point (the same probe is the natural first
thing to run on a second checkpoint).

Outputs one JSON per (question, format) under model_outputs_dir/metalinguistic_probe/, plus a
readable transcript in results_dir/metalinguistic_probe.md.

Usage:
    python scripts/20_run_metalinguistic_probe.py --config scripts/config/pipeline_config.qwen3vl_8b.yaml
    python scripts/20_run_metalinguistic_probe.py --list
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, get_logger, ensure_dir

log = get_logger("metalinguistic_probe")

SCALE_REMINDER = (
    "Use this 0-3 scale, which is the same scale used elsewhere in this study for D1 "
    "(semantic/syntactic load):\n"
    "0 = no clear contribution to propositional content, argument/event structure, grammatical "
    "marking, referential tracking, or discourse structure\n"
    "1 = weak or uncertain contribution\n"
    "2 = clear contribution\n"
    "3 = essential; removing it would change core meaning, a grammatical relation, referent "
    "tracking, or discourse structure\n"
)

# Each probe: (id, what it tests, the question). Wording is kept close to the study's own
# vocabulary so an answer is comparable to the model's video-task output, but no probe mentions
# any clip, any item, or any score the model previously produced -- nothing is being asked to
# reconcile, so a contradiction cannot be prompted into existence.
PROBES = [
    ("ndh_functions",
     "Does the model know the non-dominant hand's discourse functions at all?",
     "In sign language discourse, what functions can the NON-DOMINANT (weak) hand carry? List "
     "them, and for each say briefly whether it contributes to propositional or grammatical "
     "content, or to pragmatic/affective enrichment, or both."),

    ("ndh_hold_load",
     "Does its stated D1 for a non-dominant-hand hold match the D1 it assigns on video?",
     "Consider a signer who produces a sign with the dominant hand while the NON-DOMINANT hand "
     "stays held in place, barely moving, from an earlier sign. Typically, how much "
     "semantic/syntactic load does that held non-dominant hand itself carry?"),

    ("ndh_is_manual",
     "Are MANUAL and NON-DOMINANT HAND mutually exclusive labels for this model? "
     "(inventory-overlap objection)",
     "A response inventory for 'which articulator carries this action' offers exactly these four "
     "options: MANUAL, FACE, HEAD_TORSO, NON-DOMINANT HAND. An action is carried by a signer's "
     "non-dominant hand alone. Which single option is correct, and is any other option also true "
     "of that action? Answer explicitly whether NON-DOMINANT HAND is a subtype of MANUAL or a "
     "separate category from it."),

    ("face_grammatical",
     "Does it know facial activity can be grammatical? (its FACE D1 = 1 mapping)",
     "Can facial activity in signed discourse carry semantic or syntactic content, as opposed to "
     "only affect or emphasis? If so, give examples of grammatical nonmanual markers, and say how "
     "a grammatical facial marker can be distinguished from an affective facial expression."),

    ("face_nmm_load",
     "Its stated D1 for a grammatical nonmanual marker.",
     "Consider a signer producing a grammatical nonmanual marker with the face -- for example the "
     "brow and head configuration marking a conditional clause or a topic. Typically, how much "
     "semantic/syntactic load does that facial marking carry?"),

    ("manual_lexical_load",
     "Its stated D1 for an ordinary lexical sign -- the reference's ceiling category.",
     "Consider an ordinary lexical sign produced with the dominant hand, contributing a content "
     "word to the utterance. Typically, how much semantic/syntactic load does it carry?"),

    ("handedness_procedure",
     "Does it know how handedness is established -- the mechanism the paper says is missing?",
     "How do you determine which of a signer's hands is the dominant one from a video recording? "
     "Describe what you would look at, and say how much of the recording you would need to see to "
     "decide."),

    ("ndh_vs_face_ranking",
     "Direct ranking: does it order the non-dominant hand above or below nonmanuals?",
     "Rank these three by how much semantic/syntactic load they typically carry in signed "
     "discourse, highest first: (a) a lexical sign on the dominant hand, (b) a held non-dominant "
     "hand maintaining a discourse referent, (c) an affective facial expression. Then give each a "
     "0-3 value on the scale below."),
]

FORMAT_INSTRUCTIONS = {
    "prose": "Answer in at most 120 words. Plain prose, no JSON.",
    "scored": (SCALE_REMINDER +
               "Respond with ONLY a single JSON object and no prose around it:\n"
               '{"value": <integer 0-3, or an object mapping each item you were asked to rate to '
               'its integer 0-3>, "one_sentence_reason": "..."}'),
}

# Probes that ask for a magnitude are the ones worth running in `scored` format; the others have
# no single number to give, so forcing one would invent a quantity rather than measure one.
SCORED_PROBES = {"ndh_hold_load", "face_nmm_load", "manual_lexical_load", "ndh_vs_face_ranking"}


def build_prompt(question, answer_format):
    return (
        "You are answering a question about sign language linguistics. This is a knowledge "
        "question about signed languages in general, not about any particular video.\n\n"
        f"{question}\n\n{FORMAT_INSTRUCTIONS[answer_format]}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--probe", default=None, help="run only this probe id")
    parser.add_argument("--out-subdir", default="metalinguistic_probe")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the probes and exit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list:
        for probe_id, tests, question in PROBES:
            formats = "prose, scored" if probe_id in SCORED_PROBES else "prose"
            print(f"\n[{probe_id}]  ({formats})\n  tests: {tests}\n  Q: {question}")
        return

    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg["paths"]["model_outputs_dir"] / args.out_subdir)
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    backend_name = args.backend or cfg["model"]["backend"]

    jobs = []
    for probe_id, tests, question in PROBES:
        if args.probe and probe_id != args.probe:
            continue
        formats = ["prose"] + (["scored"] if probe_id in SCORED_PROBES else [])
        for answer_format in formats:
            out_path = out_dir / f"{probe_id}__{answer_format}.json"
            if out_path.exists() and not args.overwrite:
                continue
            jobs.append((probe_id, tests, question, answer_format, out_path))

    log.info("Planned %d text-only probe call(s), backend=%s", len(jobs), backend_name)
    if args.dry_run:
        for probe_id, _, question, answer_format, out_path in jobs:
            print(f"\n[dry-run] {probe_id}/{answer_format} -> {out_path.name}\n"
                  f"{build_prompt(question, answer_format)}")
        return

    from lib.model_backends import get_backend, build_backend_kwargs
    backend_kwargs = build_backend_kwargs(backend_name, cfg["model"])
    backend = get_backend(backend_name, **backend_kwargs)

    written = []
    for probe_id, tests, question, answer_format, out_path in jobs:
        log.info("Probing %s [%s] ...", probe_id, answer_format)
        prompt = build_prompt(question, answer_format)
        raw_text = backend.generate(
            video_path=None, prompt_text=prompt,
            temperature=cfg["model"]["temperature"], do_sample=cfg["model"]["do_sample"],
            max_new_tokens=args.max_new_tokens,
        )
        parsed = None
        if answer_format == "scored":
            try:
                from lib.schema_utils import extract_json_object
                parsed = extract_json_object(raw_text)
            except (ValueError, json.JSONDecodeError) as exc:
                log.warning("%s: could not parse scored answer (%s)", probe_id, exc)
        out_path.write_text(json.dumps({
            "probe_id": probe_id,
            "tests": tests,
            "question": question,
            "answer_format": answer_format,
            "video_shown": False,
            "backend": backend_name,
            "model_name": backend.model_name,
            "model_version": backend.model_version,
            "temperature": cfg["model"]["temperature"],
            "do_sample": cfg["model"]["do_sample"],
            "max_new_tokens": args.max_new_tokens,
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt": prompt,
            "raw_text": raw_text,
            "parsed": parsed,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append((probe_id, tests, answer_format, raw_text, parsed))

    if written:
        lines = ["# Text-only metalinguistic probe\n",
                 f"\nModel: `{backend.model_version}` ({backend.model_name}), no video shown, "
                 f"temperature {cfg['model']['temperature']}. Generated by "
                 f"`scripts/20_run_metalinguistic_probe.py`.\n",
                 "\nThe point of comparison is the model's own video output for the same "
                 "articulator: what it asserts here in text, against what it scored on the "
                 "clips. Where the two disagree the contradiction is internal to one model and "
                 "needs no reference annotation to establish.\n"]
        for probe_id, tests, answer_format, raw_text, parsed in written:
            lines.append(f"\n## `{probe_id}` / {answer_format}\n\n*{tests}*\n\n")
            if parsed is not None:
                lines.append(f"Parsed: `{json.dumps(parsed, ensure_ascii=False)}`\n\n")
            lines.append("> " + raw_text.strip().replace("\n", "\n> ") + "\n")
        out_path = results_dir / "metalinguistic_probe.md"
        existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        out_path.write_text(existing + "".join(lines) if existing else "".join(lines),
                            encoding="utf-8")
        log.info("Wrote %s", out_path)

    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
