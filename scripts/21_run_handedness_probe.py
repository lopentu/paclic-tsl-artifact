#!/usr/bin/env python3
"""Handedness pre-probe: no handedness representation, or one the prompt overrides?

The open question is precise. The model never self-selects the non-dominant hand, and when told to
describe it, reproduces the dominant-hand percept under the new label. Two explanations survive the
evidence so far:

    (a) NO REPRESENTATION. The model has no working notion of which hand is which, so
        "non-dominant hand" is a label it can attach but not resolve.
    (b) REPRESENTATION OVERRIDDEN. It can establish handedness when that is the question, but a
        prompt naming a target span and demanding one action's description drives it to the
        salient moving hand regardless.

These come apart if handedness is asked FIRST, on its own terms, with no target span and no
action to describe. Three stages, each its own call, in increasing order of how much the target
task intrudes:

    stage1_handedness_only   the window clip, and one question: which hand is dominant, and what
                             in the video tells you? No span, no target action, no output schema
                             beyond the handedness answer. If the model fails here, (a) holds and
                             the paper's open question is closed.
    stage2_two_hands_split   the target clip and span, but the task is changed: report what the
                             dominant hand does AND what the non-dominant hand does, both, with no
                             instruction to pick one. Removes the "choose the single most
                             significant action" pressure while keeping the span. If handedness
                             survives here but not in the main task, (b) holds.
    stage3_primed_target     handedness is established by the model itself in stage 1, then quoted
                             back to it verbatim as part of the main task's prompt. This is the
                             strongest version: ask which hand is dominant before
                             giving the target span. If the model still describes the moving hand
                             after being reminded of its own answer, the label is not being
                             resolved even when the answer is in context.

Stage 3 depends on stage 1's output for the same clip, so run them in order (the script enforces
this and will tell you which stage 1 outputs are missing).

Handedness is a per-SIGNER property, not per-event, so stage 1 runs once per clip -- not once per
event -- and stage 3 reuses that clip's answer for both of its events. Two events from one clip
therefore share a stage-1 answer by construction, which is the correct treatment: a signer does not
change hands between two target spans of the same passage.

What this script does NOT do: decide whether the model's handedness answer is CORRECT. That needs
an expert to label the dominant hand per clip, a few minutes' work, recorded in
`--truth-csv` (columns: clip_id,dominant_hand with values left/right). Without it the script still
reports self-consistency across stages and across cue conditions, which is informative on its own
-- a model that answers "right" and "left" for the same signer in two calls has no stable
representation to override.

Outputs land in model_outputs_dir/handedness_probe/ and results_dir/handedness_probe.md.

Usage:
    python scripts/21_run_handedness_probe.py --stage stage1_handedness_only
    python scripts/21_run_handedness_probe.py --stage stage3_primed_target --mode grounded
    python scripts/21_run_handedness_probe.py --report --truth-csv data/reference_annotations/handedness_truth.csv
"""

import argparse
import csv
import datetime
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import (load_config, get_logger, ensure_dir, REPO_ROOT, MODE_CUES,
                        CUE_ABLATION_MODES, parse_manifest_event, kind_clip_bounds)
from lib.prompt_utils import load_prompt_template, render_blocks, substitute_prompt_fields
from lib.schema_utils import extract_json_object
from lib.video_io import ffprobe_duration

log = get_logger("handedness_probe")

STAGES = ["stage1_handedness_only", "stage2_two_hands_split", "stage3_primed_target"]
MODES = ["grounded"] + CUE_ABLATION_MODES
DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"
EVENT_CLIP_KIND = {"event1": "ev1", "event2": "ev2"}

PROMPT_FILE = "prompts/prompt_c_grounded_attribution.md"
DURATION_TOLERANCE_SEC = 0.2

WHO_TO_WATCH = (
    "Some clips show a single person signing to the camera. Others are interviews, so TWO people "
    "are visible. In every clip only ONE of them is the main signer: the person actively producing "
    "the signed utterance. Answer about that person only."
)
POV = (
    "Describe every direction from the SIGNER's own body, never from the camera's or the viewer's. "
    "A hand that appears on the right-hand side of the frame, when the signer faces the camera, is "
    "on the signer's LEFT."
)

STAGE1_PROMPT = f"""You are watching a short video clip of Taiwan Sign Language (TSL) discourse.

WHO TO WATCH. {WHO_TO_WATCH}

POINT OF VIEW. {POV}

You have exactly one task. Decide which of the signer's two hands is the DOMINANT hand -- the hand
that produces one-handed signs and leads two-handed ones -- and which is the NON-DOMINANT (weak)
hand. Do not describe or evaluate any particular action, and do not translate the clip.

Respond with ONLY a single JSON object, no prose before or after it:

{{
  "dominant_hand": "left" or "right",
  "signer_position": "which person in the frame you took to be the signer",
  "visual_evidence": "one or two sentences naming what in the video told you this",
  "non_dominant_hand_behaviour": "what the OTHER hand does across this clip",
  "confidence": 0.0
}}
"""

STAGE2_PROMPT_TEMPLATE = f"""You are analyzing a short video clip of Taiwan Sign Language (TSL) discourse.

WHO TO WATCH. {WHO_TO_WATCH}

POINT OF VIEW. {POV}

A target span occurs at approximately __EVENT_START_SEC__ to __EVENT_END_SEC__ seconds into THIS
clip. Report what BOTH hands do during that span. Do not choose between them and do not decide
which one is more important -- both fields are required, and either may be at rest.

First state which hand is the dominant one. Then describe each hand separately.

Respond with ONLY a single JSON object, no prose before or after it:

{{
  "dominant_hand": "left" or "right",
  "dominant_hand_action": {{
    "handshape": "...", "movement": "...", "location": "...",
    "at_rest": true or false
  }},
  "non_dominant_hand_action": {{
    "handshape": "...", "movement": "...", "location": "...",
    "at_rest": true or false
  }},
  "which_hand_carries_the_discourse_maintaining_action": "dominant" or "non_dominant" or "neither",
  "confidence": 0.0
}}
"""

# Stage 3 prepends the model's OWN stage-1 answer to the unmodified main-task prompt. Quoting it
# back verbatim, rather than paraphrasing or asserting handedness ourselves, is what makes this a
# test of whether the label gets resolved: the information is present, in context, in the model's
# own words, so a failure here cannot be attributed to the model not having been told.
STAGE3_PREAMBLE = """ESTABLISHED HANDEDNESS FOR THIS SIGNER. Before being shown the target span,
you were asked which of this signer's hands is dominant, from the same footage. Your answer was:

  dominant hand: {dominant_hand}
  evidence you gave: {visual_evidence}
  what you said the other hand does: {non_dominant_hand_behaviour}

Take that as established. "Non-dominant hand" below means the OTHER hand -- not the {dominant_hand}
hand you just identified as dominant.

"""


def load_manifest(manifest_path):
    """[(row, [(event_id, start, end, channel), ...])] for ready rows."""
    out = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("status") or "").strip().lower() == "excluded":
                continue
            if not (row.get("clip_start_sec") or "").strip():
                continue
            events = []
            for index in (1, 2):
                parsed = parse_manifest_event(row, index)
                if parsed is not None:
                    events.append((f"event{index}",) + parsed)
            if events:
                out.append((row, events))
    return out


def stage1_path(out_dir, clip_id):
    return out_dir / f"{clip_id}__stage1_handedness_only.json"


def normalise_hand(value):
    text = str(value or "").strip().lower()
    if "left" in text and "right" not in text:
        return "left"
    if "right" in text and "left" not in text:
        return "right"
    return text or None


def run_stage(args, cfg, backend, jobs, out_dir, prompt_of):
    for job in jobs:
        log.info("Running %s [%s] ...", job["label"], args.stage)
        raw_text = backend.generate(
            video_path=job["clip_path"],
            prompt_text=prompt_of(job),
            context_video_path=job.get("context_clip_path"),
            context_text=job.get("context_text"),
            temperature=cfg["model"]["temperature"],
            do_sample=cfg["model"]["do_sample"],
            max_new_tokens=cfg["model"]["max_new_tokens"],
            **job.get("label_kwargs", {}),
        )
        parsed = None
        try:
            parsed = extract_json_object(raw_text)
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning("%s: unparseable JSON (%s)", job["label"], exc)
        job["out_path"].write_text(json.dumps({
            "clip_id": job["clip_id"],
            "event_id": job.get("event_id"),
            "stage": args.stage,
            "mode": job.get("mode"),
            "video_path": str(job["clip_path"]),
            "backend": backend.model_name,
            "model_version": backend.model_version,
            "temperature": cfg["model"]["temperature"],
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt": prompt_of(job),
            "raw_text": raw_text,
            "parsed": parsed,
        }, indent=2, ensure_ascii=False), encoding="utf-8")


def build_report(out_dir, results_dir, manifest_rows, truth_csv):
    video_of = {r["clip_id"].strip(): (r.get("video_id") or "").strip()
                for r, _ in manifest_rows}
    truth = {}
    if truth_csv and Path(truth_csv).exists():
        with open(truth_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                truth[row["clip_id"].strip()] = normalise_hand(row.get("dominant_hand"))

    lines = ["# Handedness pre-probe\n",
             "\nDoes the model lack a handedness representation, or does the target-span task "
             "override one it has? Generated by `scripts/21_run_handedness_probe.py`.\n"]

    stage1 = {}
    for path in sorted(out_dir.glob("*__stage1_handedness_only.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("parsed"):
            stage1[data["clip_id"]] = data["parsed"]

    lines.append(f"\n## Stage 1 -- handedness asked alone ({len(stage1)} clips)\n\n")
    if stage1:
        answers = Counter(normalise_hand(v.get("dominant_hand")) for v in stage1.values())
        lines.append(f"Answers: {dict(answers)}. ")
        if truth:
            scored = [(c, normalise_hand(v.get("dominant_hand")), truth.get(c))
                      for c, v in sorted(stage1.items()) if truth.get(c)]
            correct = sum(1 for _, a, t in scored if a == t)
            # The baseline must come from the label distribution, not from "1/2 on a two-way
            # choice": if the expert labels are all one hand, a constant answer scores 100% and
            # accuracy carries no evidence that the model perceives handedness at all.
            truth_dist = Counter(t for _, _, t in scored)
            majority_hand, majority_n = truth_dist.most_common(1)[0]
            lines.append(f"Against the expert labels: **{correct}/{len(scored)} correct**. ")
            lines.append(f"Expert label distribution: {dict(truth_dist)}, so always answering "
                         f"`{majority_hand}` scores {majority_n}/{len(scored)} "
                         f"= {majority_n/len(scored):.3f} -- ")
            if correct <= majority_n:
                lines.append("**the model does not beat that constant baseline, so this accuracy "
                             "is not evidence of a handedness representation.** ")
            else:
                lines.append("the model exceeds that constant baseline. ")
            if len(truth_dist) == 1:
                lines.append(f"With every label `{majority_hand}` there is no label variance to "
                             "discriminate on, so accuracy here cannot separate 'perceives "
                             "handedness' from 'always answers the majority hand'. What the labels "
                             "*do* license is the priming reading: the primed answer was the "
                             "correct one on the clips where stage 1 got it right.\n")
            else:
                lines.append("\n")
        else:
            lines.append("No `--truth-csv` supplied, so correctness is not scored -- only the "
                         "answer distribution and cross-stage consistency below. A near-constant "
                         "answer across signers with different real handedness would itself be a "
                         "result.\n")
        # The decisive check needs no expert label at all: a signer's dominant hand does not
        # change between two clips of the same episode, so two different answers for one source
        # video is the model contradicting itself about the same person.
        by_video = defaultdict(dict)
        for clip_id, parsed in stage1.items():
            by_video[video_of.get(clip_id, "?")][clip_id] = normalise_hand(
                parsed.get("dominant_hand"))
        split = {v: c for v, c in by_video.items() if len(set(c.values())) > 1}
        multi = {v: c for v, c in by_video.items() if len(c) > 1}
        lines.append(f"\n**Per-signer consistency: {len(split)} of {len(multi)} source videos "
                     f"contributing more than one clip receive more than one answer** "
                     f"({len(by_video)} videos in total). A signer's dominant hand does not change "
                     f"between two clips of one episode, so this needs no expert label to count "
                     f"as a contradiction.\n\n")
        lines.append("| source video | clips | answers |\n|---|---|---|\n")
        for video, clips in sorted(by_video.items()):
            mark = " **(inconsistent)**" if len(set(clips.values())) > 1 else ""
            lines.append(f"| {video} | {len(clips)} | " +
                         ", ".join(f"{c}={a}" for c, a in sorted(clips.items())) + mark + " |\n")

        lines.append("\n| clip | dominant_hand | confidence | evidence |\n|---|---|---|---|\n")
        for clip_id, parsed in sorted(stage1.items()):
            ev = str(parsed.get("visual_evidence") or "").replace("|", "/")[:110]
            lines.append(f"| {clip_id} | {normalise_hand(parsed.get('dominant_hand'))} | "
                         f"{parsed.get('confidence')} | {ev} |\n")
    else:
        lines.append("No stage-1 outputs found.\n")

    stage2 = {}
    for path in sorted(out_dir.glob("*__stage2_two_hands_split.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("parsed"):
            stage2[(data["clip_id"], data["event_id"])] = data["parsed"]

    lines.append(f"\n## Stage 2 -- both hands reported, no choosing ({len(stage2)} events)\n")
    lines.append("\nThe articulator-neutral version of the task: the span is given but the model "
                 "is not asked to pick one action, so a non-dominant-hand hold no longer competes "
                 "with the moving hand for the single output slot.\n\n")
    if stage2:
        channel_of = {(r["clip_id"], e[0]): e[3] for r, evs in manifest_rows for e in evs}
        lines.append("| item | target_channel | dominant | NDH at rest | carries discourse |\n"
                     "|---|---|---|---|---|\n")
        for key, parsed in sorted(stage2.items()):
            ndh = parsed.get("non_dominant_hand_action") or {}
            lines.append(f"| {key[0]}/{key[1]} | {channel_of.get(key, '?')} | "
                         f"{normalise_hand(parsed.get('dominant_hand'))} | "
                         f"{ndh.get('at_rest')} | "
                         f"{parsed.get('which_hand_carries_the_discourse_maintaining_action')} |\n")

        ndh_items = [k for k in sorted(stage2) if channel_of.get(k) == "non_dominant_hand"]
        if ndh_items:
            hits = sum(1 for k in ndh_items
                       if str(stage2[k].get(
                           "which_hand_carries_the_discourse_maintaining_action") or ""
                       ).lower().startswith("non"))
            lines.append(f"\nOn the {len(ndh_items)} non-dominant-hand target events, the model "
                         f"names the non-dominant hand as carrying the discourse-maintaining "
                         f"action in **{hits}/{len(ndh_items)}**. Compare the main task, where it "
                         f"self-selects that articulator in 0 of 14 positive trials: the "
                         f"difference between those two figures is the size of the "
                         f"task-format effect.\n")

    stage3 = {}
    for path in sorted(out_dir.glob("*__stage3_primed_target__*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        stage3[(data["clip_id"], data["event_id"], data.get("mode"))] = data

    lines.append(f"\n## Stage 3 -- main task, primed with the model's own stage-1 answer "
                 f"({len(stage3)} runs)\n")
    lines.append("\nThe main prompt, unchanged, with the model's own handedness answer quoted back "
                 "to it first. `channel_judged` and the `hands` field are what matter: if the "
                 "description still tracks the moving hand, the label is not being resolved even "
                 "with the answer in context.\n\n")
    if stage3:
        lines.append("| item | mode | primed dominant | channel_judged | `hands` field |\n"
                     "|---|---|---|---|---|\n")
        for (clip_id, event_id, mode), data in sorted(stage3.items()):
            parsed = data.get("parsed") or {}
            cands = parsed.get("candidate_actions") or [{}]
            cand = cands[0] if cands else {}
            hands = str((cand.get("form_description") or {}).get("hands") or "")[:40]
            primed = re.search(r"dominant hand: (\w+)", data.get("prompt", ""))
            lines.append(f"| {clip_id}/{event_id} | {mode} | "
                         f"{primed.group(1) if primed else '?'} | "
                         f"{cand.get('channel_judged')} | {hands} |\n")

    # Cross-stage self-consistency needs no ground truth: one signer cannot have two dominant
    # hands, so disagreement between stages is by itself evidence of no stable representation.
    both = [(c, e) for (c, e) in stage2 if c in stage1]
    if both:
        agree = sum(1 for c, e in both
                    if normalise_hand(stage1[c].get("dominant_hand")) ==
                       normalise_hand(stage2[(c, e)].get("dominant_hand")))
        lines.append(f"\n## Self-consistency across stages\n\nStage 1 and stage 2 name the same "
                     f"dominant hand on **{agree}/{len(both)}** events. A signer has one dominant "
                     f"hand, so any disagreement here is the model contradicting itself about the "
                     f"same footage and needs no expert label to count as a failure.\n")

    out_path = results_dir / "handedness_probe.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    log.info("Wrote %s", out_path)
    print("".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--stage", choices=STAGES, default=None)
    parser.add_argument("--mode", choices=MODES, default="grounded",
                        help="cue mode for stage 3 only (the main task's prompt)")
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--channel", default=None,
                        help="stage 2/3: restrict to events with this manifest target_channel")
    parser.add_argument("--pad-sec", type=float, default=0.3)
    parser.add_argument("--out-subdir", default="handedness_probe")
    parser.add_argument("--truth-csv", default=None,
                        help="optional expert labels (clip_id,dominant_hand) for scoring stage 1")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", action="store_true", help="build the report and exit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    c_clips_dir = cfg["paths"]["c_clips_dir"]
    out_dir = ensure_dir(cfg["paths"]["model_outputs_dir"] / args.out_subdir)
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    manifest_rows = load_manifest(Path(args.manifest) if args.manifest else DEFAULT_MANIFEST)

    if args.report:
        build_report(out_dir, results_dir, manifest_rows, args.truth_csv)
        return
    if not args.stage:
        parser.error("--stage is required unless --report is given")

    scripts_dir = Path(__file__).resolve().parent
    main_prompt = load_prompt_template(scripts_dir / "config" / PROMPT_FILE)

    jobs, missing_stage1 = [], []
    for row, events in manifest_rows:
        clip_id = row["clip_id"].strip()
        if args.clip_id and clip_id != args.clip_id:
            continue
        clip_start_sec, clip_end_sec = float(row["clip_start_sec"]), float(row["clip_end_sec"])
        window_clip = c_clips_dir / f"{clip_id}_window_model.mp4"

        if args.stage == "stage1_handedness_only":
            # Once per CLIP, not per event: handedness is a signer-level property, so asking it
            # twice for two events of the same passage would be asking the same question twice.
            out_path = stage1_path(out_dir, clip_id)
            if out_path.exists() and not args.overwrite:
                continue
            jobs.append({"clip_id": clip_id, "label": clip_id, "clip_path": window_clip,
                         "out_path": out_path,
                         "label_kwargs": {"video_label": "The following clip shows the signer:"}})
            continue

        for event_id, ev_start, ev_end, chan_code in events:
            if args.channel and chan_code != args.channel:
                continue
            event_kind = EVENT_CLIP_KIND[event_id]
            event_clip = c_clips_dir / f"{clip_id}_{event_kind}_model.mp4"
            kind_start, kind_end = kind_clip_bounds(event_kind, clip_start_sec, clip_end_sec,
                                                    ev_start, ev_end, args.pad_sec)
            rel_start, rel_end = ev_start - kind_start, ev_end - kind_start

            if args.stage == "stage2_two_hands_split":
                out_path = out_dir / f"{clip_id}__{event_id}__stage2_two_hands_split.json"
                if out_path.exists() and not args.overwrite:
                    continue
                jobs.append({"clip_id": clip_id, "event_id": event_id,
                             "label": f"{clip_id}/{event_id}", "clip_path": event_clip,
                             "out_path": out_path, "rel_start": rel_start, "rel_end": rel_end,
                             "expected_duration": kind_end - kind_start})
            else:  # stage3_primed_target
                s1 = stage1_path(out_dir, clip_id)
                if not s1.exists():
                    missing_stage1.append(clip_id)
                    continue
                s1_data = json.loads(s1.read_text(encoding="utf-8"))
                if not s1_data.get("parsed"):
                    missing_stage1.append(f"{clip_id} (stage 1 output unparseable)")
                    continue
                out_path = out_dir / (f"{clip_id}__{event_id}__stage3_primed_target__"
                                      f"{args.mode}.json")
                if out_path.exists() and not args.overwrite:
                    continue
                jobs.append({"clip_id": clip_id, "event_id": event_id, "mode": args.mode,
                             "label": f"{clip_id}/{event_id}", "clip_path": event_clip,
                             "context_clip_path": c_clips_dir / f"{clip_id}_window_model.mp4",
                             "out_path": out_path, "rel_start": rel_start, "rel_end": rel_end,
                             "expected_duration": kind_end - kind_start,
                             "translation": (row.get("translation") or "").strip(),
                             "chan_code": chan_code, "stage1": s1_data["parsed"]})

    if missing_stage1:
        log.error("stage 3 needs stage 1 first for: %s. Run "
                  "`--stage stage1_handedness_only` before this stage.",
                  ", ".join(sorted(set(missing_stage1))))
        if not jobs:
            return

    def prompt_of(job):
        if args.stage == "stage1_handedness_only":
            return STAGE1_PROMPT
        if args.stage == "stage2_two_hands_split":
            return (STAGE2_PROMPT_TEMPLATE
                    .replace("__EVENT_START_SEC__", f"{job['rel_start']:.1f}")
                    .replace("__EVENT_END_SEC__", f"{job['rel_end']:.1f}"))
        cues = MODE_CUES[job["mode"]]
        keep = {"MEANING_GIVEN" if cues["translation"] else "MEANING_WITHHELD",
                "CHANNEL_GIVEN" if cues["channel"] else "CHANNEL_WITHHELD"}
        body = substitute_prompt_fields(
            render_blocks(main_prompt, keep), item_label=job["label"], mode=job["mode"],
            translation=job["translation"] if cues["translation"] else None,
            channel_code=job["chan_code"] if cues["channel"] else None,
            rel_start=job["rel_start"], rel_end=job["rel_end"])
        s1 = job["stage1"]
        return STAGE3_PREAMBLE.format(
            dominant_hand=normalise_hand(s1.get("dominant_hand")) or "unstated",
            visual_evidence=str(s1.get("visual_evidence") or "(none given)").strip(),
            non_dominant_hand_behaviour=str(
                s1.get("non_dominant_hand_behaviour") or "(none given)").strip(),
        ) + body

    log.info("Planned %d call(s), stage=%s", len(jobs), args.stage)
    if args.dry_run:
        for job in jobs[:3]:
            print(f"\n[dry-run] {job['label']} clip={job['clip_path'].name} -> "
                  f"{job['out_path'].name}\n{'-' * 70}\n{prompt_of(job)}")
        if len(jobs) > 3:
            print(f"\n... and {len(jobs) - 3} more (prompts differ only in the substituted "
                  f"fields)")
        return

    for job in list(jobs):
        if not job["clip_path"].exists():
            log.warning("%s: missing clip %s", job["label"], job["clip_path"])
            jobs.remove(job)
            continue
        if "expected_duration" in job:
            actual = ffprobe_duration(job["clip_path"])
            if abs(actual - job["expected_duration"]) > DURATION_TOLERANCE_SEC:
                log.error("%s: clip duration %.2fs != expected %.2fs -- --pad-sec probably does "
                          "not match what 03b cut. Skipping rather than sending a timestamp that "
                          "may not match the video.", job["label"], actual,
                          job["expected_duration"])
                jobs.remove(job)

    from lib.model_backends import get_backend, build_backend_kwargs
    backend_name = args.backend or cfg["model"]["backend"]
    backend = get_backend(backend_name, **build_backend_kwargs(backend_name, cfg["model"]))

    run_stage(args, cfg, backend, jobs, out_dir, prompt_of)
    log.info("Done. Outputs in %s. Build the report with --report.", out_dir)


if __name__ == "__main__":
    main()
