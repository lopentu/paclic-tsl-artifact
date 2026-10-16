#!/usr/bin/env python3
"""Visual-null and degraded-video baselines: how much of the model's output survives when the
video carries no usable information?

Answers the first-order objection to the central claim. The write-up
argues that functional attribution is cue-driven rather than visually grounded, but its evidence
is a channel-only *statistical* predictor -- a property of the reference annotation's structure,
not of the model's behaviour. The decisive version is behavioural: hold the prompt byte-identical,
destroy the visual information, and see whether the scores move.

Five visual conditions, from total removal to a targeted removal of just one property:

    real      the actual model-input clip -- reproduction check. Under greedy decoding this
              should match 07b's existing output for the same item and mode exactly; if it does
              not, something in the pipeline drifted and none of the other rows can be trusted.
    none      NO video is sent at all. Text-only: translation + articulator label + target span.
              The strongest form of this control.
    black     all-black frames, same duration/resolution/frame rate. A video-shaped input with
              zero content, which separates "needs pixels" from "needs a video argument at all".
    static    the clip's middle frame held for the clip's full duration. Appearance and posture
              intact, MOTION removed. A movement description that survives this was not read off
              a trajectory.
    shuffled  the clip's own frames in a fixed pseudo-random order. Every frame is still present
              -- only temporal order is destroyed. The tightest control of the four, since
              appearance, handshape and articulator position are all preserved.

THE PROMPT IS NEVER CHANGED. Not even under `none`, where it still says "You are analyzing a
short video clip" and refers to "THIS clip". That is deliberate and is the point of the design:
the moment the prompt is edited to acknowledge a missing video, two things change at once and the
comparison stops being an ablation. A model that answers the question anyway, with the same scores
it gives for the real clip, has told us those scores did not come from the clip. Refusals and
hedges are themselves informative and are recorded verbatim, not retried.

Every output records `visual_condition`, and outputs land in their own model_outputs_dir
namespace (`--out-subdir`, default `visual_null`) so they can never be pooled with the main 220
runs.

Degraded clips are generated once into `--cache-dir` (default data/clips_visual_null/) and reused,
so re-running is cheap. They are derived from the SUBTITLE-BLURRED model-input clips, so the
caption de-identification carries through.

`--visual` and `--mode` both accept comma-separated lists, and the model is loaded ONCE for all
of them. On this setup the checkpoint lives on spinning storage and takes ~30 minutes to load
cold, which would otherwise dominate the run: five conditions as five invocations is five loads.

Usage:
    python scripts/19_run_visual_null_baselines.py --visual none --mode grounded
    python scripts/19_run_visual_null_baselines.py --visual none,black,static,shuffled --mode grounded
    python scripts/19_run_visual_null_baselines.py --visual shuffled --mode grounded --channel face
    python scripts/19_run_visual_null_baselines.py --visual black --mode grounded --dry-run
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import (load_config, get_logger, ensure_dir, REPO_ROOT, MODE_CUES,
                        CUE_ABLATION_MODES, parse_manifest_event, kind_clip_bounds)
from lib.prompt_utils import load_prompt_template, render_blocks, substitute_prompt_fields
from lib.schema_utils import extract_json_object, validate_output, load_schema
from lib.video_io import make_black_clip, make_static_clip, make_shuffled_clip

log = get_logger("run_visual_null")

MODES = ["grounded"] + CUE_ABLATION_MODES
VISUAL_CONDITIONS = ["real", "none", "black", "static", "shuffled"]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]

PROMPT_FILE = "prompts/prompt_c_grounded_attribution.md"
SCHEMA_FILE = "prompts/prompt_c_output.schema.json"
DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"

# Matches 07b_run_inference_events.py's `combined` clip kind, which is the kind the paper's
# results are computed on. Kept identical so a visual-null row differs from a main-dataset row in
# exactly one respect.
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

DEGRADERS = {"black": make_black_clip, "static": make_static_clip, "shuffled": make_shuffled_clip}


def degraded_clip(src, condition, cache_dir, overwrite=False):
    """Path to `src` with `condition` applied, generating and caching it on first use."""
    if condition == "real":
        return src
    dst = Path(cache_dir) / condition / src.name
    if dst.exists() and not overwrite:
        return dst
    log.info("generating %s clip: %s", condition, dst.name)
    return DEGRADERS[condition](src, dst)


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
    parser.add_argument("--visual", required=True,
                        help="one or more of " + ", ".join(VISUAL_CONDITIONS) +
                             " (comma-separated; all share one model load)")
    parser.add_argument("--mode", default="grounded",
                        help="one or more of " + ", ".join(MODES) + " (comma-separated)")
    parser.add_argument("--channel", choices=CHANNELS, default=None,
                        help="restrict to events whose manifest target_channel is this (e.g. "
                             "--channel face to run only the 15 face targets)")
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--pad-sec", type=float, default=0.3,
                        help="MUST match what 03b_cut_event_clips.py used (default 0.3)")
    parser.add_argument("--out-subdir", default="visual_null",
                        help="subdirectory of model_outputs_dir for these runs (default: "
                             "visual_null) -- keeps them out of the main 220-run set")
    parser.add_argument("--cache-dir", default=None,
                        help="where generated degraded clips live (default: "
                             "data/clips_visual_null)")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--regenerate-clips", action="store_true",
                        help="rebuild cached degraded clips even if present")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    visuals = [v.strip() for v in args.visual.split(",") if v.strip()]
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    for value, allowed, name in ((visuals, VISUAL_CONDITIONS, "--visual"),
                                 (modes, MODES, "--mode")):
        bad = [v for v in value if v not in allowed]
        if bad:
            parser.error(f"{name}: {bad} not in {allowed}")

    cfg = load_config(args.config)
    c_clips_dir = cfg["paths"]["c_clips_dir"]
    out_dir = ensure_dir(cfg["paths"]["model_outputs_dir"] / args.out_subdir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (REPO_ROOT / "data" / "clips_visual_null")

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
        start_raw, end_raw = (row.get("clip_start_sec") or "").strip(), (row.get("clip_end_sec") or "").strip()
        if not start_raw or not end_raw:
            continue
        clip_start_sec, clip_end_sec = float(start_raw), float(end_raw)
        translation = (row.get("translation") or "").strip()

        for index in (1, 2):
            parsed = parse_manifest_event(row, index)
            if parsed is None:
                continue
            ev_start, ev_end, chan_code = parsed
            if args.channel and chan_code != args.channel:
                continue
            event_id = f"event{index}"
            event_kind = EVENT_CLIP_KIND[event_id]
            kind_start, _ = kind_clip_bounds(event_kind, clip_start_sec, clip_end_sec,
                                             ev_start, ev_end, args.pad_sec)
            for visual in visuals:
                for mode in modes:
                    for run_idx in range(1, n_runs + 1):
                        out_path = out_dir / (f"{clip_id}__{event_id}__combined__{visual}__"
                                              f"run{run_idx}__{mode}.json")
                        if out_path.exists() and not args.overwrite:
                            continue
                        planned.append({
                            "clip_id": clip_id, "event_id": event_id, "run_idx": run_idx,
                            "visual": visual, "mode": mode,
                            "out_path": out_path, "translation": translation,
                            "chan_code": chan_code, "target_channel": chan_code,
                            "rel_start": ev_start - kind_start, "rel_end": ev_end - kind_start,
                            "target_clip": c_clips_dir / f"{clip_id}_{event_kind}_model.mp4",
                            "context_clip": c_clips_dir / f"{clip_id}_window_model.mp4",
                            "video_id": row.get("video_id", ""),
                        })

    # Cheapest conditions first, and `none` before anything needing generated clips, so a run that
    # is interrupted has still produced whole conditions rather than a slice of each.
    planned.sort(key=lambda j: (VISUAL_CONDITIONS.index(j["visual"]), modes.index(j["mode"]),
                                j["clip_id"], j["event_id"]))
    log.info("Planned %d call(s) across visual=%s, mode=%s, backend=%s (one model load)",
             len(planned), visuals, modes, backend_name)

    if args.dry_run:
        for job in planned:
            label = f"{job['clip_id']}/{job['event_id']}"
            try:
                fill_prompt(prompt_text, job["mode"], label, job["translation"],
                            job["chan_code"], job["rel_start"], job["rel_end"])
                status = "ok"
            except ValueError as exc:
                status = f"WOULD FAIL: {exc}"
            video = "(no video)" if job["visual"] == "none" \
                else f"[{job['visual']}] {job['target_clip'].name}"
            print(f"[dry-run] {label} {video} target=[{job['rel_start']:.1f},"
                  f"{job['rel_end']:.1f}]s -> {job['out_path'].name}  [{status}]")
        return

    from lib.model_backends import get_backend, build_backend_kwargs
    backend_kwargs = build_backend_kwargs(backend_name, cfg["model"])
    backend = get_backend(backend_name, **backend_kwargs)

    manifest_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "script": "19_run_visual_null_baselines.py",
        "visual_conditions": visuals,
        "prompt_identical_to_main_run": True,
        "backend": backend_name,
        "model_name": backend.model_name,
        "model_version": backend.model_version,
        "prompt_modes": modes,
        "prompt_file": PROMPT_FILE,
        "cues": {m: MODE_CUES[m] for m in modes},
        "clip_kind": "combined",
        "channel_filter": args.channel,
        "pad_sec": args.pad_sec,
        # The full resolved constructor kwargs, so the manifest records what was actually
        # applied rather than a hand-picked subset that silently omits a new backend's knobs.
        "backend_kwargs": backend_kwargs,
        "fps": backend_kwargs.get("fps"),
        "max_pixels": backend_kwargs.get("max_pixels"),
        "temperature": cfg["model"]["temperature"],
        "do_sample": cfg["model"]["do_sample"],
        "max_new_tokens": cfg["model"]["max_new_tokens"],
        "n_planned": len(planned),
    }
    mpath = out_dir / "run_manifest_visual_null.json"
    history = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else []
    history.append(manifest_entry)
    mpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    for done, job in enumerate(planned, start=1):
        label = f"{job['clip_id']}/{job['event_id']}"
        visual, mode = job["visual"], job["mode"]
        if visual == "none":
            target_path, context_path = None, None
        else:
            if not job["target_clip"].exists() or not job["context_clip"].exists():
                log.warning("%s: missing source clip(s), skipping", label)
                continue
            target_path = degraded_clip(job["target_clip"], visual, cache_dir,
                                        args.regenerate_clips)
            context_path = degraded_clip(job["context_clip"], visual, cache_dir,
                                         args.regenerate_clips)

        log.info("[%d/%d] %s visual=%s mode=%s run %d/%d ...", done, len(planned), label,
                 visual, mode, job["run_idx"], n_runs)
        row_prompt = fill_prompt(prompt_text, mode, label, job["translation"],
                                 job["chan_code"], job["rel_start"], job["rel_end"])
        generate_kwargs = {
            "context_video_path": context_path,
            "context_text": COMBINED_CONTEXT_TEXT if context_path is not None else None,
        }
        if context_path is not None:
            generate_kwargs["video_label"] = COMBINED_VIDEO_LABEL
            generate_kwargs["context_video_label"] = COMBINED_CONTEXT_LABEL

        raw_text = backend.generate(
            video_path=target_path,
            prompt_text=row_prompt,
            temperature=cfg["model"]["temperature"],
            do_sample=cfg["model"]["do_sample"],
            max_new_tokens=cfg["model"]["max_new_tokens"],
            **generate_kwargs,
        )

        parsed, schema_valid, schema_errors = None, False, ["not parsed"]
        try:
            parsed = extract_json_object(raw_text)
            schema_valid, schema_errors = validate_output(parsed, schema)
        except (ValueError, json.JSONDecodeError) as exc:
            schema_errors = [f"JSON parse error: {exc}"]
        if not schema_valid:
            # Under `none` especially, a refusal or a hedge is a RESULT. Log and keep going;
            # never retry or reword, which would make the condition non-comparable.
            log.warning("%s [%s]: schema validation failed: %s", label, visual, schema_errors)

        job["out_path"].write_text(json.dumps({
            "clip_id": job["clip_id"],
            "event_id": job["event_id"],
            "clip_kind": "combined",
            "visual_condition": visual,
            "target_channel": job["target_channel"],
            "video_path": str(target_path) if target_path else None,
            "context_clip_path": str(context_path) if context_path else None,
            "video_id": job["video_id"],
            "mode": mode,
            "prompt_file": PROMPT_FILE,
            "prompt_identical_to_main_run": True,
            "cues": MODE_CUES[mode],
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
