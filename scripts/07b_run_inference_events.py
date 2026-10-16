#!/usr/bin/env python3
"""Run Prompt C over the C01-C40 clip set (clip_manifest.csv / data/clips_c/).

Parallel driver to 07_run_inference.py for the C01-C40 workflow: same prompt file, same cue-mode
rendering (lib/prompt_utils, lib/common.MODE_CUES), same output schema -- but driven by
clip_manifest.csv instead of data/windows/manifest.csv, and run separately against EACH clip kind
cut by 03b_cut_event_clips.py, plus one synthetic kind that resends two of those files together:

    window    the full analysis window (<clip_id>_window_model.mp4)
    ev1/ev2   the tight, padded event clip (<clip_id>_ev1_model.mp4 / _ev2_model.mp4)
    combined  (2026-07-29) BOTH of the above in one call: window as context_video_path, the
              event's own ev1/ev2 file as video_path. No new clip is cut for this -- it just
              feeds the two already-cut files to the model together instead of separately.

These are genuinely different conditions, not two ways of running the same test: on an event clip
the padding itself is a deliberately weak, symmetric localization cue that the window clip does
not have. Every output filename carries its own clip_kind so the two are
never accidentally pooled -- see 08b_score_localization_events.py for what a "localization
success" means differently on each.

`combined` exists alongside `window`/`ev1`/`ev2`, not instead of them: it was added because an
annotator asked whether the model could be shown the target clip directly rather than having to
find it from a text timestamp inside undifferentiated footage . It gives the
model both the temporal-scope evidence D1 needs (visible in `window`) and an unambiguous view of
which action to score (visible in `ev1`/`ev2`) in the same call, at the cost of no longer being
comparable to the two solo conditions' localization test -- read `combined`'s Tier 1 numbers as
their own thing in 08b, never averaged with `window` or `ev1`/`ev2`.

Per event, under every applicable clip kind, the model is given the SAME target-span text
(__EVENT_START_SEC__ / __EVENT_END_SEC__) computed relative to whichever clip file is actually fed
to it, using lib.common.padded_event_bounds -- the exact same arithmetic
03b_cut_event_clips.py used to decide what to cut. --pad-sec here MUST match whatever value was
passed to 03b for these clips (both default to 0.3); as a safety net, the actual cut clip's
ffprobe'd duration is checked against the expected duration and this refuses to run on a mismatch
rather than silently handing the model a timestamp that does not correspond to what it is seeing.

event1 is fed to the model via the "window" and "ev1" clips; event2 (if present) via "window" and
"ev2". A clip with two events therefore produces two window-clip runs (one per event's target
span) sharing the same underlying video file -- this is intentional, exactly analogous to how the
older workflow's local/context conditions share a video but target one event each.

Modes: the same four cue configurations as 07_run_inference.py (lib.common.INFERENCE_MODES) MINUS
grounded_distractor, which needs distractor_start_sec/distractor_end_sec columns that
clip_manifest.csv does not have (deferred, same status as in the older workflow).

Outputs land in the SAME model_outputs_dir as 07_run_inference.py (namespaced by --config /
model_tag as usual), disambiguated by filename alone:
    <clip_id>__<event_id>__<clip_kind>__run<N>__<mode>.json
which can never collide with the older workflow's <window_id>__run<N>__<mode>.json naming.

Usage:
    python scripts/07b_run_inference_events.py --mode grounded
    python scripts/07b_run_inference_events.py --mode grounded --clip-kind combined
    python scripts/07b_run_inference_events.py --mode ablate_both --clip-id C01 --dry-run
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
from lib.video_io import ffprobe_duration

log = get_logger("run_inference_events")

# grounded_distractor is excluded: clip_manifest.csv has no distractor_start_sec/end_sec columns.
MODES = ["grounded"] + CUE_ABLATION_MODES
CLIP_KINDS = ["window", "ev1", "ev2", "combined"]
# Which clip kinds carry which event's target span.
KINDS_FOR_EVENT = {"event1": ["window", "ev1", "combined"], "event2": ["window", "ev2", "combined"]}
# "combined" has no cut file of its own -- it resends the window file as context alongside
# whichever event file (ev1/ev2) actually carries this event's target span.
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

PROMPT_FILE = "prompts/prompt_c_grounded_attribution.md"
SCHEMA_FILE = "prompts/prompt_c_output.schema.json"

DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"
DURATION_TOLERANCE_SEC = 0.2  # ffmpeg re-encode rounding; beyond this, --pad-sec likely mismatches 03b


def load_prompt(prompt_file=None):
    """Load the prompt template. `prompt_file` overrides the default PROMPT_FILE -- used for
    prompt-robustness checks that must NOT touch the main prompt (see --prompt-file)."""
    scripts_dir = Path(__file__).resolve().parent
    return load_prompt_template(Path(prompt_file) if prompt_file
                                else scripts_dir / "config" / PROMPT_FILE)


def iter_events(row):
    """clip_id row -> list of (event_id, start, end, channel), skipping an N/A event2."""
    out = []
    for index in (1, 2):
        ev = parse_manifest_event(row, index)
        if ev is not None:
            start, end, channel = ev
            out.append((f"event{index}", start, end, channel))
    return out


def fill_prompt(prompt_text, mode, item_label, translation, chan_code, rel_start, rel_end):
    cues = MODE_CUES[mode]
    keep = {"MEANING_GIVEN" if cues["translation"] else "MEANING_WITHHELD",
            "CHANNEL_GIVEN" if cues["channel"] else "CHANNEL_WITHHELD"}
    text = render_blocks(prompt_text, keep)
    return substitute_prompt_fields(
        text, item_label=item_label, mode=mode,
        translation=translation if cues["translation"] else None,
        channel_code=chan_code if cues["channel"] else None,
        rel_start=rel_start, rel_end=rel_end,
    )


def append_run_manifest(model_outputs_dir, entry):
    manifest_path = model_outputs_dir / "run_manifest_events.json"
    history = []
    if manifest_path.exists():
        history = json.loads(manifest_path.read_text(encoding="utf-8"))
    history.append(entry)
    manifest_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None, help=f"default: {DEFAULT_MANIFEST.name}")
    parser.add_argument("--clip-id", default=None, help="run only this clip_id (default: all)")
    parser.add_argument("--clip-kind", choices=CLIP_KINDS, default=None,
                        help="run only this clip kind (default: all kinds that apply per event)")
    parser.add_argument("--mode", choices=MODES, default="grounded")
    parser.add_argument("--pad-sec", type=float, default=0.3,
                        help="MUST match the --pad-sec used to cut these clips with "
                             "03b_cut_event_clips.py (default: 0.3, matching that script's "
                             "default). Only affects ev1/ev2 timestamp math, not window clips.")
    parser.add_argument("--backend", default=None, help="override model.backend from config")
    parser.add_argument("--prompt-file", default=None,
                        help=f"override the prompt template (default: config/{PROMPT_FILE}). For "
                             "prompt-robustness checks ONLY -- pair it with a --config whose "
                             "model_outputs_dir/results_dir are a separate tag, so variant-prompt "
                             "runs can never be pooled with the main dataset.")
    parser.add_argument("--n-runs", type=int, default=None, help="override model.n_runs_per_item")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="skip model loading/inference, just list planned calls")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
    c_clips_dir = cfg["paths"]["c_clips_dir"]
    out_dir = ensure_dir(cfg["paths"]["model_outputs_dir"])

    backend_name = args.backend or cfg["model"]["backend"]
    n_runs = args.n_runs or cfg["model"]["n_runs_per_item"]
    prompt_text = load_prompt(args.prompt_file)
    prompt_file_used = args.prompt_file or PROMPT_FILE
    if args.prompt_file:
        log.warning("Using OVERRIDE prompt file %s (not the default %s). Every output will record "
                    "prompt_file so these runs can never be mistaken for main-dataset runs.",
                    args.prompt_file, PROMPT_FILE)

    scripts_dir = Path(__file__).resolve().parent
    schema = load_schema(scripts_dir / "config" / SCHEMA_FILE)

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.clip_id:
        rows = [r for r in rows if (r.get("clip_id") or "").strip() == args.clip_id]

    planned, problems = [], []
    for row in rows:
        clip_id = (row.get("clip_id") or "").strip()
        if (row.get("status") or "").strip().lower() == "excluded":
            continue
        clip_start_raw = (row.get("clip_start_sec") or "").strip()
        clip_end_raw = (row.get("clip_end_sec") or "").strip()
        if not clip_start_raw or not clip_end_raw:
            continue  # manifest row not filled in yet

        try:
            events = iter_events(row)
        except ValueError as exc:
            problems.append(f"{clip_id}: {exc}")
            continue
        if not events:
            problems.append(f"{clip_id}: event1 is required but missing/N-A")
            continue

        clip_start_sec, clip_end_sec = float(clip_start_raw), float(clip_end_raw)
        translation = (row.get("translation") or "").strip()

        for event_id, ev_start, ev_end, chan_code in events:
            for kind in KINDS_FOR_EVENT[event_id]:
                if args.clip_kind and kind != args.clip_kind:
                    continue

                context_clip_path, expected_context_duration = None, None
                if kind == "combined":
                    event_kind = EVENT_CLIP_KIND[event_id]
                    clip_path = c_clips_dir / f"{clip_id}_{event_kind}_model.mp4"
                    context_clip_path = c_clips_dir / f"{clip_id}_window_model.mp4"
                    kind_clip_start, kind_clip_end = kind_clip_bounds(
                        event_kind, clip_start_sec, clip_end_sec, ev_start, ev_end, args.pad_sec)
                    ctx_start, ctx_end = kind_clip_bounds(
                        "window", clip_start_sec, clip_end_sec, ev_start, ev_end, args.pad_sec)
                    expected_context_duration = ctx_end - ctx_start
                else:
                    clip_path = c_clips_dir / f"{clip_id}_{kind}_model.mp4"
                    kind_clip_start, kind_clip_end = kind_clip_bounds(
                        kind, clip_start_sec, clip_end_sec, ev_start, ev_end, args.pad_sec)

                rel_start = ev_start - kind_clip_start
                rel_end = ev_end - kind_clip_start

                for run_idx in range(1, n_runs + 1):
                    out_path = out_dir / f"{clip_id}__{event_id}__{kind}__run{run_idx}__{args.mode}.json"
                    if out_path.exists() and not args.overwrite:
                        continue
                    planned.append({
                        "clip_id": clip_id, "event_id": event_id, "kind": kind,
                        "clip_path": clip_path, "context_clip_path": context_clip_path,
                        "run_idx": run_idx, "out_path": out_path,
                        "translation": translation, "chan_code": chan_code,
                        "rel_start": rel_start, "rel_end": rel_end,
                        "expected_duration": kind_clip_end - kind_clip_start,
                        "expected_context_duration": expected_context_duration,
                        "video_id": row.get("video_id", ""),
                    })

    for msg in problems:
        log.warning(msg)

    log.info("Planned %d inference call(s) across %d clip row(s), backend=%s, mode=%s",
              len(planned), len(rows), backend_name, args.mode)

    def item_label(job):
        return f"{job['clip_id']}/{job['event_id']}/{job['kind']}"

    if args.dry_run:
        for job in planned:
            try:
                fill_prompt(prompt_text, args.mode, item_label(job), job["translation"],
                           job["chan_code"], job["rel_start"], job["rel_end"])
                status = "ok"
            except ValueError as exc:
                status = f"WOULD FAIL: {exc}"
            ctx = f" +context={job['context_clip_path']}" if job["context_clip_path"] else ""
            print(f"[dry-run] {item_label(job)} run{job['run_idx']} clip={job['clip_path']}{ctx} "
                  f"target=[{job['rel_start']:.1f},{job['rel_end']:.1f}]s -> {job['out_path']}  [{status}]")
        return

    from lib.model_backends import get_backend, build_backend_kwargs
    backend_kwargs = build_backend_kwargs(backend_name, cfg["model"])
    backend = get_backend(backend_name, **backend_kwargs)

    append_run_manifest(out_dir, {
        "timestamp": datetime.datetime.now().isoformat(),
        "backend": backend_name,
        "model_name": backend.model_name,
        "model_version": backend.model_version,
        "prompt_mode": args.mode,
        "prompt_file": prompt_file_used,
        "cues": MODE_CUES[args.mode],
        "pad_sec": args.pad_sec,
        # The full resolved constructor kwargs, so the manifest records what was actually
        # applied rather than a hand-picked subset that silently omits a new backend's knobs.
        "backend_kwargs": backend_kwargs,
        "fps": backend_kwargs.get("fps"),
        "max_pixels": backend_kwargs.get("max_pixels"),
        "quantization": cfg["model"].get("quantization"),
        "temperature": cfg["model"]["temperature"],
        "do_sample": cfg["model"]["do_sample"],
        "max_new_tokens": cfg["model"]["max_new_tokens"],
        "n_runs_per_item": n_runs,
        "n_clip_rows": len(rows),
        "video_input_method": "local subtitle-blurred clip file (C01-C40 clip_manifest.csv), per-event/clip_kind",
        "saw_audio": False,
        "subtitles_cropped_or_blurred": True,
        "timestamps_visible_in_frame": False,
    })

    for job in planned:
        if not job["clip_path"].exists():
            log.warning("Missing model-input clip for %s: %s (run 03b/04 --source clip_manifest first)",
                        item_label(job), job["clip_path"])
            continue
        if job["context_clip_path"] is not None and not job["context_clip_path"].exists():
            log.warning("Missing model-input context clip for %s: %s (run 03b/04 --source clip_manifest first)",
                        item_label(job), job["context_clip_path"])
            continue

        actual_duration = ffprobe_duration(job["clip_path"])
        if abs(actual_duration - job["expected_duration"]) > DURATION_TOLERANCE_SEC:
            log.error(
                "%s: clip duration mismatch (actual %.2fs vs expected %.2fs from --pad-sec=%.2f). "
                "This --pad-sec probably does not match what 03b_cut_event_clips.py used to cut "
                "this clip -- refusing to send a timestamp hint that may not match the video. "
                "Re-run with the correct --pad-sec.",
                item_label(job), actual_duration, job["expected_duration"], args.pad_sec,
            )
            continue
        if job["context_clip_path"] is not None:
            actual_context_duration = ffprobe_duration(job["context_clip_path"])
            if abs(actual_context_duration - job["expected_context_duration"]) > DURATION_TOLERANCE_SEC:
                log.error(
                    "%s: context clip duration mismatch (actual %.2fs vs expected %.2fs). The "
                    "manifest's clip_start_sec/clip_end_sec probably changed after this window "
                    "clip was cut -- re-run 03b_cut_event_clips.py for this clip_id first.",
                    item_label(job), actual_context_duration, job["expected_context_duration"],
                )
                continue

        log.info("Running %s run %d/%d ...", item_label(job), job["run_idx"], n_runs)
        row_prompt_text = fill_prompt(prompt_text, args.mode, item_label(job), job["translation"],
                                      job["chan_code"], job["rel_start"], job["rel_end"])
        generate_kwargs = {"context_video_path": None, "context_text": None}
        if job["kind"] == "combined":
            generate_kwargs = {
                "context_video_path": job["context_clip_path"],
                "context_text": COMBINED_CONTEXT_TEXT,
                "video_label": COMBINED_VIDEO_LABEL,
                "context_video_label": COMBINED_CONTEXT_LABEL,
            }
        raw_text = backend.generate(
            video_path=job["clip_path"],
            prompt_text=row_prompt_text,
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
            log.warning("%s run %d: schema validation failed: %s", item_label(job), job["run_idx"], schema_errors)

        job["out_path"].write_text(json.dumps({
            "clip_id": job["clip_id"],
            "event_id": job["event_id"],
            "clip_kind": job["kind"],
            "context_clip_path": str(job["context_clip_path"]) if job["context_clip_path"] else None,
            "video_id": job["video_id"],
            "mode": args.mode,
            "prompt_file": prompt_file_used,
            "cues": MODE_CUES[args.mode],
            "pad_sec": args.pad_sec,
            "run_index": job["run_idx"],
            "backend": backend_name,
            "model_name": backend.model_name,
            "model_version": backend.model_version,
            "timestamp": datetime.datetime.now().isoformat(),
            # Frames/tokens actually fed, for backends that report them (the cross-model ones).
            "backend_call_info": getattr(backend, "last_call_info", None),
            "raw_text": raw_text,
            "parsed": parsed,
            "schema_valid": schema_valid,
            "schema_errors": schema_errors,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
