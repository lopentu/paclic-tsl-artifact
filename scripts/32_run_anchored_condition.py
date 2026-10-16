#!/usr/bin/env python3
"""Rubric anchoring: is the model's compressed D1 a judgment, or an unanchored scale?

Greedy decoding never emits D1 = 0 or D1 = 3 anywhere in the 220-run set: every rated action lands
on 1 or 2. The sampling sweep (28/30) sharpens rather than settles this. At T = 1.0 the endpoints
DO appear -- D1 = 0 on face targets in `grounded`, D1 = 3 on manual and face targets in
`ablate_channel` -- so they are present in the distribution and merely off its mode. That rules
out "the model cannot produce a 0 or a 3" and leaves two readings:

    (a) COMPRESSED JUDGMENT. The model's actual view is that these 55 actions really do all sit in
        the middle of the scale. The endpoints are rare because the items are not extreme.
    (b) UNANCHORED SCALE. `0 = no clear contribution` and `3 = essential` are labels with no worked
        referent, so the model treats them as rhetorical extremes and stays in the safe middle
        regardless of the clip.

These come apart if the endpoints are given content. This condition re-runs the fully cued
`grounded` cell with two worked illustrations of what a 0 and a 3 look like inserted into the D1
rubric -- and nothing else changed.

**No instruction to use the range.** The prompt does NOT say "use the full scale", "do not avoid
the endpoints", or anything equivalent. That was the tempting version and it would have been
worthless: it tests whether the model complies with an instruction, not whether the scale was
unanchored. The anchors are descriptive only, so a shift in the distribution is attributable to
the endpoints having acquired meaning rather than to a demand characteristic.

**D2 is deliberately left unanchored.** It shares the same 0-3 scale and the same compression, and
it sits in the same prompt, so it acts as a within-prompt control: if D2's distribution moves as
much as D1's, what changed is the prompt's length or tone rather than D1's calibration. Only D1's
rubric receives the two illustrations; D2's, and the semiotic modes', are byte-identical to the
main run.

The anchored prompt file is a separate artifact rather than a conditional block in the shared one,
for two reasons: `load_prompt_template` requires the shared file to contain exactly the four cue
blocks, and the 220 completed runs should keep a prompt file that still matches what ran. The
runner asserts at startup that the anchored file differs from the base by the inserted anchor
alone, so the two cannot drift apart unnoticed.

Outputs land in their own namespace (`--out-subdir`, default `anchored`).

Usage:
    python scripts/32_run_anchored_condition.py --dry-run
    python scripts/32_run_anchored_condition.py
"""

import argparse
import csv
import datetime
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import (load_config, get_logger, ensure_dir, REPO_ROOT, MODE_CUES,
                        parse_manifest_event, kind_clip_bounds)
from lib.prompt_utils import load_prompt_template, render_blocks, substitute_prompt_fields
from lib.schema_utils import extract_json_object, validate_output, load_schema

log = get_logger("run_anchored")

# `grounded` only: the question is about the scale, and mixing in cue ablations would confound a
# calibration change with a cue effect.
MODES = ["grounded"]

BASE_PROMPT_FILE = "prompts/prompt_c_grounded_attribution.md"
PROMPT_FILE = "prompts/prompt_c_grounded_attribution_anchored.md"
SCHEMA_FILE = "prompts/prompt_c_output.schema.json"
DEFAULT_MANIFEST = REPO_ROOT / "clip_manifest.csv"

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


def diff_against_base(base_text, anchored_text):
    """The lines the anchored prompt adds. Raises if it removes or rewrites anything.

    An anchoring condition whose prompt differs from the main run in some second, unnoticed way
    would not be an anchoring condition at all, so this is checked rather than assumed.
    """
    removed, added = [], []
    for line in difflib.ndiff(base_text.splitlines(), anchored_text.splitlines()):
        if line.startswith("- "):
            removed.append(line[2:])
        elif line.startswith("+ "):
            added.append(line[2:])
    if removed:
        raise SystemExit(f"{PROMPT_FILE} REMOVES or rewrites {len(removed)} line(s) from "
                         f"{BASE_PROMPT_FILE}; it must only add. First: {removed[0]!r}")
    if not added:
        raise SystemExit(f"{PROMPT_FILE} is identical to {BASE_PROMPT_FILE} -- nothing anchored.")
    return added


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
    parser.add_argument("--mode", default="grounded", choices=MODES)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--pad-sec", type=float, default=0.3,
                        help="MUST match what 03b_cut_event_clips.py used (default 0.3)")
    parser.add_argument("--out-subdir", default="anchored",
                        help="subdirectory of model_outputs_dir (default: anchored)")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    c_clips_dir = cfg["paths"]["c_clips_dir"]
    out_dir = ensure_dir(cfg["paths"]["model_outputs_dir"] / args.out_subdir)
    backend_name = args.backend or cfg["model"]["backend"]
    n_runs = args.n_runs or cfg["model"]["n_runs_per_item"]

    scripts_dir = Path(__file__).resolve().parent
    base_text = load_prompt_template(scripts_dir / "config" / BASE_PROMPT_FILE)
    prompt_text = load_prompt_template(scripts_dir / "config" / PROMPT_FILE)
    added = diff_against_base(base_text, prompt_text)
    log.info("Anchored prompt adds %d line(s) to the main run's prompt and removes none.",
             len(added))

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
            event_id = f"event{index}"
            event_kind = EVENT_CLIP_KIND[event_id]
            kind_start, _ = kind_clip_bounds(event_kind, clip_start_sec, clip_end_sec,
                                             ev_start, ev_end, args.pad_sec)
            for run_idx in range(1, n_runs + 1):
                out_path = out_dir / (f"{clip_id}__{event_id}__combined__run{run_idx}__"
                                      f"anchored_{args.mode}.json")
                if out_path.exists() and not args.overwrite:
                    continue
                planned.append({
                    "clip_id": clip_id, "event_id": event_id, "run_idx": run_idx,
                    "out_path": out_path, "translation": translation,
                    "true_channel": true_channel,
                    "rel_start": ev_start - kind_start, "rel_end": ev_end - kind_start,
                    "target_clip": c_clips_dir / f"{clip_id}_{event_kind}_model.mp4",
                    "context_clip": c_clips_dir / f"{clip_id}_window_model.mp4",
                    "video_id": row.get("video_id", ""),
                })

    planned.sort(key=lambda j: (j["clip_id"], j["event_id"]))
    log.info("Planned %d call(s) mode=%s, backend=%s (one model load)",
             len(planned), args.mode, backend_name)

    if args.dry_run:
        for job in planned:
            label = f"{job['clip_id']}/{job['event_id']}"
            try:
                fill_prompt(prompt_text, args.mode, label, job["translation"],
                            job["true_channel"], job["rel_start"], job["rel_end"])
                status = "ok"
            except ValueError as exc:
                status = f"WOULD FAIL: {exc}"
            print(f"[dry-run] {label} ({job['true_channel']}) "
                  f"target=[{job['rel_start']:.1f},{job['rel_end']:.1f}]s "
                  f"-> {job['out_path'].name}  [{status}]")
        print("\n--- lines the anchored prompt adds ---")
        for line in added:
            print(f"  + {line}")
        return

    from lib.model_backends import get_backend, build_backend_kwargs
    backend_kwargs = build_backend_kwargs(backend_name, cfg["model"])
    backend = get_backend(backend_name, **backend_kwargs)

    manifest_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "script": "32_run_anchored_condition.py",
        "condition": "anchored",
        "anchored_dimension": "semantic_syntactic_load (D1) only",
        "unanchored_control_dimensions": ["pragmatic_depictive_affective_enrichment (D2)",
                                          "description", "indication", "depiction"],
        "prompt_contains_instruction_to_use_full_range": False,
        "prompt_diff_from_main_run_added_lines": added,
        "prompt_diff_from_main_run_removed_lines": [],
        "backend": backend_name,
        "model_name": backend.model_name,
        "model_version": backend.model_version,
        "prompt_modes": [args.mode],
        "prompt_file": PROMPT_FILE,
        "base_prompt_file": BASE_PROMPT_FILE,
        "clip_kind": "combined",
        "pad_sec": args.pad_sec,
        "backend_kwargs": backend_kwargs,
        "temperature": cfg["model"]["temperature"],
        "do_sample": cfg["model"]["do_sample"],
        "max_new_tokens": cfg["model"]["max_new_tokens"],
        "n_planned": len(planned),
    }
    mpath = out_dir / "run_manifest_anchored.json"
    history = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else []
    history.append(manifest_entry)
    mpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    for done, job in enumerate(planned, start=1):
        label = f"{job['clip_id']}/{job['event_id']}"
        if not job["target_clip"].exists() or not job["context_clip"].exists():
            log.warning("%s: missing source clip(s), skipping", label)
            continue

        log.info("[%d/%d] %s (%s) ...", done, len(planned), label, job["true_channel"])
        row_prompt = fill_prompt(prompt_text, args.mode, label, job["translation"],
                                 job["true_channel"], job["rel_start"], job["rel_end"])
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
            log.warning("%s [anchored]: schema validation failed: %s", label, schema_errors)

        job["out_path"].write_text(json.dumps({
            "clip_id": job["clip_id"],
            "event_id": job["event_id"],
            "clip_kind": "combined",
            "condition": "anchored",
            "anchored_dimension": "semantic_syntactic_load",
            "target_channel": job["true_channel"],
            "video_path": str(job["target_clip"]),
            "context_clip_path": str(job["context_clip"]),
            "video_id": job["video_id"],
            "mode": args.mode,
            "prompt_file": PROMPT_FILE,
            "base_prompt_file": BASE_PROMPT_FILE,
            "cues": MODE_CUES[args.mode],
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
