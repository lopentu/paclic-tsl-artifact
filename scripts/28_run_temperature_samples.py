#!/usr/bin/env python3
"""Temperature sampling: is "articulator-determined" a property of the model, or of greedy decoding?

Every rated result in the study is one greedy sample per item, so phrases like "D1 is 100%
articulator-determined in the gated set" describe the outputs we observed, not a distribution
over the outputs the model could produce. This draws k samples per item in the two cells the
claim is made about -- `grounded` (where D1 is claimed fully articulator-determined) and
`ablate_channel` (where the same items flip to the self-selected articulator's value) -- so the
claim can be restated over a distribution: the modal D1's share, and the entropy of the sampled
D1 distribution, per item and per articulator.

**Temperature 1.0 by default, not the instruct model's recommended 0.7.** This is the harder
test: at T = 1.0 we sample the model's own distribution undamped, so if the modal D1 still
dominates the samples, the determinism is a property of the distribution rather than an artifact
of the decoding rule. A result at 0.7 would be weaker evidence for the same claim. Pass
`--temperature 0.7` to run the deployment-realistic setting instead; the two are written to
different subdirectories and must not be pooled.

Sampling is seeded per (clip, event, mode, sample index), so a re-run reproduces the same draws
and an interrupted run resumes without re-drawing what it already has. Existing outputs are
skipped, so this is safe to stop and restart -- which it is designed for, since k=5 over two
cells is 550 calls.

Outputs land in their own namespace (`--out-subdir`, default `temperature`) and can never pool
with the main 220 greedy runs.

Usage:
    python scripts/28_run_temperature_samples.py --dry-run
    python scripts/28_run_temperature_samples.py --k 5
    python scripts/28_run_temperature_samples.py --k 5 --mode grounded
"""

import argparse
import csv
import datetime
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import (load_config, get_logger, ensure_dir, REPO_ROOT, MODE_CUES,
                        parse_manifest_event, kind_clip_bounds)
from lib.prompt_utils import load_prompt_template, render_blocks, substitute_prompt_fields
from lib.schema_utils import extract_json_object, validate_output, load_schema

log = get_logger("run_temperature")

# The two cells the "articulator-determined" claim is made about.
MODES = ["grounded", "ablate_channel"]
CHANNELS = ["manual", "face", "non_dominant_hand", "head_torso"]

PROMPT_FILE = "prompts/prompt_c_grounded_attribution.md"
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


def sample_seed(clip_id, event_id, mode, sample_idx):
    """Deterministic per-draw seed, so the k draws for an item are reproducible and an
    interrupted run resumes with the same sequence rather than a fresh one."""
    key = f"{clip_id}|{event_id}|{mode}|{sample_idx}".encode("utf-8")
    return zlib.crc32(key) & 0x7FFFFFFF


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
    parser.add_argument("--mode", default=",".join(MODES),
                        help="comma-separated: " + ", ".join(MODES) + " (default: both)")
    parser.add_argument("--channel", choices=CHANNELS, default=None)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--k", type=int, default=5, help="samples per item per cell (default 5)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="sampling temperature (default 1.0; see module docstring for why "
                             "not the instruct model's recommended 0.7)")
    parser.add_argument("--pad-sec", type=float, default=0.3)
    parser.add_argument("--out-subdir", default=None,
                        help="default: temperature_t<temperature>, so runs at different "
                             "temperatures can never be pooled")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        parser.error(f"--mode: {bad} not in {MODES}")

    cfg = load_config(args.config)
    c_clips_dir = cfg["paths"]["c_clips_dir"]
    subdir = args.out_subdir or f"temperature_t{args.temperature:g}"
    out_dir = ensure_dir(cfg["paths"]["model_outputs_dir"] / subdir)
    backend_name = args.backend or cfg["model"]["backend"]

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
            parsed_ev = parse_manifest_event(row, index)
            if parsed_ev is None:
                continue
            ev_start, ev_end, channel = parsed_ev
            if args.channel and channel != args.channel:
                continue
            event_id = f"event{index}"
            event_kind = EVENT_CLIP_KIND[event_id]
            kind_start, _ = kind_clip_bounds(event_kind, clip_start_sec, clip_end_sec,
                                             ev_start, ev_end, args.pad_sec)
            for mode in modes:
                for sample_idx in range(1, args.k + 1):
                    out_path = out_dir / (f"{clip_id}__{event_id}__combined__sample{sample_idx}__"
                                          f"{mode}.json")
                    if out_path.exists() and not args.overwrite:
                        continue
                    planned.append({
                        "clip_id": clip_id, "event_id": event_id, "sample_idx": sample_idx,
                        "mode": mode, "out_path": out_path, "translation": translation,
                        "target_channel": channel,
                        "rel_start": ev_start - kind_start, "rel_end": ev_end - kind_start,
                        "target_clip": c_clips_dir / f"{clip_id}_{event_kind}_model.mp4",
                        "context_clip": c_clips_dir / f"{clip_id}_window_model.mp4",
                        "video_id": row.get("video_id", ""),
                    })

    # Sample index outermost: an interrupted run then leaves COMPLETE k=1, k=2, ... layers over
    # all items rather than all k draws for the first few items and none for the rest, so a
    # partial run is still analysable as a smaller k.
    planned.sort(key=lambda j: (j["sample_idx"], modes.index(j["mode"]), j["clip_id"],
                                j["event_id"]))
    log.info("Planned %d call(s): k=%d, T=%g, modes=%s -> %s",
             len(planned), args.k, args.temperature, modes, out_dir)

    if args.dry_run:
        for job in planned[:5]:
            print(f"[dry-run] {job['clip_id']}/{job['event_id']} mode={job['mode']} "
                  f"sample={job['sample_idx']} seed="
                  f"{sample_seed(job['clip_id'], job['event_id'], job['mode'], job['sample_idx'])}"
                  f" -> {job['out_path'].name}")
        print(f"[dry-run] ... {len(planned)} total")
        return

    from lib.model_backends import get_backend, build_backend_kwargs
    import torch
    backend_kwargs = build_backend_kwargs(backend_name, cfg["model"])
    backend = get_backend(backend_name, **backend_kwargs)

    manifest_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "script": "28_run_temperature_samples.py",
        "condition": "temperature_sampling",
        "k": args.k,
        "temperature": args.temperature,
        "do_sample": True,
        "seeded_per_draw": True,
        "prompt_identical_to_main_run": True,
        "backend": backend_name,
        "model_name": backend.model_name,
        "model_version": backend.model_version,
        "prompt_modes": modes,
        "prompt_file": PROMPT_FILE,
        "clip_kind": "combined",
        "pad_sec": args.pad_sec,
        "backend_kwargs": backend_kwargs,
        "max_new_tokens": cfg["model"]["max_new_tokens"],
        "n_planned": len(planned),
    }
    mpath = out_dir / "run_manifest_temperature.json"
    history = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else []
    history.append(manifest_entry)
    mpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    for done, job in enumerate(planned, start=1):
        label = f"{job['clip_id']}/{job['event_id']}"
        if not job["target_clip"].exists() or not job["context_clip"].exists():
            log.warning("%s: missing source clip(s), skipping", label)
            continue

        seed = sample_seed(job["clip_id"], job["event_id"], job["mode"], job["sample_idx"])
        torch.manual_seed(seed)
        log.info("[%d/%d] %s mode=%s sample %d/%d (T=%g) ...", done, len(planned), label,
                 job["mode"], job["sample_idx"], args.k, args.temperature)
        row_prompt = fill_prompt(prompt_text, job["mode"], label, job["translation"],
                                 job["target_channel"], job["rel_start"], job["rel_end"])
        raw_text = backend.generate(
            video_path=job["target_clip"],
            prompt_text=row_prompt,
            temperature=args.temperature,
            do_sample=True,
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
            # At T=1.0 an occasional malformed draw is expected and is itself part of the
            # stability picture. Recorded, never retried -- a retry would bias the sample.
            log.warning("%s [sample %d]: schema validation failed: %s", label,
                        job["sample_idx"], schema_errors)

        # Written via a temp file and renamed, because this sweep is designed to be killed
        # mid-run at a GPU handover deadline. A SIGINT landing inside a plain write_text would
        # leave a truncated JSON file that the resume logic would then SKIP (it exists), silently
        # putting a corrupt sample into the analysis. rename() is atomic on the same filesystem,
        # so a file either does not exist or is complete.
        tmp_path = job["out_path"].with_suffix(".json.partial")
        tmp_path.write_text(json.dumps({
            "clip_id": job["clip_id"],
            "event_id": job["event_id"],
            "clip_kind": "combined",
            "condition": "temperature_sampling",
            "target_channel": job["target_channel"],
            "video_id": job["video_id"],
            "mode": job["mode"],
            "cues": MODE_CUES[job["mode"]],
            "sample_index": job["sample_idx"],
            "k": args.k,
            "temperature": args.temperature,
            "do_sample": True,
            "seed": seed,
            "prompt_file": PROMPT_FILE,
            "prompt_identical_to_main_run": True,
            "pad_sec": args.pad_sec,
            "backend": backend_name,
            "model_name": backend.model_name,
            "model_version": backend.model_version,
            "timestamp": datetime.datetime.now().isoformat(),
            "raw_text": raw_text,
            "parsed": parsed,
            "schema_valid": schema_valid,
            "schema_errors": schema_errors,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(job["out_path"])

    log.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
