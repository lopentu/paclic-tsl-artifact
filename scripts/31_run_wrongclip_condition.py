#!/usr/bin/env python3
"""Wrong-clip condition: keep the whole prompt, swap the video for a different one.

The degraded-video ladder (19) already asks whether the functional scores survive DESTROYING the
video -- blacked out, frozen, frame-shuffled. They largely do. That result has a standing reply:
degraded input is out-of-distribution, so the model may be falling back on the prompt precisely
because what it is being shown is not a video of anything. The reply is reasonable and the ladder
cannot answer it.

This condition removes it. The model is shown a REAL, well-formed, in-distribution clip of a
signer signing -- just not the one the prompt is about. Nothing is degraded and nothing looks
unusual. If the functional scores still do not move, the scores were never about this clip.

**Which wrong clip.** Partners are drawn by a derangement WITHIN the same target articulator, so
the articulator label the prompt supplies remains TRUE of the video actually shown. This is the
whole point of the pairing rule: crossing articulators would re-introduce exactly the confound
26_run_miscue_condition.py exists to isolate, and a result would be unattributable between "the
video is wrong" and "the label is wrong". Here the label is right, the video is a real signing
clip, and the only false things are the translation and the identity of the target action.

The derangement is a rotation of the group ordered by clip duration, so it is deterministic (no
seed to record, reproducible from the manifest alone) and pairs each item with a similar-length
clip, which minimises how often the prompt's target span has to be clamped to fit.

    manual              32 items  -> deranged within
    face                15 items  -> deranged within
    non_dominant_hand    7 items  -> deranged within
    head_torso           1 item   -> EXCLUDED: a single item has no within-articulator partner

so 54 of 55, and the exclusion is reported rather than silently absorbed.

**The target span.** The prompt names a span in clip-relative seconds. It is taken from A, like
every other prompt field, and clamped into the partner clip's duration when the partner is
shorter; both the requested and the applied span are recorded, along with a `span_clamped` flag,
so an analysis can check whether clamped items behave differently from unclamped ones.

Outputs land in their own namespace (`--out-subdir`, default `wrongclip`) so they can never be
pooled with the main 220 runs.

Usage:
    python scripts/31_run_wrongclip_condition.py --dry-run
    python scripts/31_run_wrongclip_condition.py
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

log = get_logger("run_wrongclip")

# Only the CHANNEL_GIVEN modes are meaningful here: the pairing rule holds the articulator label
# constant across A and B, which is only a constraint if the label is present at all.
MODES = ["grounded", "ablate_translation"]
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

MIN_SPAN_SEC = 0.1


def fill_prompt(prompt_text, mode, item_label, translation, chan_code, rel_start, rel_end):
    """Byte-identical to 26/07b's prompt construction -- only the video sent alongside differs."""
    cues = MODE_CUES[mode]
    keep = {"MEANING_GIVEN" if cues["translation"] else "MEANING_WITHHELD",
            "CHANNEL_GIVEN" if cues["channel"] else "CHANNEL_WITHHELD"}
    return substitute_prompt_fields(
        render_blocks(prompt_text, keep), item_label=item_label, mode=mode,
        translation=translation if cues["translation"] else None,
        channel_code=chan_code if cues["channel"] else None,
        rel_start=rel_start, rel_end=rel_end,
    )


def clamp_span(rel_start, rel_end, partner_duration):
    """Fit A's target span into B's clip. Returns (start, end, was_clamped)."""
    end = min(rel_end, partner_duration)
    start = min(rel_start, max(0.0, end - MIN_SPAN_SEC))
    start = max(0.0, start)
    if end - start < MIN_SPAN_SEC:
        end = min(partner_duration, start + MIN_SPAN_SEC)
    clamped = abs(start - rel_start) > 1e-6 or abs(end - rel_end) > 1e-6
    return start, end, clamped


def build_events(rows, cfg, pad_sec, clip_filter):
    """Every runnable event, with the bounds and clip paths its own prompt would have used."""
    c_clips_dir = cfg["paths"]["c_clips_dir"]
    events = []
    for row in rows:
        clip_id = (row.get("clip_id") or "").strip()
        if (row.get("status") or "").strip().lower() == "excluded":
            continue
        start_raw = (row.get("clip_start_sec") or "").strip()
        end_raw = (row.get("clip_end_sec") or "").strip()
        if not start_raw or not end_raw:
            continue
        clip_start_sec, clip_end_sec = float(start_raw), float(end_raw)
        for index in (1, 2):
            parsed = parse_manifest_event(row, index)
            if parsed is None:
                continue
            ev_start, ev_end, true_channel = parsed
            event_id = f"event{index}"
            event_kind = EVENT_CLIP_KIND[event_id]
            kind_start, kind_end = kind_clip_bounds(event_kind, clip_start_sec, clip_end_sec,
                                                    ev_start, ev_end, pad_sec)
            events.append({
                "clip_id": clip_id, "event_id": event_id, "true_channel": true_channel,
                "translation": (row.get("translation") or "").strip(),
                "video_id": row.get("video_id", ""),
                "rel_start": ev_start - kind_start, "rel_end": ev_end - kind_start,
                "duration": kind_end - kind_start,
                "target_clip": c_clips_dir / f"{clip_id}_{event_kind}_model.mp4",
                "context_clip": c_clips_dir / f"{clip_id}_window_model.mp4",
            })
    if clip_filter:
        events = [e for e in events if e["clip_id"] == clip_filter]
    return events


def assign_partners(events):
    """Within-articulator derangement by rotating the duration-sorted order.

    A rotation by one has no fixed point for any group of 2 or more, so no item is ever paired
    with itself; ordering by duration first means the pairing is deterministic and each item's
    partner is the next-longest clip of the same articulator, which keeps span clamping rare.
    """
    by_channel = {}
    for ev in events:
        by_channel.setdefault(ev["true_channel"], []).append(ev)

    pairs, dropped = [], []
    for channel, group in sorted(by_channel.items()):
        if len(group) < 2:
            dropped.extend(group)
            log.warning("%s: only %d item(s), no within-articulator partner -- EXCLUDED",
                        channel, len(group))
            continue
        group.sort(key=lambda e: (e["duration"], e["clip_id"], e["event_id"]))
        for i, ev in enumerate(group):
            pairs.append((ev, group[(i + 1) % len(group)]))
    return pairs, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--mode", default="grounded",
                        help="one or more of " + ", ".join(MODES) + " (comma-separated; all "
                             "share one model load)")
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--pad-sec", type=float, default=0.3,
                        help="MUST match what 03b_cut_event_clips.py used (default 0.3)")
    parser.add_argument("--out-subdir", default="wrongclip",
                        help="subdirectory of model_outputs_dir (default: wrongclip) -- keeps "
                             "these out of the main 220-run set")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        parser.error(f"--mode: {bad} not in {MODES}")

    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg["paths"]["model_outputs_dir"] / args.out_subdir)
    backend_name = args.backend or cfg["model"]["backend"]
    n_runs = args.n_runs or cfg["model"]["n_runs_per_item"]

    scripts_dir = Path(__file__).resolve().parent
    prompt_text = load_prompt_template(scripts_dir / "config" / PROMPT_FILE)
    schema = load_schema(scripts_dir / "config" / SCHEMA_FILE)

    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Partners are assigned over ALL events, then --clip-id filters which prompts to run: the
    # pairing must not depend on the filter, or a spot-check would use a different partner than
    # the full run did.
    all_events = build_events(rows, cfg, args.pad_sec, clip_filter=None)
    pairs, dropped = assign_partners(all_events)
    if args.clip_id:
        pairs = [(a, b) for a, b in pairs if a["clip_id"] == args.clip_id]

    planned = []
    for prompt_ev, video_ev in pairs:
        rel_start, rel_end, clamped = clamp_span(prompt_ev["rel_start"], prompt_ev["rel_end"],
                                                 video_ev["duration"])
        for mode in modes:
            for run_idx in range(1, n_runs + 1):
                out_path = out_dir / (f"{prompt_ev['clip_id']}__{prompt_ev['event_id']}__"
                                      f"combined__run{run_idx}__wrongclip_{mode}.json")
                if out_path.exists() and not args.overwrite:
                    continue
                planned.append({
                    "prompt_ev": prompt_ev, "video_ev": video_ev, "mode": mode,
                    "run_idx": run_idx, "out_path": out_path,
                    "rel_start": rel_start, "rel_end": rel_end, "span_clamped": clamped,
                })

    planned.sort(key=lambda j: (modes.index(j["mode"]), j["prompt_ev"]["clip_id"],
                                j["prompt_ev"]["event_id"]))
    n_clamped = sum(1 for j in planned if j["span_clamped"])
    log.info("Planned %d call(s) across mode=%s, backend=%s (one model load)",
             len(planned), modes, backend_name)
    log.info("  %d event(s) excluded for having no within-articulator partner: %s",
             len(dropped), ", ".join(f"{e['clip_id']}/{e['event_id']} ({e['true_channel']})"
                                     for e in dropped) or "none")
    log.info("  %d/%d planned call(s) needed the target span clamped to the partner clip",
             n_clamped, len(planned))

    if args.dry_run:
        for job in planned:
            a, b = job["prompt_ev"], job["video_ev"]
            label = f"{a['clip_id']}/{a['event_id']}"
            try:
                fill_prompt(prompt_text, job["mode"], label, a["translation"], a["true_channel"],
                            job["rel_start"], job["rel_end"])
                status = "ok"
            except ValueError as exc:
                status = f"WOULD FAIL: {exc}"
            print(f"[dry-run] prompt={label} ({a['true_channel']}) video={b['clip_id']}/"
                  f"{b['event_id']} span=[{job['rel_start']:.1f},{job['rel_end']:.1f}]s"
                  f"{' CLAMPED' if job['span_clamped'] else ''} -> {job['out_path'].name} "
                  f"[{status}]")
        return

    from lib.model_backends import get_backend, build_backend_kwargs
    backend_kwargs = build_backend_kwargs(backend_name, cfg["model"])
    backend = get_backend(backend_name, **backend_kwargs)

    manifest_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "script": "31_run_wrongclip_condition.py",
        "condition": "wrongclip",
        "pairing": "within-articulator derangement; rotation of the duration-sorted group",
        "prompt_identical_to_main_run_except_video_and_clamped_span": True,
        "excluded_no_partner": [f"{e['clip_id']}/{e['event_id']}" for e in dropped],
        "n_span_clamped": n_clamped,
        "backend": backend_name,
        "model_name": backend.model_name,
        "model_version": backend.model_version,
        "prompt_modes": modes,
        "prompt_file": PROMPT_FILE,
        "clip_kind": "combined",
        "pad_sec": args.pad_sec,
        "backend_kwargs": backend_kwargs,
        "temperature": cfg["model"]["temperature"],
        "do_sample": cfg["model"]["do_sample"],
        "max_new_tokens": cfg["model"]["max_new_tokens"],
        "n_planned": len(planned),
    }
    mpath = out_dir / "run_manifest_wrongclip.json"
    history = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else []
    history.append(manifest_entry)
    mpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    for done, job in enumerate(planned, start=1):
        a, b = job["prompt_ev"], job["video_ev"]
        label = f"{a['clip_id']}/{a['event_id']}"
        if not b["target_clip"].exists() or not b["context_clip"].exists():
            log.warning("%s: partner %s/%s missing source clip(s), skipping", label,
                        b["clip_id"], b["event_id"])
            continue

        log.info("[%d/%d] prompt %s (%s) over video %s/%s (mode=%s) ...", done, len(planned),
                 label, a["true_channel"], b["clip_id"], b["event_id"], job["mode"])
        row_prompt = fill_prompt(prompt_text, job["mode"], label, a["translation"],
                                 a["true_channel"], job["rel_start"], job["rel_end"])
        raw_text = backend.generate(
            video_path=b["target_clip"],
            prompt_text=row_prompt,
            temperature=cfg["model"]["temperature"],
            do_sample=cfg["model"]["do_sample"],
            max_new_tokens=cfg["model"]["max_new_tokens"],
            context_video_path=b["context_clip"],
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
            # A model that notices the clip does not match the translation, and says so, is
            # producing a RESULT. Log and keep going; never retry or reword.
            log.warning("%s [wrongclip]: schema validation failed: %s", label, schema_errors)

        job["out_path"].write_text(json.dumps({
            "clip_id": a["clip_id"],
            "event_id": a["event_id"],
            "clip_kind": "combined",
            "condition": "wrongclip",
            "target_channel": a["true_channel"],
            "prompt_from": {"clip_id": a["clip_id"], "event_id": a["event_id"],
                            "translation": a["translation"], "video_id": a["video_id"],
                            "rel_start_requested": a["rel_start"],
                            "rel_end_requested": a["rel_end"]},
            "video_from": {"clip_id": b["clip_id"], "event_id": b["event_id"],
                           "video_id": b["video_id"], "target_channel": b["true_channel"],
                           "duration": b["duration"]},
            "rel_start_applied": job["rel_start"],
            "rel_end_applied": job["rel_end"],
            "span_clamped": job["span_clamped"],
            "video_path": str(b["target_clip"]),
            "context_clip_path": str(b["context_clip"]),
            "mode": job["mode"],
            "prompt_file": PROMPT_FILE,
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
