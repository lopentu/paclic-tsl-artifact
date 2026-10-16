#!/usr/bin/env python3
"""Format-only screen of the cross-model pilot (the project design log, 2026-10-09).

Reads each candidate's pilot outputs (C01-C16, `combined`, grounded + ablate_channel) and reports
ONLY format statistics: calls completed, schema validity, leaked reasoning blocks, frames and tokens
fed. It deliberately never reads the substantive fields (scores, channel_judged, descriptions): the
primary model is chosen by a pre-registered rule from these numbers alone, so the choice cannot
depend on whether a model replicates the Qwen result.

Pass rule: all 40 calls completed AND >= 38/40 schema-valid. The first passing candidate in the
fixed order GLM-4.6V-Flash -> Gemma 4 E4B -> MiMo-VL is the primary replication model.

Usage:
    python scripts/34_screen_crossmodel_pilot.py
"""

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import REPO_ROOT

ORDER = ["glm46v_flash", "gemma4_e4b", "mimo_vl_7b"]
MODES = ["grounded", "ablate_channel"]
PILOT_MANIFEST = REPO_ROOT / "data" / "clip_manifest_crossmodel_pilot.csv"
MIN_VALID = 38
OUT_MD = REPO_ROOT / "data" / "results" / "crossmodel_pilot_screen.md"


def expected_jobs():
    jobs = []
    with open(PILOT_MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            events = ["event1"]
            if (row.get("event2_channel") or "").strip() not in ("", "N/A"):
                events.append("event2")
            for ev in events:
                for mode in MODES:
                    jobs.append((row["clip_id"], ev, mode))
    return jobs


def screen(tag, jobs):
    out_dir = REPO_ROOT / "data" / "model_outputs" / tag
    n_done = n_valid = n_think = n_empty_think = 0
    frames, in_tok, out_tok, failures = [], [], [], []
    for clip_id, ev, mode in jobs:
        p = out_dir / f"{clip_id}__{ev}__combined__run1__{mode}.json"
        if not p.exists():
            failures.append(f"{clip_id}/{ev}/{mode}: missing")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        n_done += 1
        raw = d.get("raw_text") or ""
        # An empty <think></think> pair is MiMo's no-think output format, not leaked reasoning;
        # only a block with content in it counts as a reasoning leak.
        blocks = re.findall(r"<think>(.*?)</think>", raw, re.S)
        if any(b.strip() for b in blocks) or ("<think>" in raw and "</think>" not in raw):
            n_think += 1
        elif blocks:
            n_empty_think += 1
        # The screen was run on the strict validator; a later post-hoc type normalization
        # (scripts/35) must not change its recorded outcome.
        if d.get("schema_valid_strict", d.get("schema_valid")):
            n_valid += 1
        else:
            errs = d.get("schema_errors_strict", d.get("schema_errors")) or []
            failures.append(f"{clip_id}/{ev}/{mode}: {'; '.join(map(str, errs))[:160]}")
        info = d.get("backend_call_info") or {}
        if info:
            frames.append(info["frames_per_video"])
            in_tok.append(info["input_tokens"])
            out_tok.append(info["output_tokens"])
    passed = n_done == len(jobs) and n_valid >= MIN_VALID
    return dict(tag=tag, n=len(jobs), done=n_done, valid=n_valid, think=n_think,
                empty_think=n_empty_think, passed=passed,
                frames=frames, in_tok=in_tok, out_tok=out_tok, failures=failures)


def main():
    jobs = expected_jobs()
    results = [screen(tag, jobs) for tag in ORDER]
    primary = next((r["tag"] for r in results if r["passed"]), None)

    lines = ["# Cross-model pilot: format screen", "",
             f"Rule (the project design log 2026-10-09): all {len(jobs)} calls complete and >= {MIN_VALID}/{len(jobs)} "
             "schema-valid; first passing model in the fixed order is primary. Format statistics only.", "",
             "| order | model tag | completed | schema-valid | reasoning leaked | empty think tags | input tokens (min-max) | output tokens (max) | pass |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        tok = f"{min(r['in_tok'])}-{max(r['in_tok'])}" if r["in_tok"] else "n/a"
        otok = str(max(r["out_tok"])) if r["out_tok"] else "n/a"
        lines.append(f"| {i} | {r['tag']} | {r['done']}/{r['n']} | {r['valid']}/{r['n']} | {r['think']} | {r['empty_think']} | "
                     f"{tok} | {otok} | {'PASS' if r['passed'] else 'fail'} |")
    lines += ["", f"**Primary replication model: {primary or 'none passed'}**", ""]
    for r in results:
        if r["frames"]:
            ctx = sorted(f[0] for f in r["frames"] if len(f) == 2)
            tgt = sorted(f[-1] for f in r["frames"])
            two = sum(len(f) == 2 for f in r["frames"])
            lines.append(f"- {r['tag']}: both videos fed on {two}/{len(r['frames'])} logged calls; "
                         f"window frames {ctx[0] if ctx else '-'}-{ctx[-1] if ctx else '-'}, "
                         f"target frames {tgt[0]}-{tgt[-1]}")
    for r in results:
        if r["failures"]:
            lines += ["", f"### {r['tag']} failures", ""] + [f"- {x}" for x in r["failures"]]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
