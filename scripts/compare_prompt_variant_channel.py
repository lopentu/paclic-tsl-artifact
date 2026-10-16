#!/usr/bin/env python3
"""Prompt-robustness check: does the model's `channel_judged` depend on how the CHANNEL_WITHHELD
block words its four articulator options?

Background . In the reported dataset the model never once self-selects
`non_dominant_hand` under either CHANNEL_WITHHELD mode (0/165 each), while 7 manifest events have
target_channel=non_dominant_hand and `manual` takes ~92-95% of all self-judgments. Two causes could
not be separated from that dataset alone: (a) a list-order/anchoring artifact of the prompt wording
-- the main prompt names the options inline with "the hands" first and plainest and "the
non-dominant hand alone" buried in a qualified sub-clause -- or (b) genuine difficulty recognizing a
communicatively active but visually quiet articulator, which several of those 7 events may
deliberately target (non-dominant-hand hold / 副手 carryover).

This compares the main dataset against a variant-prompt run that changes ONLY the CHANNEL_WITHHELD
block (prompts/prompt_c_grounded_attribution_altlist.md: parallel bulleted list, "equally likely",
manual moved last, plus one sentence noting an articulator can carry the action while barely
moving). The main prompt is deliberately NOT edited -- it stays pinned to the data it generated --
so this is an additive check, not a replacement.

Reports, per CHANNEL_WITHHELD mode:
  - channel_judged distribution, main vs variant, and agreement with clip_manifest.csv
  - every item whose channel_judged changed (the manipulation's "power": did it move anything?)
  - the 7 non_dominant_hand-target events specifically
  - schema-validity rate (a longer prompt could plausibly degrade an 8B model's JSON compliance)
  - D1/D2 drift, as a sanity check only -- NOT a result: the variant set is not perception-gated,
    not human-rated, and must never be reported as an attribution finding.

Usage:
    python scripts/compare_prompt_variant_channel.py
    python scripts/compare_prompt_variant_channel.py --variant-tag qwen3vl_8b_altlist
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import get_logger, ensure_dir, REPO_ROOT, parse_manifest_event

log = get_logger("compare_prompt_variant_channel")

# Only these two modes render the CHANNEL_WITHHELD block; the variant is byte-identical to the main
# prompt under CHANNEL_GIVEN, so grounded/ablate_translation are intentionally out of scope.
MODES = ["ablate_channel", "ablate_both"]
CLIP_KIND = "combined"


def load_target_channels(manifest_path):
    out = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for index, event_id in ((1, "event1"), (2, "event2")):
                parsed = parse_manifest_event(row, index)
                if parsed is not None:
                    out[(row["clip_id"], event_id)] = parsed[2]
    return out


def load_runs(outputs_dir, mode):
    """(clip_id, event_id) -> dict of the fields this check compares."""
    out = {}
    for path in sorted(Path(outputs_dir).glob(f"*__{CLIP_KIND}__run1__{mode}.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "clip_id" not in data:
            continue
        parsed = data.get("parsed") or {}
        cands = parsed.get("candidate_actions") if isinstance(parsed, dict) else None
        cand = cands[0] if cands else {}
        out[(data["clip_id"], data["event_id"])] = {
            "schema_valid": bool(data.get("schema_valid")),
            "channel_judged": cand.get("channel_judged"),
            "d1": cand.get("semantic_syntactic_load"),
            "d2": cand.get("pragmatic_depictive_affective_enrichment"),
            "prompt_file": data.get("prompt_file"),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--main-tag", default="qwen3vl_8b", help="reported dataset's output tag")
    parser.add_argument("--variant-tag", default="qwen3vl_8b_altlist", help="variant-prompt tag")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else REPO_ROOT / "clip_manifest.csv"
    target = load_target_channels(manifest_path)
    ndh_events = sorted(k for k, v in target.items() if v == "non_dominant_hand")

    main_dir = REPO_ROOT / "data" / "model_outputs" / args.main_tag
    var_dir = REPO_ROOT / "data" / "model_outputs" / args.variant_tag
    out_dir = ensure_dir(REPO_ROOT / "data" / "results" / args.variant_tag)

    rows, summary_lines = [], []
    summary_lines.append("# Prompt-robustness check: CHANNEL_WITHHELD articulator-list wording\n")
    summary_lines.append(f"\nMain dataset tag: `{args.main_tag}` (the reported prompt). "
                         f"Variant tag: `{args.variant_tag}`.\n")
    summary_lines.append(f"Scope: `clip_kind={CLIP_KIND}`, the two CHANNEL_WITHHELD modes only "
                         f"(the variant is byte-identical to the main prompt under CHANNEL_GIVEN).\n")

    for mode in MODES:
        m, v = load_runs(main_dir, mode), load_runs(var_dir, mode)
        keys = sorted(set(m) & set(v))
        if not keys:
            log.warning("No overlapping runs for mode %s -- skipped.", mode)
            continue

        n = len(keys)
        agree_m = sum(m[k]["channel_judged"] == target.get(k) for k in keys)
        agree_v = sum(v[k]["channel_judged"] == target.get(k) for k in keys)
        ndh_m = sum(m[k]["channel_judged"] == "non_dominant_hand" for k in keys)
        ndh_v = sum(v[k]["channel_judged"] == "non_dominant_hand" for k in keys)
        changed = [k for k in keys if m[k]["channel_judged"] != v[k]["channel_judged"]]

        log.info("=== %s (n=%d) ===", mode, n)
        log.info("  channel_judged matches target: main %d/%d (%.0f%%) -> variant %d/%d (%.0f%%)",
                 agree_m, n, 100 * agree_m / n, agree_v, n, 100 * agree_v / n)
        log.info("  channel_judged == non_dominant_hand: main %d -> variant %d", ndh_m, ndh_v)
        log.info("  main dist:    %s", dict(Counter(m[k]["channel_judged"] for k in keys)))
        log.info("  variant dist: %s", dict(Counter(v[k]["channel_judged"] for k in keys)))
        log.info("  channel_judged changed on %d/%d items", len(changed), n)
        log.info("  schema_valid: main %d/%d -> variant %d/%d",
                 sum(m[k]["schema_valid"] for k in keys), n,
                 sum(v[k]["schema_valid"] for k in keys), n)

        summary_lines.append(f"\n## `{mode}` (n={n})\n\n")
        summary_lines.append(f"| | main prompt | variant prompt |\n|---|---|---|\n")
        summary_lines.append(f"| `channel_judged` matches manifest | {agree_m}/{n} "
                             f"({100*agree_m/n:.0f}%) | {agree_v}/{n} ({100*agree_v/n:.0f}%) |\n")
        summary_lines.append(f"| selected `non_dominant_hand` | {ndh_m} | {ndh_v} |\n")
        summary_lines.append(f"| schema-valid | {sum(m[k]['schema_valid'] for k in keys)}/{n} | "
                             f"{sum(v[k]['schema_valid'] for k in keys)}/{n} |\n")
        summary_lines.append(f"| distribution | {dict(Counter(m[k]['channel_judged'] for k in keys))} | "
                             f"{dict(Counter(v[k]['channel_judged'] for k in keys))} |\n")
        summary_lines.append(f"\n`channel_judged` changed on **{len(changed)}/{n}** items "
                             f"(the manipulation's power -- it did move judgments):\n\n")
        for k in changed:
            summary_lines.append(f"- {k[0]}/{k[1]}: `{m[k]['channel_judged']}` -> "
                                 f"`{v[k]['channel_judged']}` (manifest target: `{target.get(k)}`)\n")

        summary_lines.append(f"\nThe {len(ndh_events)} `non_dominant_hand`-target events:\n\n")
        for k in ndh_events:
            if k in keys:
                summary_lines.append(f"- {k[0]}/{k[1]}: main `{m[k]['channel_judged']}`, "
                                     f"variant `{v[k]['channel_judged']}`\n")

        d1m = [m[k]["d1"] for k in keys if m[k]["d1"] is not None]
        d1v = [v[k]["d1"] for k in keys if v[k]["d1"] is not None]
        d2m = [m[k]["d2"] for k in keys if m[k]["d2"] is not None]
        d2v = [v[k]["d2"] for k in keys if v[k]["d2"] is not None]
        same_d1 = sum(m[k]["d1"] == v[k]["d1"] for k in keys)
        same_d2 = sum(m[k]["d2"] == v[k]["d2"] for k in keys)
        log.info("  sanity (NOT a result): D1 mean %.2f -> %.2f, identical %d/%d; "
                 "D2 mean %.2f -> %.2f, identical %d/%d",
                 statistics.mean(d1m), statistics.mean(d1v), same_d1, n,
                 statistics.mean(d2m), statistics.mean(d2v), same_d2, n)
        summary_lines.append(f"\nSanity check only, **not an attribution result** (this set is not "
                             f"perception-gated or human-rated): D1 mean "
                             f"{statistics.mean(d1m):.2f} -> {statistics.mean(d1v):.2f} "
                             f"(identical on {same_d1}/{n}); D2 mean "
                             f"{statistics.mean(d2m):.2f} -> {statistics.mean(d2v):.2f} "
                             f"(identical on {same_d2}/{n}).\n")

        for k in keys:
            rows.append({
                "mode": mode, "clip_id": k[0], "event_id": k[1], "clip_kind": CLIP_KIND,
                "target_channel": target.get(k, ""),
                "main_channel_judged": m[k]["channel_judged"],
                "variant_channel_judged": v[k]["channel_judged"],
                "channel_judged_changed": m[k]["channel_judged"] != v[k]["channel_judged"],
                "main_matches_target": m[k]["channel_judged"] == target.get(k),
                "variant_matches_target": v[k]["channel_judged"] == target.get(k),
                "main_schema_valid": m[k]["schema_valid"],
                "variant_schema_valid": v[k]["schema_valid"],
                "main_d1": m[k]["d1"], "variant_d1": v[k]["d1"],
                "main_d2": m[k]["d2"], "variant_d2": v[k]["d2"],
            })

    if not rows:
        raise SystemExit("No comparable runs found -- check both tags exist and were run for "
                         f"--clip-kind {CLIP_KIND}.")

    csv_path = out_dir / "prompt_variant_channel_comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = out_dir / "prompt_variant_channel_comparison.md"
    md_path.write_text("".join(summary_lines), encoding="utf-8")
    log.info("Wrote %s", csv_path)
    log.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
