#!/usr/bin/env python3
"""Admit outputs whose only schema violations are JSON types the prompt never asked for (the project design log 2026-10-09).

Prompt C states its requirements explicitly: valid JSON, one object, exactly one candidate, every
required field, the enum values, and the ranges (scores 0-3, confidence 0.0-1.0). It never states
the JSON TYPE of a field; that is only implied by the example. A model that writes `0.3` for a span
time or `"1"` for a score has given the same value in a type it was never told to use, so excluding
it would be the validator's fault, not the model's. Such an output is converted and re-validated.

Scope:
- An output is touched only if EVERY strict-validation error is a type error on a string, integer or
  number field. Any other violation (missing field, enum, range, malformed JSON) leaves it invalid.
- Conversions are lossless only: number -> string for string fields; an integer string ("2") ->
  int for integer fields; a decimal string ("0.8") -> float for number fields. Anything else (e.g.
  "2.5" for an integer field, "high" for confidence) is left as is, so the output stays invalid.
- The converted output must then pass the UNCHANGED schema, so ranges and enums are still enforced.

Nothing is lost: raw_text is never touched, and the strict verdict is kept alongside as
schema_valid_strict / schema_errors_strict, with the converted fields named in
post_hoc_normalization. Subdirectories (e.g. mis-cue runs) are included.

Usage:
    python scripts/35_normalize_type_only_errors.py --tag mimo_vl_7b [--dry-run]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import REPO_ROOT
from lib.schema_utils import load_schema, validate_output

TYPE_ERROR = re.compile(r"^(?P<path>[^:]+): .* is not of type '(?P<type>string|integer|number)'$")
INT_STR = re.compile(r"^-?\d+$")
NUM_STR = re.compile(r"^-?\d+(\.\d+)?$")


def convert(value, target):
    """Return (ok, new_value) for a lossless conversion to the schema's type, else (False, value)."""
    if target == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return True, str(value)
    if target == "integer" and isinstance(value, str) and INT_STR.match(value.strip()):
        return True, int(value.strip())
    if target == "number" and isinstance(value, str) and NUM_STR.match(value.strip()):
        return True, float(value.strip())
    return False, value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    schema = load_schema(Path(__file__).resolve().parent / "config" / "prompts" / "prompt_c_output.schema.json")
    out_dir = REPO_ROOT / "data" / "model_outputs" / args.tag
    for p in sorted(out_dir.rglob("C*__run*__*.json")):
        name = p.relative_to(out_dir)
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("schema_valid") or not d.get("parsed"):
            continue
        errors = [str(e) for e in d.get("schema_errors") or []]
        matches = [TYPE_ERROR.match(e) for e in errors]
        if not errors or not all(matches):
            print(f"left invalid (not type-only): {name}")
            continue
        parsed = d["parsed"]
        converted = []
        for m in matches:
            *parents, leaf = [int(k) if k.isdigit() else k for k in m["path"].split("/")]
            node = parsed
            for k in parents:
                node = node[k]
            ok, node[leaf] = convert(node[leaf], m["type"])
            if not ok:
                break
            converted.append(f"{m['path']} -> {m['type']}")
        ok, new_errors = validate_output(parsed, schema)
        if len(converted) < len(matches) or not ok:
            print(f"left invalid (not losslessly convertible): {name}: {new_errors or errors}")
            continue
        print(f"admitted: {name}: {', '.join(converted)}")
        if args.dry_run:
            continue
        d.update({
            "schema_valid_strict": False,
            "schema_errors_strict": errors,
            "post_hoc_normalization": "type-only, re-validated (the project design log 2026-10-09): " + ", ".join(converted),
            "parsed": parsed,
            "schema_valid": True,
            "schema_errors": [],
        })
        p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
