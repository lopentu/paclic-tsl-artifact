"""JSON-schema loading/validation for model output (see config/prompts/prompt_c_output.schema.json)."""

import json

import jsonschema


def load_schema(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_output(instance, schema):
    """Return (is_valid, list_of_error_messages). Never raises -- model output is untrusted."""
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    messages = [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]
    return (len(messages) == 0), messages


def extract_json_object(raw_text):
    """Best-effort extraction of a JSON object from raw model text (handles ```json fences, prose, etc.)."""
    text = raw_text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[len("json"):].strip()
            if part.startswith("{"):
                text = part
                break
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start:end + 1])
