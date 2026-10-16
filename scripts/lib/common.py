"""Shared paths, config loading, and logging setup for the pipeline scripts.

Every top-level script under scripts/ does:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lib.common import load_config, get_logger, REPO_ROOT

so scripts can be invoked from any working directory, e.g. `python scripts/07_run_inference.py`.
"""

import logging
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPTS_DIR / "config" / "pipeline_config.yaml"

# --- Inference modes -------------------------------------------------------------------
# Every mode is one prompt-cue configuration of Prompt C, and names the __run<N>__<mode>.json
# output set. Defined here (not per script) so 07 and every scorer that takes --mode cannot
# drift out of sync.
#
# The 2x2 cue ablation (added 2026-07-27) exists to answer: is the model reading the visible
# form, or reconstructing D1/D2 from the translation and the named channel? The target time
# span is given in every cell -- it is what identifies the action being scored, so withholding
# it would leave nothing to compare against the reference annotation.
#
#   mode                 translation given   target channel given
#   grounded                    yes                 yes
#   ablate_channel              yes                 no
#   ablate_translation          no                  yes
#   ablate_both                 no                  no
#
# grounded_distractor is a different control: both cues given, but the target span is moved to
# another interval in the same clip (see 16_score_target_sensitivity.py).
CUE_ABLATION_MODES = ["ablate_channel", "ablate_translation", "ablate_both"]
INFERENCE_MODES = ["grounded"] + CUE_ABLATION_MODES + ["grounded_distractor"]

# Which cues each mode supplies to the model.
MODE_CUES = {
    "grounded":            {"translation": True,  "channel": True,  "distractor_span": False},
    "ablate_channel":      {"translation": True,  "channel": False, "distractor_span": False},
    "ablate_translation":  {"translation": False, "channel": True,  "distractor_span": False},
    "ablate_both":         {"translation": False, "channel": False, "distractor_span": False},
    "grounded_distractor": {"translation": True,  "channel": True,  "distractor_span": True},
}


def load_config(config_path=None):
    """Load pipeline_config.yaml (or an override) and resolve relative paths against REPO_ROOT."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    resolved_paths = {}
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        resolved_paths[key] = p if p.is_absolute() else (REPO_ROOT / p)
    cfg["paths"] = resolved_paths
    return cfg


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --- clip_manifest.csv event parsing ---------------------------------------------------
# Shared by 03b_cut_event_clips.py, 07b_run_inference_events.py, and
# 08b_score_localization_events.py so all three agree on what a manifest row's event N is.
NA_VALUES = {"n/a", "na", "none", "-"}
VALID_CHANNELS = {"face", "head_torso", "non_dominant_hand", "manual"}


def is_na(value):
    return (value or "").strip().lower() in NA_VALUES


def parse_manifest_event(row, index):
    """Return (start, end, channel) for clip_manifest.csv's event `index` (1 or 2), or None if
    the row marks it N/A in all three columns.

    Raises ValueError for a half-filled event so a partially-annotated manifest is caught at the
    point of use rather than silently producing fewer clips/prompts than the study expects.
    """
    keys = [f"event{index}_start_sec", f"event{index}_end_sec", f"event{index}_channel"]
    raw = [(row.get(k) or "").strip() for k in keys]

    if all(is_na(v) for v in raw):
        return None
    if any(is_na(v) for v in raw):
        raise ValueError(f"event{index}: mixes N/A with real values ({dict(zip(keys, raw))}) -- "
                         "mark all three N/A or fill all three")
    if not all(raw):
        missing = [k for k, v in zip(keys, raw) if not v]
        raise ValueError(f"event{index}: blank {missing}. Blank means 'not decided yet'; write N/A "
                         f"in all three columns if this clip genuinely has no event {index}")

    start, end = float(raw[0]), float(raw[1])
    if end < start:
        raise ValueError(f"event{index}: end ({end}) is before start ({start}) -- looks reversed")
    # end == start is allowed: a point-in-time marker for an action too fast to bound with two
    # distinct timestamps -- padded_event_bounds still builds a real clip around it.
    if raw[2] not in VALID_CHANNELS:
        raise ValueError(f"event{index}: channel '{raw[2]}' not one of {sorted(VALID_CHANNELS)}")
    return start, end, raw[2]


def padded_event_bounds(event_start, event_end, pad_sec, duration=None):
    """The [clip_start, clip_end] a tight event clip is cut to: the event span padded by a fixed,
    symmetric amount on each side, clamped to [0, duration] (duration is optional -- omit it to
    skip the upper clamp, e.g. when checking bounds before the source video is available).

    Single source of truth for this arithmetic: 03b_cut_event_clips.py uses it to decide what to
    actually cut, and 07b_run_inference_events.py / 08b_score_localization_events.py use it to
    recover, from clip_manifest.csv alone, exactly where the event sits inside the clip that was
    cut -- these two computations must be identical, or the timestamp given to the model (or the
    reference span a scorer compares against) silently stops matching the real video content.
    """
    clip_start = max(0.0, event_start - pad_sec)
    clip_end = event_end + pad_sec
    if duration is not None:
        clip_end = min(clip_end, duration)
    return clip_start, clip_end


def kind_clip_bounds(kind, clip_start_sec, clip_end_sec, ev_start, ev_end, pad_sec):
    """The [start, end] (source-video absolute seconds) of a C01-C40 clip file for clip kind
    `kind` ("window", "ev1", or "ev2"), using the same arithmetic 03b_cut_event_clips.py used to
    cut it. Shared by 07b_run_inference_events.py (to compute the model-facing relative
    timestamp) and 08b_score_localization_events.py (to compute the reference span in the same
    clip-relative units the model's own self-reported span is in) -- both must derive identical
    bounds from clip_manifest.csv alone, with no dependence on what --pad-sec was actually passed
    to 03b at cut time.
    """
    if kind == "window":
        return max(0.0, clip_start_sec), clip_end_sec
    return padded_event_bounds(ev_start, ev_end, pad_sec)


def parse_timestamp(value):
    """Parse a timestamp given as seconds (float/int) or 'MM:SS' / 'HH:MM:SS' string into seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    parts = [float(p) for p in value.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds
