"""Shared Prompt C rendering: conditional-block substitution and channel wording.

Used by both 07_run_inference.py (the original windows_manifest-driven pipeline) and
07b_run_inference_events.py (the clip_manifest.csv / C01-C40 pipeline) so the two drivers can
never render the cue-ablation blocks differently or diverge on channel wording -- both read the
same prompt file under the same rules.
"""

import re
from pathlib import Path

# [[NAME]] ... [[/NAME]] conditional blocks in the prompt file.
BLOCK_RE = re.compile(r"\[\[(?P<name>[A-Z_]+)\]\]\n(?P<body>.*?)\[\[/(?P=name)\]\]\n", re.DOTALL)
ALL_BLOCKS = {"MEANING_GIVEN", "MEANING_WITHHELD", "CHANNEL_GIVEN", "CHANNEL_WITHHELD"}

# Coarse, FIXED channel wording. Kept generic on purpose so naming the channel does not leak
# the specific form (never "eyebrow raise"/"head nod"/"point") or its function.
CHANNEL_PHRASES = {
    "face": "the signer's facial activity",
    "head_torso": "the signer's head and upper-body (torso) movement",
    "non_dominant_hand": "the signer's non-dominant (weak) hand",
    "manual": "the signer's hand/manual activity",
}


def load_prompt_template(path):
    """Load a prompt file, stripping the trailing researcher-facing NOTE ON USE section, and
    verify all four conditional blocks are present -- a renamed or missing block would silently
    hand the model the wrong cues under some --mode."""
    text = Path(path).read_text(encoding="utf-8")
    text = text.split("\nNOTE ON USE:")[0].rstrip() + "\n"
    found = {m.group("name") for m in BLOCK_RE.finditer(text)}
    if found != ALL_BLOCKS:
        raise ValueError(
            f"{path}: expected conditional blocks {sorted(ALL_BLOCKS)}, found {sorted(found)}. "
            "The cue ablation renders these per --mode; a renamed or missing block would silently "
            "hand the model the wrong cues."
        )
    return text


def render_blocks(text, keep):
    """Keep the bodies of the named [[BLOCK]]s, delete the others, then tidy blank lines."""
    rendered = BLOCK_RE.sub(lambda m: m.group("body") if m.group("name") in keep else "", text)
    return re.sub(r"\n{3,}", "\n\n", rendered).strip() + "\n"


def substitute_prompt_fields(text, *, item_label, mode, translation=None, channel_code=None,
                             rel_start, rel_end):
    """Fill in __UTTERANCE_MEANING__ / __TARGET_CHANNEL__ / __EVENT_START_SEC__ / __EVENT_END_SEC__
    and confirm nothing was left unsubstituted.

    `translation` / `channel_code` are None exactly when that cue is withheld under `mode` (the
    corresponding block was already rendered out by render_blocks, so there is no placeholder left
    to fill for it). `item_label` is only used for error messages (a window_id, or a
    clip_id/event_id/clip_kind combination).
    """
    if translation is not None:
        if not translation.strip():
            raise ValueError(f"{item_label}: --mode {mode} supplies the translation, but the "
                             "manifest's translation column is empty (see README)")
        text = text.replace("__UTTERANCE_MEANING__", translation)
    if channel_code is not None:
        if channel_code not in CHANNEL_PHRASES:
            raise ValueError(f"{item_label}: target_channel '{channel_code}' not one of {sorted(CHANNEL_PHRASES)}")
        text = text.replace("__TARGET_CHANNEL__", CHANNEL_PHRASES[channel_code])

    text = (text
            .replace("__EVENT_START_SEC__", f"{rel_start:.1f}")
            .replace("__EVENT_END_SEC__", f"{rel_end:.1f}"))

    leftover = re.findall(r"__[A-Z_]+__", text)
    if leftover:
        raise ValueError(f"{item_label}: unsubstituted placeholder(s) {leftover} under --mode {mode}")
    return text
