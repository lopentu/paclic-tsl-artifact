"""Backend interface that every VLM backend must implement.

Deliberately narrow: one method that takes a video (plus optional context video/text)
and a prompt, and returns raw text. All schema parsing/validation happens upstream in
07_run_inference.py so backends stay swappable without touching scoring code.
"""

from abc import ABC, abstractmethod


class VLMBackend(ABC):
    #: Human-readable model name/version, logged into run_manifest.json for every run
    #: (the project design log: model settings must be reported in the paper).
    model_name = "unset"
    model_version = "unset"

    @abstractmethod
    def generate(self, video_path, prompt_text, context_video_path=None,
                 context_text=None, temperature=0.0, do_sample=False,
                 max_new_tokens=1500, video_label=None, context_video_label=None):
        """Run one inference call and return the raw decoded text (not yet JSON-parsed).

        `video_label` / `context_video_label` override the text introducing each video --
        implementations default them to wording for the older local/context condition
        ("TARGET analysis window" / "PRECEDING DISCOURSE CONTEXT"), which callers that don't
        pass these get unchanged. 07b_run_inference_events.py's `combined` clip kind overrides
        both to describe a same-passage context+close-up pairing instead.
        """
        raise NotImplementedError
