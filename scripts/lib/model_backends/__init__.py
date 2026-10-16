"""Registry for VLM backends. Add new backends here so 07b_run_inference_events.py
can select them by name from config/pipeline_config.yaml (`model.backend`).
"""

_REGISTRY = {}


def _lazy_register():
    if _REGISTRY:
        return
    from .qwen3_vl_backend import Qwen3VLBackend
    from .crossmodel_backends import Gemma4Backend, Glm46VBackend, MimoVLBackend

    _REGISTRY.update({
        "qwen3_vl": Qwen3VLBackend,
        "gemma4": Gemma4Backend,
        "glm46v": Glm46VBackend,
        "mimo_vl": MimoVLBackend,
    })


def get_backend(name, **kwargs):
    _lazy_register()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


# Which pipeline_config `model:` keys each backend actually accepts. Kept here, next to the
# registry, because every runner script (07b, 19-21, 26-28, 31, 32) builds the same kwargs and
# the failure mode when they drift is silent: a key the backend does not take is simply not
# passed, so a config setting the run manifest reports as active was never applied. Adding a
# backend means adding one line here rather than editing every runner.
BACKEND_CONFIG_KEYS = {
    "qwen3_vl": ["fps", "max_pixels", "quantization"],
    "gemma4": ["fps", "max_pixels"],
    "glm46v": ["fps", "max_pixels"],
    "mimo_vl": ["fps", "max_pixels"],
}


def build_backend_kwargs(backend_name, model_cfg):
    """Constructor kwargs for `backend_name` from a pipeline config's `model:` block.

    model_id and device are always passed. Everything else is passed only if the backend accepts
    it AND the config actually sets it (a null/absent value is left to the backend default, so a
    config need not spell out every knob).
    """
    kwargs = {"model_id": model_cfg["model_id"], "device": model_cfg["device"]}
    for key in BACKEND_CONFIG_KEYS.get(backend_name, []):
        if model_cfg.get(key) is not None:
            kwargs[key] = model_cfg[key]
    return kwargs
