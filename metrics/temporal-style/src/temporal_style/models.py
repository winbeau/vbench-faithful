"""Lazy ViCLIP requirements for temporal style."""

MODEL_ASSETS = ("viclip_checkpoint", "viclip_bpe")


def required_assets() -> tuple[str, ...]:
    return MODEL_ASSETS


def build_model(*_args, **_kwargs):
    raise NotImplementedError("temporal-style models are deferred to a later milestone")
