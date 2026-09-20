"""Lazy GRiT requirements for color."""

MODEL_ASSETS = ("grit_densecap_checkpoint", "grit_objectdet_checkpoint")


def required_assets() -> tuple[str, ...]:
    return MODEL_ASSETS


def build_model(*_args, **_kwargs):
    raise NotImplementedError("color models are deferred to a later milestone")
