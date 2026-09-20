"""Lazy GRiT requirements for object class."""

MODEL_ASSETS = ("grit_objectdet_checkpoint",)


def required_assets() -> tuple[str, ...]:
    return MODEL_ASSETS


def build_model(*_args, **_kwargs):
    raise NotImplementedError("object-class models are deferred to a later milestone")
