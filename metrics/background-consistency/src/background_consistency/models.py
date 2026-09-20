"""Lazy model requirements for background consistency.

M1 records the contract only.  Importing this module never imports torch or
attempts to resolve a checkpoint.
"""

MODEL_ASSETS = ("clip_vit_b32",)


def required_assets() -> tuple[str, ...]:
    return MODEL_ASSETS


def build_model(*_args, **_kwargs):
    raise NotImplementedError("background-consistency models are deferred to a later milestone")
