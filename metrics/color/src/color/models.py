"""Lazy GRiT requirements for color."""

MODEL_ASSETS = ("grit_densecap_checkpoint", "grit_objectdet_checkpoint")


def required_assets() -> tuple[str, ...]:
    return MODEL_ASSETS


def build_model(config, *, device):
    from pathlib import Path
    from vbench_audit_models.grit import GritEvidenceModel
    return GritEvidenceModel("color", Path(config["grit"]["checkpoint"]),
                             device=device or "cpu",
                             upstream=config.get("runtime", {}).get("upstream_root"))
