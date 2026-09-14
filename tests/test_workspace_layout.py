from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "dynamic-degree": "dynamic_degree",
    "motion-smoothness": "motion_smoothness",
    "subject-consistency": "subject_consistency",
    "scene": "scene",
    "human-action": "human_action",
    "spatial-relationship": "spatial_relationship",
    "overall-consistency": "overall_consistency",
    "multiple-objects": "multiple_objects",
}

KNOWN_MODEL_CLOSURE = {
    "motion-smoothness": {"audit-models[models]", "imageio", "omegaconf"},
    "subject-consistency": {"decord", "opencv-python", "pillow", "tqdm"},
    "spatial-relationship": {
        "boto3",
        "botocore",
        "decord",
        "easydict",
        "fvcore",
        "lvis",
        "opencv-python",
        "pillow",
        "requests",
        "timm<=1.0.12",
        "tqdm",
        "transformers==4.33.2",
    },
    "multiple-objects": {
        "boto3",
        "botocore",
        "decord",
        "easydict",
        "fvcore",
        "lvis",
        "opencv-python",
        "pillow",
        "requests",
        "timm<=1.0.12",
        "tqdm",
        "transformers==4.33.2",
    },
    "scene": {"decord", "fairscale>=0.4.4", "scipy", "timm<=1.0.12"},
    "overall-consistency": {"ftfy", "regex", "timm<=1.0.12"},
}


def test_workspace_members_use_english_src_layout():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["uv"]["workspace"]["members"] == ["metrics/*", "packages/*"]
    assert (ROOT / "packages/audit-core/src/vbench_audit_core").is_dir()
    for distribution, import_name in METRICS.items():
        project = ROOT / "metrics" / distribution
        assert (project / "pyproject.toml").is_file()
        assert (project / "src" / import_name).is_dir()
        assert not (project / "tests/__init__.py").exists()


def test_metric_metadata_does_not_depend_on_another_metric():
    names = set(METRICS)
    for distribution in METRICS:
        text = (ROOT / "metrics" / distribution / "pyproject.toml").read_text(encoding="utf-8")
        for other in names - {distribution}:
            assert f'"{other}"' not in text


def test_known_official_model_import_closures_are_declared():
    """Keep known upstream video/model imports explicit in each member extra."""
    for distribution, expected in KNOWN_MODEL_CLOSURE.items():
        config = tomllib.loads(
            (ROOT / "metrics" / distribution / "pyproject.toml").read_text(encoding="utf-8")
        )
        declared = set(config["project"]["optional-dependencies"]["models"])
        assert expected <= declared
