from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.four_dimension import DIMENSIONS, build_parser, main


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "VBench"
PROTOCOL = ROOT / "configs" / "four_dimension" / "protocol.toml"
MODELS = ROOT / "configs" / "four_dimension" / "models.h100.toml"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_driver_help_and_common_arguments_are_shared() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "offline",
            "--root",
            "/tmp/m2-contract",
            "--upstream",
            "../VBench",
            "--protocol",
            str(PROTOCOL),
            "--models",
            str(MODELS),
            "--dimension",
            "color",
        ]
    )
    assert parsed.command == "offline"
    assert parsed.dimension == "color"
    assert parsed.root == Path("/tmp/m2-contract")
    assert set(DIMENSIONS) == {
        "background_consistency",
        "temporal_style",
        "object_class",
        "color",
    }


def test_offline_writes_source_witnesses_and_null_model_measurements(tmp_path: Path) -> None:
    output = tmp_path / "offline"
    rc = main(
        [
            "offline",
            "--root",
            str(output),
            "--upstream",
            str(UPSTREAM),
            "--protocol",
            str(PROTOCOL),
            "--models",
            str(MODELS),
        ]
    )
    assert rc == 0
    source = read_json(output / "offline" / "source-evidence.json")
    assert source["counts"] == {
        "background_consistency": 86,
        "color": 85,
        "object_class": 79,
        "temporal_style": 100,
    }
    assert source["background_scene_prompt_equal"] is True
    assert source["temporal_normalized_AST_equal"] is True
    assert source["fixtures"]["object_rule_fixture_couch_sofa"] == [0, 1]
    assert read_json(output / "offline" / "temporal-tokens.json")["status"] in {
        "blocked",
        "verified",
    }
    for dimension in DIMENSIONS:
        report = read_json(output / "offline" / f"{dimension}.json")
        assert report["status"] in {"verified", "blocked"}
        assert report["model_measurement"] == {"status": "not_run", "value": None}


def test_offline_dimension_selection_does_not_require_temporal_tokenizer(tmp_path: Path) -> None:
    output = tmp_path / "selected"
    rc = main(
        [
            "offline",
            "--dimension",
            "object_class",
            "--root",
            str(output),
            "--upstream",
            str(UPSTREAM),
            "--protocol",
            str(PROTOCOL),
            "--models",
            str(MODELS),
        ]
    )
    assert rc == 0
    assert (output / "offline" / "object_class.json").is_file()
    assert not (output / "offline" / "temporal_style.json").is_file()
    assert read_json(output / "offline" / "temporal-tokens.json")["status"] == "not_applicable"


def test_inventory_has_stable_five_file_schema_and_records_missing_gpu_assets(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for generator in ("cogvideo", "lavie", "modelscope", "videocrafter"):
        for folder in ("scene", "temporal_style", "object_class", "color"):
            folder_path = dataset / generator / folder
            folder_path.mkdir(parents=True)
            (folder_path / "video_000.mp4").write_bytes(b"fixture")

    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("clip.pt", "viclip.pt", "bpe.gz", "grit.pt"):
        (assets / name).write_bytes((name + "-fixture").encode())
    models = tmp_path / "models.toml"
    models.write_text(
        "\n".join(
            [
                "[runtime]",
                f'python = "{sys.executable}"',
                f'upstream = "{UPSTREAM}"',
                "",
                "[models]",
                f'clip_vit_b32 = "{assets / "clip.pt"}"',
                f'viclip_checkpoint = "{assets / "viclip.pt"}"',
                f'viclip_bpe = "{assets / "bpe.gz"}"',
                f'grit_checkpoint = "{assets / "grit.pt"}"',
                "",
                '[dimensions.background_consistency]',
                'model = "clip_vit_b32"',
                '[dimensions.temporal_style]',
                'model = "viclip_checkpoint"',
                'text_assets = ["viclip_bpe"]',
                '[dimensions.object_class]',
                'model = "grit_checkpoint"',
                '[dimensions.color]',
                'model = "grit_checkpoint"',
                'heads = ["densecap", "objectdet"]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "inventory"
    rc = main(
        [
            "inventory",
            "--root",
            str(output),
            "--upstream",
            str(UPSTREAM),
            "--protocol",
            str(PROTOCOL),
            "--models",
            str(models),
            "--dataset-root",
            str(dataset),
            "--gpus",
            "1,2",
            "--python",
            "/does/not/exist",
        ]
    )
    assert rc == 0
    expected = {
        "assets.json",
        "source-videos.json",
        "source-code.json",
        "gpus.json",
        "environment.json",
    }
    assert {path.name for path in (output / "inventory").glob("*.json")} == expected
    assets_report = read_json(output / "inventory" / "assets.json")
    assert assets_report["status"] == "verified"
    source_videos = read_json(output / "inventory" / "source-videos.json")
    assert source_videos["dimensions"]["background_consistency"]["source_folder"] == "scene"
    assert source_videos["dimensions"]["background_consistency"]["shared_with"] == ["scene"]
    relation = source_videos["shared_video_sources"]["background_consistency"]["generators"][0]
    assert relation["same_canonical_path"] is True
    assert read_json(output / "inventory" / "gpus.json")["status"] == "blocked"
