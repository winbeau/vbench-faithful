"""The package entry selects one real official task and preserves its reducer."""
import json
from types import SimpleNamespace

from imaging_quality.cli import main
from imaging_quality import official


def test_yaml_entry_selects_only_its_dimension(tmp_path, capsys):
    (tmp_path / "clip.mp4").write_bytes(b"plan only; never decoded")
    (tmp_path / "inputs.json").write_text(json.dumps([{
        "video": "clip.mp4", "dimensions": ["imaging_quality", "scene"]
    }]))
    (tmp_path / "assets.json").write_text("{}")
    config = tmp_path / "eval.yaml"
    config.write_text("input: inputs.json\nassets: assets.json\ndimensions: all\nbackend: repair\n")
    assert main(["--config", str(config), "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert list(plan["methods"]) == ["imaging_quality/accelerated"]
    assert plan["official_fallback"] == []
    assert plan["accelerated_dimensions"] == ["imaging_quality"]
    assert not (tmp_path / "output").exists()


def test_adapter_preserves_arguments_rows_and_native_aggregate(monkeypatch):
    returned = (.7, [{"video_path": "clip.mp4", "video_results": 70.0}])
    arguments = []

    def compute(*args):
        arguments.append(args)
        return returned

    def load(dimension):
        assert dimension == "imaging_quality"
        return SimpleNamespace(compute_imaging_quality=compute), None

    monkeypatch.setattr(official, "import_official_module", load)
    assets, device = object(), object()
    assert official.compute("manifest.json", device, assets) is returned
    assert arguments == [("manifest.json", device, assets)]
