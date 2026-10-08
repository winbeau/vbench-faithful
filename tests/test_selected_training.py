"""Integration contracts for the retained paper training/data entry points."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.semantic import audit_scene_sources, prepare_action_isolation, prepare_repair_training
from scripts.semantic.construction import rows_from_manifest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", [p for p in sorted((ROOT / "scripts/semantic").glob("*.py"))
                                 if p.name not in {"__init__.py", "construction.py", "paths.py"}])
def test_training_and_construction_help_does_not_need_models(path):
    result = subprocess.run([sys.executable, str(path), "--help"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_restored_k400_location_can_be_overridden_without_changing_the_parser(tmp_path):
    import json
    import os

    labels = tmp_path / "k400/labels.json"
    labels.parent.mkdir()
    labels.write_text(json.dumps([{"id": 7, "label": "running"}]))
    code = ("from scripts.semantic import paths; "
            "from vbench_prompts_compile.sources import load_k400; "
            "assert load_k400().resolve('running') == 'running'")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                            text=True, env=dict(os.environ, VBENCH_TRAINING_RAW=str(tmp_path)))
    assert result.returncode == 0, result.stderr


def test_spatial_construction_retains_only_explicit_signed_directions():
    row = {"input": {"prompt": "a cat on top of a box, near a tree"}, "target": {
        "relationships": [{"subject": "cat", "relation": "on", "object": "box"},
                          {"subject": "cat", "relation": "near", "object": "tree"}]}}
    converted = prepare_repair_training.spatial_row(row)
    assert converted["target"]["relationships"] == [
        {"subject": "cat", "relation": "above", "object": "box"}]
    assert converted["meta"]["omitted_unsupported_relations"] == [row["target"]["relationships"][1]]


def test_scene_audit_connects_both_pair_endpoints_and_shared_frames():
    rows = [{"input": {"prompt": "a room"}, "meta": {"frame_source_prompt": "a park"}},
            {"input": {"prompt": "a park"}, "meta": {"image": "frame.png"}},
            {"input": {"prompt": "a beach"}, "meta": {"image": "frame.png"}},
            {"input": {"prompt": "a desert"}}]
    groups = audit_scene_sources.components(rows)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_action_isolation_removes_the_whole_exposed_template_family():
    class Vocabulary:
        entries = [{"id": 7, "label": "running"}]

        def resolve(self, value):
            return value if value == "running" else None

    def row(prompt, group, label_id):
        return {"input": {"prompt": prompt}, "group_id": group,
                "meta": {"k400_id": label_id}}

    splits = {"train": [row("a runner", "template-7", 7),
                        row("a paraphrase", "template-7", None),
                        row("a swimmer", "safe-train", 8)],
              "dev": [row("a diver", "safe-dev", 9)]}
    matrix = [{"prompt": "someone running", "original_prompt": "a runner",
               "official_target": "running", "base_official_target": "running",
               "transform": "identity"}]
    kept, excluded, _ = prepare_action_isolation.isolate(splits, matrix, Vocabulary())
    assert [r["group_id"] for r in kept["train"]] == ["safe-train"]
    assert len(excluded) == 2
    assert kept["dev"] == splits["dev"]


def test_matrix_source_selection_deduplicates_media_and_balances_prompts(tmp_path):
    manifest = tmp_path / "pairs.csv"
    manifest.write_text("prompt_en,video_a_path,video_b_path\na,a0,a1\na,a1,a2\nb,b0,b1\n")
    rows = rows_from_manifest(manifest, per_prompt=2)
    assert [(r["prompt"], r["relative_path"]) for r in rows] == [
        ("a", "a0"), ("b", "b0"), ("a", "a1"), ("b", "b1")]
