from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.counterfactual.static_jitter import build, export_prepared, merge_builds, motion_ladder, perturb, select_sources, variants

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/dynamic-static-jitter/construction.dev-v1.json").read_text())


@pytest.mark.parametrize("family", CONFIG["families"])
def test_appearance_only_noise_replays_and_preserves_grid(family):
    frames = np.full((16, 192, 256, 3), 120, np.uint8)
    result, info = perturb(frames, family, 20, 1701)
    again, replay = perturb(frames, family, 20, 1701)
    assert np.array_equal(result, again) and info == replay
    assert result.shape == frames.shape and info["coordinate_map"] == "identity"
    assert abs(result.astype(float) - frames).max() <= 20
    assert info["preencode_difference"] > 0


def test_zero_dose_and_clean_identity():
    frames = np.full((16, 192, 256, 3), 120, np.uint8)
    for family in CONFIG["families"] + ["clean"]:
        assert np.array_equal(perturb(frames, family, 0, 1701)[0], frames)


def test_motion_stationary_control_and_oscillation_are_real_geometric_changes():
    image = np.random.default_rng(4).integers(20, 220, (192, 256, 3), dtype=np.uint8)
    still = motion_ladder(image, 0, CONFIG)
    moving = motion_ladder(image, 4, CONFIG)
    reversal = motion_ladder(image, 4, CONFIG, oscillating=True)
    assert np.array_equal(still[0], still[-1])
    assert not np.array_equal(moving[0], moving[-1])
    assert np.array_equal(reversal[0], reversal[8])
    assert np.array_equal(still[0], moving[0])


def test_sources_are_prompt_disjoint_balanced_and_deterministic():
    rows = select_sources(ROOT / "data/processed/e0_scoring_manifest.csv", CONFIG)
    assert len(rows) == 152
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    assert len(dev) == 32 and len(test) == 120
    assert not {r["prompt_id"] for r in dev} & {r["prompt_id"] for r in test}
    for prompt in {r["prompt_id"] for r in rows}:
        assert len([r for r in rows if r["prompt_id"] == prompt]) == 4


def test_every_motion_level_has_matched_interventions():
    records = list(variants(CONFIG))
    static = {(r["family"], r["amplitude"], r["seed"]) for r in records if r["kind"] == "static"}
    for speed in CONFIG["motion_pixels_per_frame"]:
        control = {(r["family"], r["amplitude"], r["seed"]) for r in records if r["kind"] == "translation" and r["motion"] == speed}
        assert control == static


def test_missing_encoder_fails_before_starting_any_construction(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.counterfactual.static_jitter.shutil.which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="no construction started"):
        build(SimpleNamespace(ffmpeg="missing", output=str(tmp_path / "not-created")))
    assert not (tmp_path / "not-created").exists()


def test_merge_rejects_duplicate_sources_and_preserves_rejected_candidates(tmp_path):
    paths = []
    expected = len(list(variants(CONFIG)))
    for ordinal in range(2):
        shard = tmp_path / str(ordinal)
        shard.mkdir()
        metadata = {"config": CONFIG, "config_sha256": "same", "sources_sha256": "same-sources",
                    "split": "dev", "controls": "all", "source_count": 1}
        (shard / "construction.json").write_text(json.dumps(metadata))
        rows = [{"candidate_id": f"video_{ordinal * expected + j:06d}", "base_id": f"base-{ordinal}",
                 "status": "rejected", "qualified": False, "reason": "quality", "video": "unused"}
                for j in range(expected)]
        path = shard / "candidates.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        paths.append(path)
    merge_builds(paths, tmp_path / "merged")
    assert len((tmp_path / "merged/candidates.jsonl").read_text().splitlines()) == expected * 2
    with pytest.raises(ValueError, match="source duplicated"):
        merge_builds([paths[0], paths[0]], tmp_path / "invalid")


def test_prepared_export_keeps_source_without_image_as_explicit_error(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "candidates.jsonl"
    path.write_text(json.dumps({"base_id": "base", "reason": "source decode failed"}) + "\n")
    export_prepared([path], tmp_path / "prepared")
    row = json.loads((tmp_path / "prepared/sources.jsonl").read_text())
    assert row["base_id"] == "base" and row["preparation_error"] == "source decode failed"
