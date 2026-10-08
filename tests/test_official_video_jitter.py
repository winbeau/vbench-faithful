import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.counterfactual import official_video_jitter as native
from scripts.counterfactual.static_jitter import SourceRejected, decode


CONFIG = json.loads((Path(__file__).resolve().parents[1] /
                     "tests/fixtures/construction/construction.official-dev-v1.json").read_text())


def sequence():
    # Synthetic fixture only, never part of the scientific source manifest.
    image = np.random.default_rng(12).integers(30, 220, (32, 48, 3), dtype=np.uint8)
    return np.stack([np.roll(image, t, axis=1) for t in range(6)])


def test_native_validation_rejects_staticization_resizing_and_changed_time():
    original = sequence()
    pts = (np.arange(6) / 10).tolist()
    check = lambda candidate, timeline=pts: native.validate_native(
        candidate, original, original, timeline, 10, pts, 10, "encoding_control",
        {"clipped_fraction": 0}, CONFIG)
    assert check(original)["qualified"]
    static = np.repeat(original[:1], len(original), axis=0)
    assert "lossless_pixel_verification_failed" in check(static)["reason"]
    assert check(original[:, ::2, ::2])["reason"] == "native_shape_changed"
    assert "native_timeline_changed" in check(original, (np.arange(6) / 8).tolist())["reason"]


def test_native_loader_never_resamples_nonuniform_timeline(monkeypatch, tmp_path):
    monkeypatch.setattr(native, "decode", lambda _: (sequence(), [0, .1, .2, .4, .5, .6], 10))
    with pytest.raises(SourceRejected, match="not resampled"):
        native.native_video(tmp_path / "clip.mp4", 1e-6)


def test_original_and_zero_edit_control_are_separate_from_noise():
    specs = list(native.interventions(CONFIG))
    assert len(specs) == 38
    assert [r["family"] for r in specs[:2]] == ["original", "encoding_control"]
    assert all("motion" not in row and "frames" not in row for row in specs)


def test_real_lossless_encoding_preserves_all_moving_frames_and_native_timing(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("existing local ffmpeg unavailable")
    original = sequence()
    path = tmp_path / "native.mp4"
    native.encode_lossless(path, original, 10, ffmpeg)
    actual, pts, fps = native.native_video(path, 1e-6)
    assert np.array_equal(actual, original)
    assert fps == 10 and np.allclose(pts, np.arange(6) / 10)
    with pytest.raises(FileExistsError):
        native.encode_lossless(path, original, 10, ffmpeg)


def test_builder_scores_original_mp4_directly_and_preserves_source_bytes(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("existing local ffmpeg unavailable")
    source = tmp_path / "original.mp4"
    native.encode_lossless(source, sequence(), 10, ffmpeg)
    original_bytes = source.read_bytes()
    sources = tmp_path / "sources.jsonl"
    sources.write_text(json.dumps({"base_id": "base", "video_uid": "base", "split": "dev", "prompt_id": "p",
                                   "relative_video_path": source.name}) + "\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**CONFIG, "families": ["pixel_rgb"], "amplitudes": [8], "noise_seeds": [1701]}))
    output = tmp_path / "output"
    native.build(SimpleNamespace(ffmpeg=ffmpeg, sources=str(sources), config=str(config), split="dev",
                                source_start=0, limit=None, data_root=str(tmp_path), output=str(output)))
    rows = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["status"] == "qualified" for r in rows)
    assert rows[0]["video"] == str(source) and source.read_bytes() == original_bytes
    assert rows[0]["source_format_adaptation"] == "none"
    assert rows[1]["decoded_difference_from_original"] == 0
    assert all(r["frame_map"] == "identity_all_source_frames" and r["coordinate_map"] == "identity" for r in rows)
    decoded, _, _ = decode(Path(rows[2]["video"]))
    assert decoded.shape == sequence().shape


def test_local_builder_records_displacement_evidence_and_exact_pixels(tmp_path):
    from scripts.counterfactual.local_texture_jitter import local_texture_jitter
    from scripts.counterfactual.analyze_official_video_jitter import validate_population

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("existing local ffmpeg unavailable")
    config = json.loads((Path(__file__).resolve().parents[1] /
                       "tests/fixtures/construction/construction.local-texture-dev5-v1.json").read_text())
    config.update(amplitudes=[2], seeds=[1701])
    source = tmp_path / "original.mp4"
    native.encode_lossless(source, sequence(), 10, ffmpeg)
    original_bytes = source.read_bytes()
    sources = tmp_path / "sources.jsonl"
    sources.write_text(json.dumps({"base_id": "base", "video_uid": "base", "split": "dev", "prompt_id": "p",
                                   "relative_video_path": source.name}) + "\n")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "output"
    native.build(SimpleNamespace(ffmpeg=ffmpeg, sources=str(sources), config=str(config_path), split="dev",
                                source_start=0, limit=None, data_root=str(tmp_path), output=str(output)))
    rows = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["status"] == "qualified" for r in rows)
    assert len(validate_population(rows, config)) == 1
    assert rows[0]["video"] == str(source) and source.read_bytes() == original_bytes
    changed = rows[2]
    assert changed["coordinate_map"] == "bounded_local_displacement_field"
    assert changed["geometry_qualified"] and not changed["intensity_noise_added"]
    assert Path(changed["displacement_evidence"]).is_file()
    intended, _, field = local_texture_jitter(sequence(), 2, 1701, config)
    decoded, _, _ = decode(Path(changed["video"]))
    assert np.array_equal(intended, decoded)
    with np.load(changed["displacement_evidence"]) as saved:
        assert np.array_equal(saved["displacement_xy"], field["displacement_xy"])
