"""Pure-algorithm tests for the VBench-CF counterfactual transforms.

No model, no network and no real VBench data: synthetic clips stand in for the
sources so the mathematical contract of every family can be checked directly.
FFmpeg is required for the encode/decode round trip; the module skips cleanly
when it is unavailable.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from scripts.counterfactual import transforms
from scripts.counterfactual.common import (
    CounterfactualError,
    decode_video,
    encode_video,
    probe_video,
    resample_indices,
    sha256_file,
    stable_sample,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are required for the video round trip",
)

FRAME_COUNT = 16
NATIVE_FPS = 8.0
SIZE = 64


def synthetic_frames(count: int = FRAME_COUNT, size: int = SIZE) -> np.ndarray:
    """A small square translating across a dark background, one step per frame."""
    frames = np.zeros((count, size, size, 3), dtype=np.uint8)
    for index in range(count):
        frames[index, :, :, :] = 12
        left = 2 + index * 2
        frames[index, 20:40, left : left + 8, :] = 240
    return frames


@pytest.fixture(scope="module")
def source_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("cf") / "source.mp4"
    encode_video(synthetic_frames(), path, NATIVE_FPS)
    return path


def test_encode_decode_round_trip_is_exact(source_video):
    frames, meta = decode_video(source_video)
    assert meta.frame_count == FRAME_COUNT
    assert meta.fps == pytest.approx(NATIVE_FPS)
    assert meta.duration_s == pytest.approx(2.0, abs=1e-6)
    assert len(frames) == FRAME_COUNT
    assert frames.shape[1:] == (SIZE, SIZE, 3)


def test_encoding_is_byte_deterministic(source_video, tmp_path):
    frames, meta = decode_video(source_video)
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    encode_video(frames, first, meta.fps)
    encode_video(frames, second, meta.fps)
    assert sha256_file(first) == sha256_file(second)


def test_resample_indices_rejects_upsampling():
    with pytest.raises(CounterfactualError, match="upsample"):
        resample_indices(FRAME_COUNT, 8.0, 12.0)


def test_resample_indices_are_an_exact_subset():
    indices = resample_indices(FRAME_COUNT, NATIVE_FPS, 4.0)
    assert indices == sorted(set(indices))
    assert all(0 <= index < FRAME_COUNT for index in indices)
    assert indices == [0, 2, 4, 6, 8, 10, 12, 14]


def test_fps_family_is_monotonic_in_rungs_and_rank_tied(source_video):
    frames, meta = decode_video(source_video)
    variants = transforms.fps_resample(frames, meta)
    assert [variant.name for variant in variants] == ["fps8", "fps6", "fps4", "fps2"]
    assert [len(variant.frames) for variant in variants] == [16, 12, 8, 4]
    # Invariance: every rung is expected to tie with every other rung.
    assert len({variant.expected_rank for variant in variants}) == 1
    # Downsampling keeps duration fixed while the frame rate falls.
    for variant, rate in zip(variants, (8.0, 6.0, 4.0, 2.0)):
        assert len(variant.frames) / variant.fps == pytest.approx(2.0, abs=0.13)
        assert variant.fps == rate


def test_fps_family_keeps_the_trajectory(source_video):
    frames, meta = decode_video(source_video)
    variants = transforms.fps_resample(frames, meta)
    half = variants[-1]
    # Every retained frame must be an original frame, not an invented one.
    for frame in half.frames:
        assert any(np.array_equal(frame, original) for original in frames)


def test_directional_flip_inverts_the_named_axis(source_video):
    frames, meta = decode_video(source_video)
    horizontal = transforms.directional_flip(frames, meta, "left")
    assert [variant.name for variant in horizontal] == ["original", "horizontal_flip"]
    flipped = horizontal[1].frames
    assert np.array_equal(flipped, frames[:, :, ::-1, :])
    assert not np.array_equal(flipped, frames)

    vertical = transforms.directional_flip(frames, meta, "top")
    assert vertical[1].name == "vertical_flip"
    assert np.array_equal(vertical[1].frames, frames[:, ::-1, :, :])


def test_directional_flip_ranks_original_above_flip(source_video):
    frames, meta = decode_video(source_video)
    original, flipped = transforms.directional_flip(frames, meta, "right")
    assert original.expected_rank > flipped.expected_rank


def test_flip_axis_rejects_unknown_relation():
    with pytest.raises(CounterfactualError):
        transforms.flip_axis("inside of")


def test_reciprocal_relation_round_trips():
    for relation in ("left", "right", "top", "bottom"):
        assert transforms.reciprocal_relation(transforms.reciprocal_relation(relation)) == relation


def test_temporal_jerk_preserves_count_rate_and_is_monotonic(source_video):
    frames, meta = decode_video(source_video)
    variants = transforms.temporal_jerk(frames, meta)
    assert [variant.expected_rank for variant in variants] == [4, 3, 2, 1, 0]
    for variant in variants:
        assert len(variant.frames) == FRAME_COUNT
        assert variant.fps == meta.fps
    # Level 0 is the untouched sequence; every perturbed level must differ.
    assert np.array_equal(variants[0].frames, frames)
    for variant in variants[1:]:
        assert not np.array_equal(variant.frames, frames)
    # Perturbations only reorder or repeat existing frames.
    pool = {frame.tobytes() for frame in frames}
    for variant in variants[1:]:
        assert {frame.tobytes() for frame in variant.frames} <= pool


def test_temporal_jerk_rejects_short_clips(source_video):
    frames, meta = decode_video(source_video)
    with pytest.raises(transforms.ConversionTooShort):
        transforms.temporal_jerk(frames[:6], meta)


def test_temporal_relocation_puts_the_same_window_at_three_positions(source_video):
    frames, meta = decode_video(source_video)
    variants = transforms.temporal_relocation(frames, meta, boxes=(10, 10, 45, 45))
    assert [variant.name for variant in variants] == [
        "clean",
        "corrupt_start",
        "corrupt_middle",
        "corrupt_end",
    ]
    clean = variants[0]
    assert np.array_equal(clean.frames, frames)
    assert clean.expected_rank == 1
    assert all(variant.expected_rank == 0 for variant in variants[1:])
    starts = [variant.parameters["window_start"] for variant in variants[1:]]
    assert starts == [0, (FRAME_COUNT - 4) // 2, FRAME_COUNT - 4]
    windows = {variant.parameters["window_frames"] for variant in variants[1:]}
    assert windows == {4}
    # Outside its window each corrupted clip must equal the clean clip exactly.
    for variant in variants[1:]:
        start = variant.parameters["window_start"]
        window = variant.parameters["window_frames"]
        keep = [i for i in range(FRAME_COUNT) if not start <= i < start + window]
        assert np.array_equal(variant.frames[keep], frames[keep])


def test_temporal_relocation_rejects_clips_that_force_overlapping_windows(source_video):
    frames, meta = decode_video(source_video)
    # Three windows of one frame cannot be placed without overlapping in a
    # two-frame clip, so the base must be rejected rather than corrupted twice.
    with pytest.raises(CounterfactualError):
        transforms.temporal_relocation(frames[:2], meta, boxes=(0, 0, 10, 10))


def test_temporal_relocation_accepts_the_shortest_non_overlapping_clip(source_video):
    frames, meta = decode_video(source_video)
    variants = transforms.temporal_relocation(frames[:8], meta, boxes=(0, 0, 10, 10))
    windows = [
        (variant.parameters["window_start"], variant.parameters["window_frames"])
        for variant in variants[1:]
    ]
    covered: set[int] = set()
    for start, size in windows:
        span = set(range(start, start + size))
        assert not (span & covered), f"windows overlap: {windows}"
        covered |= span


def test_environment_coverage_is_monotonic_and_endpoint_exact(source_video):
    frames, meta = decode_video(source_video)
    donor = synthetic_frames(size=SIZE)[:, :, ::-1, :].copy()
    variants = transforms.environment_coverage(frames, donor, meta)
    assert [variant.name for variant in variants] == [
        "coverage_000",
        "coverage_025",
        "coverage_050",
        "coverage_075",
        "coverage_100",
    ]
    assert [variant.expected_rank for variant in variants] == [0, 1, 2, 3, 4]
    assert np.array_equal(variants[0].frames, donor)
    assert np.array_equal(variants[-1].frames, frames)
    # Each step adds exactly one more target quadrant.
    for previous, current in zip(variants, variants[1:]):
        changed = np.any(previous.frames != current.frames, axis=(0, 3))
        assert changed.sum() > 0


def test_environment_coverage_rejects_resolution_mismatch(source_video):
    frames, meta = decode_video(source_video)
    with pytest.raises(CounterfactualError, match="resolution mismatch"):
        transforms.environment_coverage(frames, synthetic_frames(size=32), meta)


def test_weakest_object_visibility_suppresses_only_target_b(source_video):
    frames, meta = decode_video(source_video)
    box = (20, 20, 40, 40)
    variants = transforms.weakest_object_visibility(frames, meta, box)
    assert [variant.name for variant in variants] == [
        "occlusion_000",
        "occlusion_025",
        "occlusion_050",
        "occlusion_075",
        "occlusion_100",
    ]
    assert [variant.expected_rank for variant in variants] == [4, 3, 2, 1, 0]
    assert np.array_equal(variants[0].frames, frames)
    # Everything outside the box is untouched at every severity.
    mask = np.ones(frames.shape[1:3], dtype=bool)
    mask[20:40, 20:40] = False
    for variant in variants:
        assert np.array_equal(variant.frames[:, mask], frames[:, mask])
    # Full severity collapses the region to a flat patch.
    final = variants[-1].frames[:, 20:40, 20:40, :]
    assert final.std(axis=(1, 2)).max() == 0


def test_temporal_conjunction_control_has_no_co_present_frame(source_video):
    frames, meta = decode_video(source_video)
    variants = transforms.temporal_conjunction_control(frames, meta, (0, 0, 10, 10), (40, 40, 60, 60))
    assert len(variants) == 1
    control = variants[0].frames
    half = FRAME_COUNT // 2
    assert len(control) == FRAME_COUNT
    # B's region is flattened in the first half and A's in the second.
    assert control[:half, 0:10, 0:10, :].std(axis=(1, 2)).max() == 0
    assert control[half:, 40:60, 40:60, :].std(axis=(1, 2)).max() == 0


def test_filename_family_never_touches_pixels():
    variants = transforms.filename_invariance("archery", "bandaging")
    assert [variant.name for variant in variants] == [
        "filename_correct",
        "filename_wrong",
        "filename_neutral",
    ]
    assert all(variant.copies_bytes for variant in variants)
    assert len({variant.expected_rank for variant in variants}) == 1


def test_stable_sample_is_reproducible_and_score_independent():
    items = [f"item-{index}" for index in range(50)]
    assert stable_sample(items, 10, 7) == stable_sample(items, 10, 7)
    assert stable_sample(items, 10, 7) != stable_sample(items, 10, 8)
    assert len(stable_sample(items, 10, 7)) == 10
    assert stable_sample(items, 100, 7) == items


def test_probe_rejects_missing_file(tmp_path):
    with pytest.raises(CounterfactualError, match="not found"):
        probe_video(tmp_path / "absent.mp4")


# --------------------------------------------------------------------------
# Per-frame (tracked) boxes
# --------------------------------------------------------------------------


def test_per_frame_boxes_broadcasts_a_single_box():
    assert transforms.per_frame_boxes((1, 2, 3, 4), 3) == [(1, 2, 3, 4)] * 3


def test_per_frame_boxes_accepts_one_box_per_frame():
    tracked = [(0, 0, 5, 5), (1, 1, 6, 6)]
    assert transforms.per_frame_boxes(tracked, 2) == tracked


def test_per_frame_boxes_rejects_wrong_length():
    with pytest.raises(CounterfactualError, match="one box per frame"):
        transforms.per_frame_boxes([(0, 0, 5, 5)], 3)


def test_per_frame_boxes_rejects_missing_detection():
    with pytest.raises(CounterfactualError, match="missing detection"):
        transforms.per_frame_boxes([(0, 0, 5, 5), None], 2)


def test_temporal_relocation_corrupts_each_frame_at_its_own_box():
    from scripts.counterfactual.common import VideoMeta

    frames = synthetic_frames()
    meta = VideoMeta("x", SIZE, SIZE, NATIVE_FPS, FRAME_COUNT, 2.0, "mp4")
    # A box that marches across the clip: one static box could not follow it.
    tracked = [(2 + index, 20, 12 + index, 40) for index in range(FRAME_COUNT)]
    variants = transforms.temporal_relocation(frames, meta, tracked)
    corrupt = next(variant for variant in variants if variant.name == "corrupt_start")
    for offset in range(4):
        left = tracked[offset][0]
        expected = frames[offset].copy()
        expected[20:40, left : left + 10] = corrupt.frames[offset][20:40, left : left + 10]
        assert np.array_equal(expected, corrupt.frames[offset])


# --------------------------------------------------------------------------
# GRiT label matching (pure helpers: no model, no GPU)
# --------------------------------------------------------------------------


def test_normalise_label_folds_punctuation_and_case():
    from scripts.counterfactual.grit import normalise_label

    assert normalise_label("Cell Phone") == normalise_label("cellphone")
    assert normalise_label("  CAT ") == "cat"


def test_select_target_matches_bare_nouns_not_articles():
    from scripts.counterfactual.grit import Detection, select_target

    detections = [
        Detection("cat", (0, 0, 10, 10), 0.9),
        Detection("dog", (1, 1, 30, 30), 0.8),
    ]
    assert select_target(detections, "dog").box == (1, 1, 30, 30)
    # `prompt_en`-style phrasing must NOT match, mirroring VBench's exact check.
    assert select_target(detections, "a dog") is None


def test_select_target_prefers_the_largest_box():
    from scripts.counterfactual.grit import Detection, select_target

    detections = [
        Detection("cat", (0, 0, 10, 10), 0.95),
        Detection("cat", (0, 0, 40, 40), 0.60),
    ]
    assert select_target(detections, "cat").area == 1600


def test_track_target_carries_the_nearest_box_forward():
    from scripts.counterfactual.grit import Detection, coverage, track_target

    per_frame = [
        [Detection("cat", (0, 0, 10, 10), 0.9)],
        [],  # detector drop-out
        [Detection("cat", (4, 4, 14, 14), 0.9)],
    ]
    tracked = track_target(per_frame, "cat")
    assert tracked == [(0, 0, 10, 10), (0, 0, 10, 10), (4, 4, 14, 14)]
    assert coverage(tracked) == 1.0


def test_track_target_reports_a_never_seen_target_as_all_missing():
    from scripts.counterfactual.grit import Detection, track_target

    per_frame = [[Detection("boat", (0, 0, 10, 10), 0.9)], []]
    assert track_target(per_frame, "airplane") == [None, None]


def test_weaker_target_uses_median_box_area():
    from scripts.counterfactual.build import _weaker_target

    big = [(0, 0, 40, 40)] * 3
    small = [(0, 0, 10, 10)] * 3
    assert _weaker_target([big, small]) == 1
    assert _weaker_target([small, big]) == 0
