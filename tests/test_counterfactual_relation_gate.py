"""Contract tests for the `directional_flip` eligibility gate.

The metamorphic expectation `score(original) > score(flip)` is only meaningful
when the source clip actually satisfies the ordered relation.  Plan section 9.2
requires that confirmation; the counterfactual build originally skipped it, so
every direction-sensitive metric was being scored against a coin flip.  These
tests pin the gate that now enforces it.

No model and no ffmpeg: the detector and the decoder are stubbed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.counterfactual import pick_detectable
from scripts.counterfactual.grit import Detection


class StubDetector:
    """Returns a canned detection list per frame index."""

    def __init__(self, per_frame):
        self.per_frame = per_frame

    def detect(self, frame):
        return self.per_frame[frame]


def _boxes(labels_and_boxes):
    return [Detection(label=label, box=box, score=0.9) for label, box in labels_and_boxes]


# `cat` on the left of `dog`; x grows to the right.
CAT_LEFT = _boxes([("cat", (0, 0, 2, 2)), ("dog", (8, 0, 10, 2))])
CAT_RIGHT = _boxes([("cat", (8, 0, 10, 2)), ("dog", (0, 0, 2, 2))])
ONLY_CAT = _boxes([("cat", (0, 0, 2, 2))])


def test_relation_frame_rate_counts_only_both_detected_frames():
    detections = [CAT_LEFT, CAT_RIGHT, ONLY_CAT, CAT_LEFT]
    rate, scored = pick_detectable.relation_frame_rate(detections, "cat", "dog", "on the left of")
    # The third frame is skipped rather than counted as a violation.
    assert scored == 3
    assert rate == pytest.approx(2 / 3)


def test_relation_frame_rate_is_directional():
    left, _ = pick_detectable.relation_frame_rate([CAT_LEFT], "cat", "dog", "on the left of")
    right, _ = pick_detectable.relation_frame_rate([CAT_LEFT], "cat", "dog", "on the right of")
    assert left == 1.0
    assert right == 0.0


def test_relation_frame_rate_is_zero_when_nothing_is_scoreable():
    rate, scored = pick_detectable.relation_frame_rate([ONLY_CAT, ONLY_CAT], "cat", "dog", "on the left of")
    assert (rate, scored) == (0.0, 0)


def test_detector_verdict_rejects_a_base_below_the_threshold(monkeypatch):
    monkeypatch.setattr(pick_detectable, "decode_video", lambda path: (list(range(4)), None))
    detector = StubDetector([CAT_LEFT, CAT_LEFT, CAT_RIGHT, ONLY_CAT])
    base = {"video_uid": "v1", "relative_video_path": "x.mp4"}
    ok, why, stats = pick_detectable.detector_verdict(
        detector, base, Path("/data"), ["cat", "dog"], ("cat", "dog", "on the left of"), 0.75, 0.25
    )
    assert not ok
    assert "holds in only" in why
    assert stats["frame_rate"] == pytest.approx(2 / 3)
    assert stats["frames_scored"] == 3


def test_detector_verdict_accepts_a_base_above_the_threshold(monkeypatch):
    monkeypatch.setattr(pick_detectable, "decode_video", lambda path: (list(range(4)), None))
    detector = StubDetector([CAT_LEFT, CAT_LEFT, CAT_LEFT, CAT_LEFT])
    base = {"video_uid": "v1", "relative_video_path": "x.mp4"}
    ok, why, stats = pick_detectable.detector_verdict(
        detector, base, Path("/data"), ["cat", "dog"], ("cat", "dog", "on the left of"), 0.75, 0.25
    )
    assert (ok, why) == (True, "")
    assert stats["frame_rate"] == 1.0


def test_detector_verdict_still_rejects_an_untrackable_target(monkeypatch):
    monkeypatch.setattr(pick_detectable, "decode_video", lambda path: (list(range(2)), None))
    detector = StubDetector([ONLY_CAT, ONLY_CAT])
    base = {"video_uid": "v1", "relative_video_path": "x.mp4"}
    ok, why, _ = pick_detectable.detector_verdict(
        detector, base, Path("/data"), ["cat", "dog"], None, 0.75, 0.25
    )
    assert not ok
    assert "not tracked" in why


def test_box_presence_alone_does_not_require_a_relation(monkeypatch):
    monkeypatch.setattr(pick_detectable, "decode_video", lambda path: (list(range(2)), None))
    detector = StubDetector([CAT_LEFT, CAT_LEFT])
    base = {"video_uid": "v1", "relative_video_path": "x.mp4"}
    assert pick_detectable.detector_verdict(
        detector, base, Path("/data"), ["cat", "dog"], None, 0.75, 0.25
    ) == (True, "", None)


def test_load_relation_validity_accepts_a_mapping_or_a_list(tmp_path):
    mapping = tmp_path / "rates.json"
    mapping.write_text(json.dumps({"b1": 0.9, "b2": None}), encoding="utf-8")
    assert pick_detectable.load_relation_validity(mapping) == {"b1": 0.9, "b2": None}
    listed = tmp_path / "ids.json"
    listed.write_text(json.dumps(["b1", "b2"]), encoding="utf-8")
    assert pick_detectable.load_relation_validity(listed) == {"b1": None, "b2": None}


def test_human_verdict_rejects_an_unlisted_base():
    ok, why, stats = pick_detectable.human_verdict({"base_id": "b9"}, {"b1": 0.8})
    assert not ok
    assert "not in the human validity list" in why
    assert stats is None


def test_detector_verdict_applies_the_co_detection_floor(monkeypatch):
    """A base cannot pass on a single lucky frame."""

    frames = [CAT_LEFT, CAT_LEFT, ONLY_CAT, ONLY_CAT, ONLY_CAT, ONLY_CAT]
    monkeypatch.setattr(pick_detectable, "decode_video", lambda path: (list(range(len(frames))), None))
    detector = StubDetector(frames)
    base = {"video_uid": "v1", "relative_video_path": "x.mp4"}
    ok, why, stats = pick_detectable.detector_verdict(
        detector, base, Path("/data"), ["cat", "dog"], ("cat", "dog", "on the left of"), 0.75, 0.5
    )
    assert not ok
    assert "detected natively in only" in why
    assert stats["frames_scored"] == 2


def test_spatial_targets_are_the_bare_nouns(monkeypatch):
    monkeypatch.setattr(
        pick_detectable, "_load_annotations",
        lambda path: [{
            "prompt_en": "a fire hydrant on the right of a stop sign, front view",
            "object_a_en": "fire hydrant",
            "object_b_en": "stop sign",
            "relationship_en": "right",
        }],
    )
    base = {"prompt_en": "a fire hydrant on the right of a stop sign, front view", "parsed": {"relation": "right"}}
    assert pick_detectable.detectable_targets("spatial_relationship", base, Path("/data")) == ["fire hydrant", "stop sign"]
    assert pick_detectable.spatial_query(base, Path("/data")) == ("fire hydrant", "stop sign", "on the right of")


def test_spatial_annotation_without_bare_nouns_is_rejected(monkeypatch):
    monkeypatch.setattr(
        pick_detectable, "_load_annotations",
        lambda path: [{"prompt_en": "p", "object_a_en": "", "object_b_en": "car", "relationship_en": "left"}],
    )
    with pytest.raises(ValueError, match="bare object nouns"):
        pick_detectable.spatial_query({"prompt_en": "p", "parsed": {"relation": "left"}}, Path("/data"))
