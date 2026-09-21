import pytest
from vbench_audit_models.grit import bind_heads
from vbench_audit_models.labels import LabelVocabulary
from color.algorithms import color_support, score_frames, legacy_trace_score


@pytest.fixture
def vocab():
    return LabelVocabulary({"objects": ["car", "dog"], "colors": ["red", "blue", "gray", "crimson", "navy", "maroon"],
                            "object_aliases": {"car": ["cars"]}, "color_aliases": {"gray": ["grey"]}})


def frame(caption, name="car"):
    primary = [{"text": caption, "box": [0, 0, 10, 10], "score": .8}]
    objects = [{**primary[0], "text": name}]
    return {"status": "succeeded", "objects": objects, "primary": primary,
            "binding": bind_heads(primary, objects)}


@pytest.mark.parametrize("caption, expected", [
    ("a red car", True), ("a colored car", False), ("a hundred cars", False),
    ("a car that is not red", False), ("a car near a red dog", False),
    ("a red dog and a blue car", False), ("a crimson car", False),
    ("a car on a red field", False), ("the car is red", True),
])
def test_whole_color_is_bound_to_target(caption, expected, vocab):
    assert color_support(caption, "red", "car", vocab)["supported"] is expected


def test_full_denominator_has_five_levels_and_retains_zero(vocab):
    values = []
    empty = {"status": "succeeded", "objects": [], "primary": [], "binding": bind_heads([], [])}
    for visible in [16, 12, 8, 4, 0]:
        frames = [frame("a red car")] * visible + [empty] * (16-visible)
        repair = score_frames(frames, "car", "red", vocab)
        assert repair["status"] == "succeeded" and repair["frame_count"] == 16
        values.append(repair["score"])
        conditional = score_frames(frames, "car", "red", vocab, variant="binding_lexical")
        assert conditional["score"] == (1 if visible else None)
    assert values == [1, .75, .5, .25, 0]
    missing = score_frames([frame("a car")], "car", "red", vocab)
    assert missing["score"] == 0 and missing["reason_counts"] == {"insufficient_color_evidence": 1}
    runtime = score_frames([frame("a red car"), {"status": "failed"}], "car", "red", vocab)
    assert runtime["score"] is None and runtime["support_lower_bound"] == .5


def test_binding_does_not_borrow_first_class_or_other_instance_color(vocab):
    left, right = [0, 0, 10, 10], [20, 20, 30, 30]
    primary = [{"text": "a blue dog", "box": left, "score": .8},
               {"text": "a red car", "box": right, "score": .7}]
    objects = [{**primary[0], "text": "dog"}, {**primary[1], "text": "car"}]
    evidence = {"status": "succeeded", "binding": bind_heads(primary, objects)}
    assert score_frames([evidence], "car", "red", vocab)["score"] == 1
    assert score_frames([evidence], "dog", "red", vocab)["score"] == 0
    swapped = {"status": "succeeded", "binding": bind_heads(primary[::-1], objects)}
    assert score_frames([swapped], "car", "red", vocab)["score"] == 1


def test_legacy_diagnostic_preserves_substring_and_first_instance_defects(vocab):
    f = frame("a colored car")
    assert legacy_trace_score([f], "a red car", "red")["score"] == 1
    assert score_frames([f], "car", "red", vocab)["score"] == 0
    assert legacy_trace_score([f], "a red cars", "red")["score"] is None


@pytest.mark.parametrize("variant", ["binding", "binding_lexical", "repair"])
def test_summary_declares_the_denominator_actually_scored(vocab, variant):
    from color.runtime import summarize
    from vbench_audit_core.schemas import VideoResult

    empty = {"status": "succeeded", "binding": bind_heads([], [])}
    result = score_frames([frame("a red car"), empty], "car", "red", vocab, variant=variant)
    row = VideoResult("test.mp4", result["status"], result["score"], result)
    summary = summarize("audit", [row])
    if variant == "repair":
        assert summary.aggregate == .5
        assert summary.denominator_kind == "equal_video_mean_of_all_frame_rates"
    else:
        assert summary.aggregate == 1
        assert summary.denominator_kind == "equal_video_mean_of_conditional_rates"
