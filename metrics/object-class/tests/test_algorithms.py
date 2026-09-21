from vbench_audit_models.labels import LabelVocabulary
from vbench_audit_core.schemas import VideoResult
from object_class.algorithms import score_frames
from object_class.runtime import summarize


def test_alias_effect_is_distinct_from_no_detection_and_failure():
    vocab = LabelVocabulary({"objects": ["couch", "dog"], "colors": [],
                             "object_aliases": {"couch": ["sofa"]}})
    instance = {"box": [0, 0, 4, 4], "score": .7, "text": "sofa"}
    frames = [{"status": "succeeded", "objects": [instance, dict(instance)]},
              {"status": "succeeded", "objects": []},
              {"status": "succeeded", "objects": [{**instance, "text": "dog"}]}]
    result = score_frames(frames, "couch", vocab)
    assert result["score"] == 1/3
    assert len(result["frames"][0]["instances"]) == 2
    assert result["frames"][0]["instances"][0]["score"] == .7
    assert result["reason_counts"] == {"exact_miss_alias_hit": 1, "no_detection": 1, "other_class_detected": 1}
    assert score_frames(frames, "couch", vocab, lexical=False)["score"] == 0
    assert score_frames(frames, "COUCH", vocab)["score"] == result["score"]
    failed = score_frames(frames + [{"status": "failed", "objects": []}], "couch", vocab)
    assert failed["score"] is None
    assert failed["frame_count"] == 4 and failed["support_lower_bound"] == .25
    assert failed["reason_counts"]["runtime_failure"] == 1
    assert score_frames(frames, "furniture", vocab)["status"] == "unsupported"


def test_weighted_summary_keeps_incomplete_denominator_visible():
    rows = [VideoResult("a", "succeeded", 1., {"success_frame_count": 4, "frame_count": 4}),
            VideoResult("b", "succeeded", 0., {"success_frame_count": 0, "frame_count": 12})]
    assert summarize("audit", rows).aggregate == .25
    incomplete = summarize("audit", rows + [VideoResult("c", "unsupported")])
    assert incomplete.aggregate is None
    assert incomplete.counts["observed_subset_aggregate_diagnostic_only"] == .25
    assert incomplete.counts["coverage"] == 2/3
