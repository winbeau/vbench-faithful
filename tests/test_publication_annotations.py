import pytest

from scripts.publish_dimension_origins import normalize_pairs, safe_relative


def test_pair_orientation_and_ties_are_preserved():
    rows = [{"prompt_en": "a cat", "videos": {"modelscope": "modelscope/a.mp4", "lavie": "lavie/a.mp4"},
             "human_anno": {"modelscope": {"lavie": 0}, "lavie": {"modelscope": 1}}},
            {"prompt_en": "a dog", "videos": {"modelscope": "modelscope/b.mp4", "lavie": "lavie/b.mp4"},
             "human_anno": {"modelscope": {"lavie": 0.5}, "lavie": {"modelscope": 0.5}}}]
    available = {p for r in rows for p in r["videos"].values()}
    pairs, issues = normalize_pairs(rows, "scene", available)
    assert not issues
    assert pairs[0]["generator_a"] == "lavie"
    assert pairs[0]["label_a_over_b"] == 1
    assert pairs[1]["label_a_over_b"] == 0.5
    assert all(p["usable_exact_pair"] and p["reciprocal_labels"] for p in pairs)


def test_wrong_generator_and_missing_paths_are_not_repaired_by_guessing():
    rows = [{"videos": {"lavie": "cogvideo/scene/巷子.gif", "cogvideo": "cogvideo/scene/an alley.mp4"},
             "human_anno": {"lavie": {"cogvideo": 1}, "cogvideo": {"lavie": 0}}}]
    pairs, issues = normalize_pairs(rows, "background_consistency", {"cogvideo/scene/巷子.gif", "cogvideo/scene/alley.gif"})
    assert len(issues) == 2
    assert pairs[0]["video_a"] is None and pairs[0]["video_b"] is None
    assert not pairs[0]["usable_exact_pair"]
    assert rows[0]["videos"]["lavie"] == "cogvideo/scene/巷子.gif"


def test_shared_videos_do_not_share_dimension_labels():
    videos = {"lavie": "lavie/subject_consistency/a.mp4", "modelscope": "modelscope/subject_consistency/a.mp4"}
    first = {"videos": videos, "human_anno": {"lavie": {"modelscope": 1}, "modelscope": {"lavie": 0}}}
    second = {"videos": videos, "human_anno": {"lavie": {"modelscope": 0}, "modelscope": {"lavie": 1}}}
    motion, _ = normalize_pairs([first], "motion_smoothness", set(videos.values()))
    dynamic, _ = normalize_pairs([second], "dynamic_degree", set(videos.values()))
    assert motion[0]["label_a_over_b"] == 1
    assert dynamic[0]["label_a_over_b"] == 0
    assert motion[0]["video_a"] != dynamic[0]["video_a"]


@pytest.mark.parametrize("path", ["../video.mp4", "/tmp/video.mp4", "a/../../video.mp4"])
def test_source_paths_cannot_escape_the_archive(path):
    with pytest.raises(ValueError):
        safe_relative(path)
