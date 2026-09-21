import pytest

from scripts.counterfactual.prepare_background_inputs import resolve_media


def test_scene_media_mapping_preserves_background_identity_and_ignores_labels():
    rows = [{"video_uid": "background_uid", "prompt_id": "an alley", "split": "test",
             "generator": "lavie", "relative_video_path": "cogvideo/scene/一条小巷-2.gif"}]
    scene = [{"prompt_en": "an alley", "videos": {"lavie": "lavie/scene/alley-2.mp4"},
              "human_anno": {"invalid": "this must not be read"}}]
    mapped = resolve_media(rows, scene)[0]
    assert mapped["relative_video_path"] == "lavie/scene/alley-2.mp4"
    assert mapped["manifest_video_path"] == rows[0]["relative_video_path"]
    assert mapped["video_uid"] == "background_uid" and mapped["split"] == "test"
    assert "human_anno" not in mapped


def test_media_mapping_refuses_generator_conflict_and_ambiguity():
    rows = [{"prompt_id": "an alley", "generator": "lavie", "relative_video_path": "alley-0.mp4"}]
    scene = {"prompt_en": "an alley", "videos": {"lavie": "cogvideo/scene/alley-0.gif"}}
    with pytest.raises(ValueError, match="generator"):
        resolve_media(rows, [scene])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_media(rows, [scene, scene])
