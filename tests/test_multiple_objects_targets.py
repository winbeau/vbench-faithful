"""The selected repair checks every parsed target, independently of official's pair."""
import pytest

from vbench_prompts_compile.experiments import text_key
from vbench_prompts_compile.scoring import score_one


def evaluate(entities, frames, scheme="Repair-model", official="cup and dining table"):
    prompt = "A cup, a dining table, and all additional named objects are visible."
    return score_one(
        row={"task": "objects", "prompt": prompt, "eligible": True, "official_target": official},
        scheme=scheme,
        evidence={"status": "ok", "frame_detections": frames,
                  "frame_labels": [[item["label"] for item in frame] for frame in frames]},
        predictions={text_key("objects", prompt): {"entities": entities}},
        vocab=None,
    )


def detection_frame(entities):
    return [{"label": name, "box": [20 * i, 0, 20 * i + 10, 10]}
            for i, name in enumerate(entities)]


@pytest.mark.parametrize("entities", [
    ["cup", "dining table", "vase"],
    ["cup", "dining table", "vase", "bottle"],
])
def test_all_parsed_objects_are_required_beyond_the_official_pair(entities):
    original = [detection_frame(entities)] * 16
    counterfactual = [detection_frame(entities[:-1])] * 16
    assert evaluate(entities, original)["score"] == 1.0
    repaired = evaluate(entities, counterfactual)
    assert repaired["backend_target"]["entities"] == entities
    assert repaired["score"] == 0.0
    assert repaired["coverage"] == 1.0
    assert repaired["status"] == "ok"
    assert evaluate(entities, original, "Origin")["score"] == 1.0
    assert evaluate(entities, counterfactual, "Origin")["score"] == 1.0


def test_changing_official_metadata_cannot_narrow_the_repair_targets():
    entities = ["cup", "dining table", "vase"]
    frames = [detection_frame(entities[:2])] * 16
    first = evaluate(entities, frames)
    second = evaluate(entities, frames, official="cup and vase")
    assert first == second
    assert first["score"] == 0.0


def test_three_object_missing_evidence_keeps_the_planned_denominator():
    entities = ["cup", "dining table", "vase"]
    result = evaluate(entities, [detection_frame(entities)] * 8)
    assert result["score"] == 0.5
    assert result["coverage"] == 0.5
    assert result["missing"] == 0.5
    assert result["status"] == "partial"


def test_third_object_still_requires_adjacent_frame_confirmation():
    entities = ["cup", "dining table", "vase"]
    frames = [detection_frame(entities[:2])] * 16
    frames[8] = detection_frame(entities)
    result = evaluate(entities, frames)
    assert result["score"] == 0.0
    assert result["objects_resolution"]["frame_entity_states"][8]["vase"] == "unconfirmed"
    assert result["abstention"] == 1 / 16
