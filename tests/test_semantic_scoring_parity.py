"""Golden CPU parity against the published, pre-integration semantic scorer."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def cases(text_key):
    """Exercise score direction, temporal confirmation, abstention and missingness."""
    frame = [{"label": "cat", "box": [0, 0, 10, 10]},
             {"label": "dog", "box": [20, 0, 30, 10]}]
    examples = []
    for relation in ("left", "right"):
        row = {"task": "spatial", "prompt": f"a cat on the {relation} of a dog",
               "eligible": True, "official_target": {"object_a": "cat", "object_b": "dog",
                                                       "relationship": f"on the {relation} of"}}
        target = {"relationships": [{"subject": "a cat", "relation": relation, "object": "the dog"}]}
        for state, frames in (("complete", [frame] * 16), ("partial", [frame] * 7),
                              ("missing", []), ("empty", [[]] * 16),
                              ("invalid", [[{"label": "cat", "box": [0, 0, 0, 0]}]] * 16)):
            examples.append((f"spatial-{relation}-{state}", row,
                             {"status": "ok", "frame_detections": frames},
                             {text_key("spatial", row["prompt"]): target}))
    row = {"task": "objects", "prompt": "a cat and a dog", "eligible": True,
           "official_target": "cat and dog"}
    for state, frames in (("complete", [frame] * 16), ("isolated", [frame] + [[]] * 15),
                          ("boundary", [frame] + [[]] * 14 + [frame]),
                          ("partial", [frame] * 2), ("missing", [])):
        examples.append((f"objects-{state}", row,
                         {"status": "ok", "frame_detections": frames,
                          "frame_labels": [[item["label"] for item in f] for f in frames]},
                         {text_key("objects", row["prompt"]): {"entities": ["cat", "dog"]}}))
    for target in ({"actions": ["playing violin"]}, {"actions": ["other"]}, None):
        for confidence in (.84994, .84996):
            row = {"task": "action", "prompt": "A person is playing violin", "eligible": True,
                   "official_target": "playing violin"}
            name = "missing" if target is None else target["actions"][0]
            examples.append((f"action-{name}-{confidence}", row,
                             {"status": "ok", "top5": [{"label": "playing violin", "score": confidence}]},
                             {text_key("action", row["prompt"]): target}))
    row = {"task": "scene", "prompt": "a sea", "eligible": True, "official_target": "sea"}
    for label in ("supported", "contradicted", "insufficient", None):
        examples.append((f"scene-{label}", row,
                         {"status": "partial", "frame_captions": ["ocean"] * 8},
                         {text_key("scene", row["prompt"], "ocean"): label}))
    examples.append(("ineligible", {**row, "eligible": False}, {}, {}))
    return {f"{name}/{scheme}": {"row": row, "scheme": scheme, "evidence": evidence,
                                  "predictions": predictions}
            for name, row, evidence, predictions in examples
            for scheme in ("Origin", "Repair-rule", "Repair-model")}


def result_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


GOLDEN = json.loads((ROOT / "tests/fixtures/semantic-scoring-golden.json").read_text())


@pytest.mark.parametrize("name", GOLDEN["cases"])
def test_integrated_scorer_matches_published_result_and_status(name):
    from vbench_prompts_compile.experiments import text_key
    from vbench_prompts_compile.scoring import score_one
    from vbench_prompts_compile.sources import load_k400

    example = cases(text_key)[name]
    actual = score_one(**example, vocab=load_k400(ROOT / "configs/reproduction/k400-labels.json"))
    expected = GOLDEN["cases"][name]
    assert actual["score"] == expected["score"]
    assert actual["status"] == expected["status"]
    assert actual["coverage"] == expected["coverage"]
    assert result_hash(actual) == expected["result_sha256"]
