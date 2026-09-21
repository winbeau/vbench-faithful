"""Shared text/instance boundaries; all examples are explicitly synthetic."""
import json

import pytest

from vbench_audit_models.grit import bind_heads
from vbench_audit_models.labels import LabelVocabulary, compile_prompt, validate_compilation
from vbench_audit_models.prompt_compiler import PromptCompiler


@pytest.fixture
def vocabulary():
    return LabelVocabulary({"objects": ["car", "dog", "couch", "sofa", "orange", "hot dog"],
        "colors": ["red", "blue", "orange", "gray", "crimson", "navy", "maroon"],
        "object_aliases": {"car": ["cars"], "dog": ["dogs"], "couch": ["sofa", "sofas"]},
        "color_aliases": {"gray": ["grey"]}})


def instance(text, box):
    return {"text": text, "box": box, "score": .8}


def test_shared_labels_keep_observed_union_but_resolve_true_synonyms(vocabulary):
    assert "sofa" in vocabulary.objects
    assert vocabulary.object(" SOFAS ") == "couch"
    assert vocabulary.object("vehicle") is None
    assert vocabulary.color("grey") == "gray"
    assert vocabulary.color("crimson") != vocabulary.color("red")
    assert vocabulary.color("navy") != vocabulary.color("blue")


@pytest.mark.parametrize("prompt, expected", [
    ("a red car", {"object": "car", "color": "red"}),
    ("three navy cars", {"object": "car", "color": "navy"}),
    ("a car on a red field", {"object": "car", "color": None}),
    ("a red car and a blue dog", {"object": None, "color": None}),
    ("a car that is not red", {"object": None, "color": None}),
    ("an orange car", {"object": "car", "color": "orange"}),
    ("a hot dog", {"object": "hot dog", "color": None}),
    ("an unknown vehicle", {"object": None, "color": None}),
])
def test_prompt_only_baseline_is_conservative(vocabulary, prompt, expected):
    assert compile_prompt("color", prompt, vocabulary) == expected


@pytest.mark.parametrize("value", [{"object": "car", "status": "ok"}, {"object": []},
                                    {"object": "vehicle"}, {"object": True}])
def test_minimal_schema_rejects_extra_fields_and_nonlabels(vocabulary, value):
    with pytest.raises(ValueError):
        validate_compilation("object_class", value, vocabulary)


def test_two_head_permutation_and_duplicate_boxes_abstain():
    left, right = [0, 0, 10, 10], [20, 20, 30, 30]
    captions = [instance("red car", left), instance("blue dog", right)]
    objects = [instance("car", left), instance("dog", right)]
    assert [p["method"] for p in bind_heads(captions, objects)["pairs"]] == ["verified_index"] * 2
    swapped = bind_heads(captions, objects[::-1])
    assert [p["object"]["text"] for p in swapped["pairs"]] == ["car", "dog"]
    assert [p["method"] for p in swapped["pairs"]] == ["unique_iou"] * 2
    ambiguous = bind_heads(captions[:1]*2, objects[:1]*2)
    assert all(p["object"] is None and p["method"] == "ambiguous" for p in ambiguous["pairs"])
    unmatched = bind_heads(captions, objects[:1])
    assert unmatched["pairs"][1]["method"] == "unmatched"


def test_offline_compilation_keys_only_raw_prompt_and_checks_route(tmp_path, vocabulary):
    vocabulary.provenance = {"sha256": "fixture-vocab"}
    payload = {"dimension": "color", "mode": "lora", "vocabulary_sha256": "fixture-vocab",
               "input_fields": ["prompt"], "model": {"fixture": True},
               "records": [{"prompt": "a red car", "output": {"object": "car", "color": "red"}}]}
    path = tmp_path / "compiled.json"
    path.write_text(json.dumps(payload))
    compiler = PromptCompiler("color", vocabulary, {"kind": "lora", "path": str(path)})
    assert compiler("a red car") == {"object": "car", "color": "red"}
    with pytest.raises(ValueError, match="exact raw prompt"):
        compiler("a blue car")
    with pytest.raises(ValueError, match="route/mode"):
        PromptCompiler("object_class", vocabulary, {"kind": "lora", "path": str(path)})
