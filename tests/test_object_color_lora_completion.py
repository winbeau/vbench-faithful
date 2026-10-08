from copy import deepcopy

import pytest

from scripts.complete_object_color_lora_cf import select_endpoints
from scripts.reproduce_main_table import CELLS, scalar, summarize


def object_pair():
    base = {"base_id": "a", "split": "test", "variant": "canonical", "prompt": "a cat",
            "source_video_sha256": "same", "dimension_metadata": {"object_class": {"object": "cat"}},
            "query_uid": "a:canonical", "video": "video_00000.mp4"}
    cf = {**deepcopy(base), "variant": "uppercase", "query_uid": "a:uppercase", "video": "video_00001.mp4",
          "dimension_metadata": {"object_class": {"object": "CAT"}}}
    return [base, cf]


def test_select_only_frozen_primary_test_endpoints():
    rows = object_pair()
    rows += [{**rows[0], "split": "dev"}, {**rows[0], "variant": "absent"}]
    selected, pairs = select_endpoints(rows, "object_class", 1)
    assert len(selected) == 2
    assert pairs == [{"base_id": "a", "base_query_uid": "a:canonical", "cf_query_uid": "a:uppercase"}]


@pytest.mark.parametrize("kind", ["duplicate", "missing", "prompt", "sha", "target", "filename"])
def test_reject_silent_pair_or_intervention_changes(kind):
    rows = object_pair()
    if kind == "duplicate":
        rows.append(rows[0])
    elif kind == "missing":
        rows.pop()
    elif kind == "prompt":
        rows[1]["prompt"] = "another prompt"
    elif kind == "sha":
        rows[1]["source_video_sha256"] = "different"
    elif kind == "target":
        rows[1]["dimension_metadata"]["object_class"]["object"] = "DOG"
    else:
        rows[1]["video"] = rows[0]["video"]
    with pytest.raises(ValueError):
        select_endpoints(rows, "object_class", 1)


def test_color_keeps_both_endpoints_including_official_failures():
    rows = [{"base_id": "a", "split": "test", "family": "color_visibility_denominator",
             "visible_fraction": level, "query_uid": f"a:{level}", "video": f"video_{i:05}.mp4",
             "prompt": "a white car", "dimension_metadata": {"color": {"object": "car", "color": "white"}}}
            for i, level in enumerate((1.0, 0.75, 0.5, 0.25, 0.0))]
    selected, _ = select_endpoints(rows, "color", 1)
    assert [r["visible_fraction"] for r in selected] == [1.0, 0.0]
    with pytest.raises(ValueError, match="cohort size"):
        select_endpoints(rows, "color", 2)


def test_shared_denominator_never_imputes_official_missing_as_zero():
    complete = {"sample_id": "complete", "primary": True,
                "scores": {key: {"status": "succeeded", "score": 0.5} for key in CELLS}}
    missing = deepcopy(complete)
    missing["sample_id"] = "missing"
    missing["scores"]["origin_cf"] = scalar({"status": "dropped_by_official", "score": None})
    missing["scores"]["repair_cf"]["score"] = 0.0
    summary = summarize([complete, missing])
    assert summary["planned_primary"] == 2
    assert summary["complete_pairs"] == 1
    assert summary["repair_cf"] == 0.5
    assert missing["scores"]["origin_cf"]["score"] is None
