import json
from types import SimpleNamespace

import pytest

from scripts.counterfactual.label_subject_silver import TeacherIdentityDrift, check_teacher_identity
from scripts.counterfactual.subject_semantics import (
    ReusedSubjectHead, prepare_prompt_pool, reviewed_agreement, text_request, validate_label,
)


LABEL = {"subject": "person", "phrase": "person in red jacket", "count": 1, "status": "ok"}
PROMPT = "A person in red jacket walks through a garden."
VOCABULARY = {"person", "garden"}


def test_fixed_schema_extractive_phrase_and_no_count_coercion():
    assert validate_label(LABEL, PROMPT, VOCABULARY) == LABEL
    for change in ({"phrase": "person in a blue jacket"}, {"count": True}, {"subject": "man"}, {"extra": "forbidden"}):
        with pytest.raises(ValueError):
            validate_label({**LABEL, **change}, PROMPT, VOCABULARY)
    with pytest.raises(ValueError, match="extractive"):
        validate_label(LABEL, PROMPT, VOCABULARY, original_prompt="A person walks.")


def test_non_ok_is_explicit_unsupported_and_text_payload_is_whitelisted():
    value = {"subject": None, "phrase": "", "count": None, "status": "ambiguous"}
    assert validate_label(value, PROMPT, VOCABULARY) == value
    system, user = text_request(PROMPT, VOCABULARY)
    assert "JSON" in system
    assert json.loads(user) == {"prompt": PROMPT}


def test_source_groups_and_duplicate_moviegen_rows_cannot_cross_split():
    inputs = [{"prompt": f"A person walking on path {i}.", "source": "moviegen", "source_group": f"m{i}"} for i in range(180)]
    inputs.extend([
        {"prompt": "A person walking on path 0.", "source": "training", "source_group": "video0", "seen_in_training": True},
        {"prompt": "A person on the same video.", "source": "training", "source_group": "video0", "seen_in_training": True},
    ])
    pool, review = prepare_prompt_pool(inputs, review_size=150)
    assert len(review) == 150
    assert all(row["split"] == "test" and not row["seen_in_existing_training"] for row in review)
    members = [row for row in pool if "video0" in row["source_group_ids"]]
    assert len(members) == 2
    assert len({row["group_id"] for row in members}) == 1
    assert {row["split"] for row in members} == {"train"}


def reviewed_rows():
    human, silver = [], []
    for i in range(150):
        row = {"sample_id": str(i), "group_id": f"g{i}", "prompt": PROMPT, "split": "test", "label": dict(LABEL)}
        human.append({**row, "reviewed": True, "reviewer": "fixture", "quality": "human_reviewed", "seen_in_existing_training": False})
        silver.append({**row, "quality": "silver"})
    return human, silver


def test_accuracy_requires_actual_human_review_and_invalid_teacher_outputs_count():
    human, silver = reviewed_rows()
    human[0]["reviewed"] = False
    with pytest.raises(ValueError, match="human reviews"):
        reviewed_agreement(human, silver, VOCABULARY)
    human[0]["reviewed"] = True
    silver[0].pop("label")
    report = reviewed_agreement(human, silver, VOCABULARY)
    assert report["llm_human_exact_agreement"] == pytest.approx(149 / 150)
    assert report["invalid_teacher_outputs"] == 1
    assert report["head_accuracy"] == "NOT RUN"
    with pytest.raises(ValueError, match="exact human-review subset"):
        reviewed_agreement(human, silver, VOCABULARY, predictions=silver[:-1])


def test_teacher_model_fingerprint_drift_fails_closed():
    config = {"expected_response_model": "m", "expected_system_fingerprint": "pinned"}
    with pytest.raises(TeacherIdentityDrift):
        check_teacher_identity(SimpleNamespace(model="m", system_fingerprint="changed", finish_reason="stop"), config)


def test_subject_head_reuses_existing_backbone_and_restores_the_adapter():
    calls = []

    class Model:
        active_adapter = "spatial"

        def load_adapter(self, path, **kwargs):
            calls.append((path, kwargs))

        def set_adapter(self, name):
            self.active_adapter = name

    model = Model()

    def generate(user, max_new_tokens):
        assert model.active_adapter == "subject"
        assert PROMPT in user
        return json.dumps(LABEL)

    head = ReusedSubjectHead(SimpleNamespace(model=model, generate=generate), "subject-adapter", VOCABULARY)
    assert head.predict(PROMPT) == LABEL
    assert model.active_adapter == "spatial"
    assert calls == [("subject-adapter", {"adapter_name": "subject", "is_trainable": False})]
