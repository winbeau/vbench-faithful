"""Do not let numerical tolerance hide changed coverage or text evidence."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_speed32 import numerical_differences


def test_numeric_roundoff_is_recorded_but_evidence_and_structure_remain_strict():
    old = [{"status": "succeeded", "score": .5, "raw": "target", "diagnostics": [1., None]}]
    new = [{**old[0], "score": .5000001}]
    differences = numerical_differences(old, new)
    assert len(differences) == 1 and differences[0]["path"] == "/0/score"
    for bad in ([{**new[0], "raw": "other"}], [{**new[0], "status": "failed"}], [],
                [{**new[0], "score": .50001}], [{**new[0], "score": float("nan")}],
                [{**new[0], "unexpected": True}]):
        with pytest.raises(AssertionError):
            numerical_differences(old, bad)
    with pytest.raises(AssertionError):
        numerical_differences(old, new, tolerance=0)
