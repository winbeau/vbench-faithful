import numpy as np
import pytest

from scripts.counterfactual.probe_line_structure import inspect_video, main


def test_no_structure_keeps_full_phases_explicit_missing_and_null_score():
    result = inspect_video(np.zeros((6, 32, 32, 3), np.uint8), np.arange(6) / 8)
    assert result["score"] is None and result["status"] == "diagnostic_only"
    assert [(p["start"], p["lag"]) for p in result["pairs"]] == [(i, lag) for lag in (1, 2, 3, 4) for i in range(6 - lag)]
    assert all(p["junctions"]["matches"] == [] and p["lines"]["matches"] == [] for p in result["pairs"])
    assert all(p["seconds"] == p["lag"] / 8 for p in result["pairs"])
    assert len(result["frames"]) == 6


def test_invalid_shard_rejected_before_reading_inputs():
    with pytest.raises(SystemExit) as exc:
        main(["--manifest", "absent", "--review", "absent", "--output", "absent", "--shard", "4", "--shards", "4"])
    assert exc.value.code == 2


def test_ambiguous_time_is_not_silently_resampled():
    with pytest.raises(ValueError):
        inspect_video(np.zeros((3, 32, 32, 3), np.uint8), [0, 0, .125])
