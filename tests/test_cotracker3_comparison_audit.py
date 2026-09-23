import numpy as np
import pytest

from scripts.counterfactual.audit_cotracker3_comparison import pair_values, motion_summary, verify_arrays


def fixture():
    times = np.arange(16)/8
    xy = np.array([[10., 20.], [30., 40.]])
    phases = [{"tracks": np.broadcast_to(xy, (16,2,2)).copy()+np.arange(16)[:, None, None]*np.array([1.,0.]),
               "visible": np.ones((16,2),bool), "timestamps": times, "shape": np.array([16,256,256,3]),
               "queries_txy": np.c_[np.full(2,t), xy+[t,0]]} for t in range(16)]
    return phases


def test_native_speed_keeps_all_phases_and_real_linear_motion():
    result = motion_summary(fixture())
    assert result["score"] is None
    assert [r["native_start_count"] for r in result["lags"]] == [15,14,13,12]
    for row in result["lags"]:
        assert row["unfiltered_model_speed"] == 8/256
        assert row["visible_inframe_conditional_speed"] == 8/256


def test_missing_and_outside_predictions_are_not_zero_or_success():
    phases = fixture()
    for z in phases:
        z["visible"][:] = False
    result = motion_summary(phases)
    assert result["lags"][0]["unfiltered_model_speed"] > 0
    assert result["lags"][0]["visible_inframe_conditional_speed"] is None
    phases[0]["visible"][:] = True
    phases[0]["tracks"][1,0] = [-1,20]
    _, known = pair_values(phases[0],0,1)
    assert known.tolist() == [False,True]


def test_query_geometry_must_match_and_visibility_must_be_boolean():
    z = fixture()[0]
    row = {"input_shape":z["shape"].tolist(), "timestamps":z["timestamps"].tolist(),"queries_txy":z["queries_txy"].tolist()}
    assert verify_arrays(z,row) == 0
    z["visible"] = z["visible"].astype(float)
    with pytest.raises(ValueError):
        verify_arrays(z,row)
