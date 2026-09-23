import numpy as np
import pytest

from scripts.counterfactual.probe_feature_tracks import main
from scripts.counterfactual.audit_feature_tracks import point_statistics


def test_invalid_query_frame_rejected_before_inputs():
    args = ["prepare", "--manifest", "absent", "--review", "absent", "--tracker-reference", "absent",
            "--output", "absent", "--candidates", "x", "--query-frame", "-1"]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_no_visible_motion_does_not_become_zero():
    xy = np.array([[[2., 1.]], [[10., 4.]], [[2., 1.]]])
    result = point_statistics(xy, np.zeros((3, 1), bool), 0)
    assert result["all_query_step_mean_pixels"] > 8
    assert result["visible_only_step_mean_pixels"] is None
    assert point_statistics(xy, np.ones((3, 1), bool), 1)["all_query_step_mean_pixels"] > 8
