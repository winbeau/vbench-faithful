import numpy as np
import pytest

from dynamic_degree.image_plane_modes import regional_reversal_ablation
from scripts.counterfactual.audit_motion_boundaries import verify_removal


def test_audit_recomputes_constrained_removal_and_rejects_changed_scalar():
    y, x = np.mgrid[:12, :12]
    xy = np.c_[x.ravel(), y.ravel()]
    owners = (x // 4).ravel()
    field = np.c_[np.sin(xy[:, 0] * .7), np.cos(xy[:, 1] * .8)]
    values = np.sin(np.arange(15.) * 2.3)[:, None, None] * field
    result, arrays = regional_reversal_ablation(values, np.arange(16.) / 8, xy, owners, np.ones(values.shape[:2], bool), (12, 12))
    checked = verify_removal(arrays, (12, 12), result)
    assert checked["constraints_checked"]
    wrong = result | {"conditional_guarded_speed": 0.}
    with pytest.raises(ValueError, match="speed"):
        verify_removal(arrays, (12, 12), wrong)
    wrong_arrays = arrays | {"corrected_velocity": arrays["corrected_velocity"] + .01}
    with pytest.raises(ValueError, match="deletion"):
        verify_removal(wrong_arrays, (12, 12), result)
