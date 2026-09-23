import numpy as np

from dynamic_degree.dense_correspondence import DenseCorrespondenceConfig
from scripts.counterfactual.probe_image_plane_modes import inspect_fields


def test_zero_flow_has_no_temporal_mode_or_fabricated_score():
    rng = np.random.default_rng(8)
    image = rng.integers(10, 240, (40, 48, 3), np.uint8)
    frames = np.stack([image] * 4)
    fields = np.zeros((3, 40, 48, 2), np.float32)
    masks = np.zeros((2, 40, 48), bool)
    masks[0, :, :24] = True; masks[1, :, 24:] = True
    result, arrays = inspect_fields(frames, np.arange(4.) / 8, fields, fields, masks, DenseCorrespondenceConfig())
    assert result["score"] is None
    assert not result["modes_all_points"]
    assert not result["modes_reliable_points"]
    assert result["points"] == 1024
    assert all(r["explained_on_common"] is None for r in result["leave_region_out"])
    assert arrays["velocity"].shape == (3, 1024, 2)
