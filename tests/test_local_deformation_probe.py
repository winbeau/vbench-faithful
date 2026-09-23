import numpy as np

from scripts.counterfactual.probe_local_appearance import probe_pair
from scripts.counterfactual.probe_local_deformation import declared_supports


def test_refinement_keeps_all_point_region_refs_for_declared_factor():
    rng = np.random.default_rng(8)
    frame = rng.integers(10, 240, (24, 24, 3), np.uint8)
    masks = np.ones((2, 24, 24), bool)
    features = {"xy": np.array([[12., 12.]]), "scale": np.array([2.])}
    pair = probe_pair(frame, frame, masks, features)
    chosen = declared_supports(pair, masks, [4])
    assert len(chosen) == 1
    item = next(iter(chosen.values()))
    assert len(item["references"]) == 3
    assert [r["region"] for r in item["references"]] == [0, 1, 2]
    assert item["references"][-1]["whole_frame_control"]
