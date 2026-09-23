from scripts.counterfactual.audit_local_deformation import point_displacement


def test_report_uses_query_point_not_patch_center():
    fit = {"status": "diagnostic_only", "affine_matrix": [[2, 0, 1], [0, 1, -3]],
           "center_displacement_pixels": [99, 99]}
    assert point_displacement(fit, [4, 5]) == [5, -3]
    assert point_displacement({"status": "insufficient_evidence"}, [4, 5]) is None
