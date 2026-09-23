from copy import deepcopy

import pytest

from dynamic_degree.local_appearance import consensus_hypotheses
from scripts.counterfactual.audit_appearance_consensus import verify_consensus


def test_independent_peak_vote_check_rejects_invented_weight():
    appearance = {"hypotheses": [{"displacement_pixels": [3, 4], "correlation": .9},
                                {"displacement_pixels": [-20, 10], "correlation": .85}],
                  "self_alternatives": [{"maximum_support_intersection": .5, "hypotheses": [{"correlation": .6}]}]}
    entries = [{"xy": xy, "appearance": appearance} for xy in ([10., 12.], [20., 21.], [12., 25.])]
    result = consensus_hypotheses(entries, (48, 48))
    verify_consensus(result, entries)
    wrong = deepcopy(result)
    wrong["folds"][0]["peaks"][0]["vote_weight"] += .01
    with pytest.raises(ValueError, match="peak"):
        verify_consensus(wrong, entries)
