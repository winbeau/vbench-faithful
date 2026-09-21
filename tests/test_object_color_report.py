import pytest
from scripts.object_color_report import paired, gate


def test_base_bootstrap_keeps_zero_effects_and_requires_direction():
    assert paired([0, 0, 1])["zero_effect_count"] == 2
    assert paired([0, 0, 1])["n"] == 3
    assert gate(paired([0]*20), "invariance") is True
    assert gate(paired([.9]*20), "change") is True
    assert gate(paired([-.9]*20), "ordered_drop") is False
    assert gate(paired([]), "change") is None
    with pytest.raises(ValueError):
        paired([float("nan"), 0])
