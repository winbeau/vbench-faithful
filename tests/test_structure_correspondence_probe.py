import numpy as np
import pytest

from scripts.counterfactual.probe_structure_correspondence import describe_evidence, main


def test_replay_uses_every_lag_start_and_reports_no_scalar_score():
    y, x = np.mgrid[:3, :3]
    q = np.stack((x, y), axis=-1).reshape(-1, 2).astype(np.float32)
    f = np.tile(np.eye(9, dtype=np.float16)[None], (16, 1, 1))
    cache = {"key9": f, "token11": f, "attention": np.ones((16, 9), np.float32),
             "queries": q, "grid_shape": np.array([3, 3]), "input_shape": np.array([16, 3, 3, 3])}
    summary, arrays = describe_evidence(cache, np.arange(16) / 8, ["key9", "token11"], [1, 2, 3, 4], "cpu")
    for facet in ("key9", "token11"):
        assert len(summary[facet]["pairs"]) == 54
        assert len(summary[facet]["transitivity"]) == 14
        for lag in range(1, 5):
            part = summary[facet]["by_lag"][str(lag)]
            assert part["frame_pairs"] == 16 - lag
            assert part["summary"]["mutual_conditional_speed"]["mean"] == 0.
            for start in range(16 - lag):
                assert np.array_equal(arrays[f"{facet}_{start}_{lag}_target_xy"], q)
    assert "score" not in summary


@pytest.mark.parametrize("options", [[], ["--replay-source", "cached", "--dino-root", "root"]])
def test_research_cli_rejects_missing_model_or_mixed_replay(tmp_path, options):
    with pytest.raises(SystemExit) as e:
        main(["--manifest", "not_read", "--review", "not_read", "--config", "not_read", "--output", str(tmp_path / "out"), *options])
    assert e.value.code == 2
    assert not (tmp_path / "out").exists()
