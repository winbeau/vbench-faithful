"""The timing cohort must remain identical across all sixteen dimensions."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_speed32_inputs as benchmark


def test_all_dimensions_receive_same_32_media_and_reviewed_queries(tmp_path, monkeypatch):
    annotations = json.loads((ROOT / "configs/benchmarks/same32-20261008.json").read_text())
    expected = {str((tmp_path / v["video"]).resolve()): v for v in annotations["videos"]}
    monkeypatch.setattr(benchmark, "digest", lambda p: expected[str(p)]["video_sha256"])
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda args, **kw:
                        json.dumps({"streams": [expected[args[-1]]["media"]]}))
    rows = benchmark.prepare(annotations, tmp_path)
    assert len(rows) == 512 and len({r["id"] for r in rows}) == 512
    for dimension in benchmark.OFFICIAL:
        selected = [r for r in rows if r["dimensions"] == [dimension]]
        assert len(selected) == 32 and {r["video"] for r in selected} == expected.keys()
    first_scene = rows[0]
    assert isinstance(first_scene["auxiliary_info"]["scene"]["scene"]["scene"], str)
    for row in rows:
        original = expected[row["video"]]
        assert Path(row["video"]).name == original["filename"]
        assert row["prompt"] == original["queries"].get(row["dimensions"][0], original["source_prompt"])
    duplicate = copy.deepcopy(annotations)
    duplicate["videos"][1] = duplicate["videos"][0]
    with pytest.raises(ValueError, match="distinct"):
        benchmark.prepare(duplicate, tmp_path)
    monkeypatch.setattr(benchmark, "digest", lambda p: "changed")
    with pytest.raises(ValueError, match="Media changed"):
        benchmark.prepare(annotations, tmp_path)
