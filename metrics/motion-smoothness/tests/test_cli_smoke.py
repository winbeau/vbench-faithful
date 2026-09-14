import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from motion_smoothness.cli import build_parser, main


class CliSmokeTests(unittest.TestCase):
    def test_parser_modes_and_parameters(self):
        args = build_parser().parse_args(["--audit", "--video", "/tmp/video_000.mp4", "--tail-quantile", "0.8", "--tail-weight", "0.1"])
        self.assertTrue(args.audit)
        self.assertEqual(args.tail_quantile, 0.8)
        self.assertEqual(args.tail_weight, 0.1)

    def test_both_dispatches_backend_specific_weights_and_shared_run(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "clip.mp4"
            video.touch()
            seen = []

            def fake_run(backend, videos, gpu_ids, weight, config, seed):
                seen.append((backend, weight))
                return [{"video": str(videos[0]), "backend": backend, "score": 1.0, "status": "succeeded", "diagnostics": {}}]

            with (
                patch("motion_smoothness.cli.check_cuda", return_value={"requested_gpu_ids": [0]}),
                patch.dict("os.environ", {"VBENCH_AUDIT_AMT_WEIGHT": str(root / "amt.pth"), "VBENCH_AUDIT_RAFT_WEIGHT": str(root / "raft.pth")}),
                patch("motion_smoothness.cli.evaluate_backend_sharded", side_effect=fake_run),
            ):
                code = main(["--both", "--video", str(video), "--output", str(root / "out")])
            self.assertEqual(code, 0)
            self.assertEqual([(name, path.name) for name, path in seen], [("vbench", "amt.pth"), ("audit", "raft.pth")])
            runs = [path.name for path in (root / "out" / "motion-smoothness" / "vbench").iterdir()]
            audit_runs = [path.name for path in (root / "out" / "motion-smoothness" / "audit").iterdir()]
            self.assertEqual(runs, audit_runs)

    def test_failed_video_returns_one_and_writes_result(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "clip.mp4"
            video.touch()
            with (
                patch("motion_smoothness.cli.check_cuda", return_value={"requested_gpu_ids": [0]}),
                patch("motion_smoothness.cli.evaluate_backend_sharded", return_value=[{"video": str(video), "backend": "audit", "score": None, "status": "failed", "error": "broken", "diagnostics": None}]),
            ):
                code = main(["--audit", "--video", str(video), "--output", str(root / "out")])
            self.assertEqual(code, 1)
            result = next((root / "out").glob("motion-smoothness/audit/*/results.json"))
            self.assertEqual(json.loads(result.read_text(encoding="utf-8"))[0]["status"], "failed")

    def test_invalid_config_returns_two_before_cuda(self):
        with tempfile.TemporaryDirectory() as root_value:
            video = Path(root_value) / "clip.mp4"
            video.touch()
            with patch("motion_smoothness.cli.check_cuda") as check:
                code = main(["--audit", "--video", str(video), "--tail-quantile", "2"])
            self.assertEqual(code, 2)
            check.assert_not_called()

    def test_backend_error_in_both_is_recorded_and_other_backend_continues(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "clip.mp4"
            video.touch()
            calls = []

            def run(backend, videos, gpu_ids, weight, config, seed):
                calls.append(backend)
                if backend == "vbench":
                    raise RuntimeError("AMT unavailable")
                return [{"video": str(video), "backend": "audit", "score": 0.5, "status": "succeeded", "diagnostics": {}}]

            with (
                patch("motion_smoothness.cli.check_cuda", return_value={"requested_gpu_ids": [0]}),
                patch("motion_smoothness.cli.evaluate_backend_sharded", side_effect=run),
            ):
                code = main(["--both", "--video", str(video), "--output", str(root / "out")])
            self.assertEqual(code, 1)
            self.assertEqual(calls, ["vbench", "audit"])
            self.assertTrue(next((root / "out").glob("motion-smoothness/vbench/*/results.json")).is_file())
            self.assertTrue(next((root / "out").glob("motion-smoothness/audit/*/results.json")).is_file())

    def test_audit_default_raft_weight_uses_shared_cache(self):
        with tempfile.TemporaryDirectory() as root_value:
            video = Path(root_value) / "clip.mp4"
            video.touch()
            seen = []

            def run(backend, videos, gpu_ids, weight, config, seed):
                seen.append(weight)
                return [{"video": str(video), "backend": backend, "score": 1.0, "status": "succeeded", "diagnostics": {}}]

            with (
                patch("motion_smoothness.cli.check_cuda", return_value={"requested_gpu_ids": [0]}),
                patch("motion_smoothness.cli.evaluate_backend_sharded", side_effect=run),
                patch.dict("os.environ", {}, clear=True),
            ):
                self.assertEqual(main(["--audit", "--video", str(video), "--output", str(Path(root_value) / "out")]), 0)
            self.assertEqual(seen, [Path.home() / ".cache/vbench/raft_model/models/raft-things.pth"])
