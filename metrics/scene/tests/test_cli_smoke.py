import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scene.cli import main


class CliSmokeTests(unittest.TestCase):
    def test_help(self):
        with self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_global_mode_smoke_with_injected_scorer(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / "video_000.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"videos": [{"video": video.name, "prompt": "beach", "dimension_metadata": {"scene": "beach"}}]}), encoding="utf-8")
            def fake_spawn(mode, videos, metadata, gpu_ids, **kwargs):
                self.assertEqual(mode, "global")
                self.assertEqual(kwargs["scorer_config"]["scorer"], "openclip")
                return [{"video": str(video), "prompt": "beach", "scene": "beach", "backend": "audit", "score": 0.75, "status": "succeeded", "failure_reason": None, "error": None, "diagnostics": {"frame_scores": [0.75]}}]
            with (
                patch("scene.cli.check_cuda", return_value={"requested_gpu_ids": [0]}),
                patch("scene.cli.evaluate_backend_sharded", side_effect=fake_spawn),
            ):
                code = main(["--audit", "--audit-variant", "global", "--video", str(video), "--metadata", str(metadata), "--output", str(root / "out")])
            self.assertEqual(code, 0)
            result_path = next((root / "out").glob("scene/audit/*/results.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(result["score"], 0.75)

    def test_vbench_dispatches_official_and_uses_workspace_default_output(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / "video_000.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"videos": [{"video": video.name, "prompt": "beach", "dimension_metadata": {"scene": "beach"}}]}), encoding="utf-8")
            with (
                patch("scene.cli.check_cuda", return_value={"requested_gpu_ids": [0]}),
                patch("scene.cli.verify_upstream"),
                patch("scene.cli.evaluate_backend_sharded", return_value=[{"video": str(video), "backend": "vbench", "score": 1.0, "status": "succeeded", "diagnostics": {}}]) as run,
                patch("scene.cli.environment_record", return_value={"upstream_sha": "mock", "input_sha256": {}}),
                patch("scene.cli.output_base", return_value=root / "out"),
            ):
                self.assertEqual(main(["--vbench", "--video", str(video), "--metadata", str(metadata)]), 0)
            self.assertEqual(run.call_args.args[0], "official")

    def test_both_honors_global_audit_variant(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "video_000.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"videos": [{"video": video.name, "prompt": "beach", "dimension_metadata": {"scene": "beach"}}]}), encoding="utf-8")
            calls = []

            def fake_spawn(mode, videos, metadata, gpu_ids, **kwargs):
                calls.append(mode)
                return [{"video": str(video), "backend": "vbench" if mode == "official" else "audit", "score": 1.0, "status": "succeeded", "diagnostics": {}}]

            with (
                patch("scene.cli.check_cuda", return_value={"requested_gpu_ids": [0]}),
                patch("scene.cli.verify_upstream"),
                patch("scene.cli.environment_record", return_value={"upstream_sha": "mock", "input_sha256": {}}),
                patch("scene.cli.evaluate_backend_sharded", side_effect=fake_spawn),
            ):
                self.assertEqual(main(["--both", "--audit-variant", "global", "--video", str(video), "--metadata", str(metadata), "--output", str(root / "out")]), 0)
            self.assertEqual(calls, ["official", "global"])
            self.assertEqual(
                next((root / "out").glob("scene/vbench/*")).name,
                next((root / "out").glob("scene/audit/*")).name,
            )


if __name__ == "__main__":
    unittest.main()
