import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from human_action.cli import main


def categories():
    return ("cutting watermelon",) + tuple(f"action {index}" for index in range(1, 400))


def metadata_for(names):
    return {
        "videos": [
            {
                "video": name,
                "prompt": "A person is cutting watermelon",
                "dimension_metadata": {"target_action": "cutting watermelon"},
            }
            for name in names
        ]
    }


def fake_backend(backend, videos, metadata, gpu_ids, model_weight, category_names, level, seed):
    return [
        {
            "video": str(video),
            "prompt": metadata[video.name]["prompt"],
            "target_action": "cutting watermelon",
            "backend": backend,
            "score": 1.0 if backend == "vbench" else 0.8,
            "status": "succeeded",
            "failure_reason": None,
            "error": None,
            "task_relevant_action_evidence": (
                {"temporal_mean_target_probability": 0.8, "temporal_coverage": 0.5}
                if backend == "audit"
                else None
            ),
            "diagnostics": {"test": True},
        }
        for video in videos
    ]


class CliSmokeTests(unittest.TestCase):
    def run_cli(self, argv, side_effect=fake_backend):
        environment = {
            "python": "test",
            "input_sha256": {},
            "metadata_sha256": "mock",
            "model_weight_sha256": "mock",
            "upstream_sha": "mock",
        }
        with (
            patch("human_action.cli.verify_upstream"),
            patch("human_action.cli.load_categories", return_value=categories()),
            patch("human_action.cli.check_cuda", return_value={"requested_gpu_ids": [0, 2]}),
            patch("human_action.cli.weight_path", return_value=Path("/etc/hosts")),
            patch("human_action.cli.environment_record", return_value=environment),
            patch("human_action.cli.release_cuda_resources"),
            patch("human_action.cli.evaluate_backend_sharded", side_effect=side_effect) as backend,
        ):
            code = main(argv)
        return code, backend

    def test_single_video_audit_smoke(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "arbitrary-name.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for([video.name])), encoding="utf-8")
            code, backend = self.run_cli(
                ["--audit", "--video", str(video), "--metadata", str(metadata), "--output", str(root / "out")]
            )
            self.assertEqual(code, 0)
            self.assertEqual(backend.call_args.args[0], "audit")
            self.assertEqual(backend.call_args.args[-1], 42)
            result_path = next((root / "out").glob("human-action/audit/*/results.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(result["target_action"], "cutting watermelon")
            self.assertAlmostEqual(result["score"], 0.8)
            self.assertTrue(result_path.with_name("run.log").is_file())

    def test_both_uses_shared_run_id_and_gpu_list(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            names = ["video_000.mp4", "video_003.mp4"]
            for name in names:
                (root / name).touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for(names)), encoding="utf-8")
            code, backend = self.run_cli(
                ["--both", "--video-dir", str(root), "--metadata", str(metadata), "--output", str(root / "out"), "--gpu", "0,2"]
            )
            self.assertEqual(code, 0)
            self.assertEqual([call.args[0] for call in backend.call_args_list], ["vbench", "audit"])
            self.assertTrue(all(call.args[3] == [0, 2] for call in backend.call_args_list))
            self.assertTrue(all(call.args[-1] == 42 for call in backend.call_args_list))
            official_dir = next((root / "out").glob("human-action/vbench/*"))
            audit_dir = next((root / "out").glob("human-action/audit/*"))
            self.assertEqual(official_dir.name, audit_dir.name)

    def test_missing_metadata_fails_before_cuda(self):
        with tempfile.TemporaryDirectory() as root_value:
            video = Path(root_value) / "A person is cutting watermelon_0.mp4"
            video.touch()
            with patch("human_action.cli.check_cuda") as cuda:
                code = main(["--audit", "--video", str(video)])
            self.assertEqual(code, 2)
            cuda.assert_not_called()

    def test_interruption_is_recorded(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "video.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for([video.name])), encoding="utf-8")
            code, _ = self.run_cli(
                ["--audit", "--video", str(video), "--metadata", str(metadata), "--output", str(root / "out")],
                KeyboardInterrupt,
            )
            self.assertEqual(code, 130)
            summary = next((root / "out").glob("human-action/audit/*/summary.json"))
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["status"], "interrupted")
