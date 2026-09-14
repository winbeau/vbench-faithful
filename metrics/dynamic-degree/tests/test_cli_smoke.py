import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dynamic_degree.cli import main


def metadata_for(names):
    return {
        "videos": [
            {
                "video": name,
                "prompt": "a person is running",
                "dimension_metadata": {"motion_target": "subject"},
            }
            for name in names
        ]
    }


def fake_backend(backend, videos, metadata, gpu_ids, model_weight, level, seed):
    return [
        {
            "video": str(video),
            "prompt": metadata[video.name].get("prompt", ""),
            "backend": backend,
            "score": 1.0,
            "status": "succeeded",
            "failure_reason": None,
            "error": None,
            "selected_evidence_channel": "residual" if backend == "audit" else None,
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
            patch("dynamic_degree.cli.check_cuda", return_value={"requested_gpu_ids": [0, 2]}),
            patch("dynamic_degree.cli.weight_path", return_value=Path("/etc/hosts")),
            patch("dynamic_degree.cli.environment_record", return_value=environment),
            patch("dynamic_degree.cli.release_cuda_resources"),
            patch("dynamic_degree.cli.evaluate_backend_sharded", side_effect=side_effect) as backend,
        ):
            code = main(argv)
        return code, backend

    def test_single_video_audit_smoke(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / "arbitrary.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for([video.name])), encoding="utf-8")
            code, backend = self.run_cli(
                ["--audit", "--video", str(video), "--metadata", str(metadata), "--output", str(root / "out")]
            )
            self.assertEqual(code, 0)
            self.assertEqual(backend.call_args.args[0], "audit")
            self.assertEqual(backend.call_args.args[-1], 42)
            result_path = next((root / "out").glob("dynamic-degree/audit/*/results.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(result["selected_evidence_channel"], "residual")
            self.assertTrue(result_path.with_name("run.log").is_file())

    def test_batch_both_shared_run_id_and_gpu_forwarding(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            names = ["video_000.mp4", "video_003.mp4", "video_1000.mp4"]
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
            vbench_dir = next((root / "out").glob("dynamic-degree/vbench/*"))
            audit_dir = next((root / "out").glob("dynamic-degree/audit/*"))
            self.assertEqual(vbench_dir.name, audit_dir.name)

    def test_missing_metadata_is_allowed_and_not_inferred_from_filename(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / "running-person.mp4"
            video.touch()
            captured = {}

            def inspect_backend(backend, videos, metadata, gpu_ids, model_weight, level, seed):
                captured.update(metadata[videos[0].name])
                return fake_backend(backend, videos, metadata, gpu_ids, model_weight, level, seed)

            code, _ = self.run_cli(
                ["--audit", "--video", str(video), "--output", str(root / "out")], inspect_backend
            )
            self.assertEqual(code, 0)
            self.assertEqual(captured["prompt"], "")

    def test_interrupt_is_saved(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / "video.mp4"
            video.touch()
            code, _ = self.run_cli(
                ["--audit", "--video", str(video), "--output", str(root / "out")], KeyboardInterrupt
            )
            self.assertEqual(code, 130)
            summary = next((root / "out").glob("dynamic-degree/audit/*/summary.json"))
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["status"], "interrupted")
