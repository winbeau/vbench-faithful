import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from spatial_relationship.cli import main
from spatial_relationship.models import AblationMode


def metadata_for(names):
    return {
        "videos": [
            {
                "video": name,
                "prompt": "a cat on the left of a dog",
                "dimension_metadata": {
                    "object_a": "cat",
                    "object_b": "dog",
                    "relationship": "on the left of",
                },
            }
            for name in names
        ]
    }


def fake_backend(backend, videos, metadata, gpu_ids, model_weight, level, seed, **kwargs):
    return [
        {
            "video": str(video), "prompt": metadata[video.name]["prompt"],
            "subject": "cat", "relation": "on the left of", "object": "dog",
            "backend": backend, "score": 1.0, "status": "succeeded",
            "failure_reason": None, "error": None,
            "diagnostics": {"frame_scores": [1.0], "sampled_frame_indices": [0]},
        }
        for video in videos
    ]


class CliSmokeTests(unittest.TestCase):
    def run_cli(self, argv):
        environment = {
            "python": "test", "input_sha256": {}, "metadata_sha256": "mock",
            "model_weight_sha256": "mock", "upstream_sha": "mock",
        }
        with (
            patch("spatial_relationship.cli.check_cuda", return_value={"requested_gpu_ids": [0, 2]}),
            patch("spatial_relationship.cli.weight_path", return_value=Path("/etc/hosts")),
            patch("spatial_relationship.cli.environment_record", return_value=environment),
            patch("spatial_relationship.cli.evaluate_backend_sharded", side_effect=fake_backend) as backend,
        ):
            code = main(argv)
        return code, backend

    def test_single_video_smoke(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / "arbitrary-name.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for([video.name])), encoding="utf-8")
            code, backend = self.run_cli(["--audit", "--video", str(video), "--metadata", str(metadata), "--output", str(root / "out")])
            self.assertEqual(code, 0)
            self.assertEqual(backend.call_args.args[0], "audit")
            self.assertEqual(backend.call_args.args[-1], 42)
            result_path = next((root / "out").glob("spatial-relationship/audit/*/results.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))[0]
            self.assertEqual((result["subject"], result["relation"], result["object"]), ("cat", "on the left of", "dog"))
            self.assertTrue(result_path.with_name("run.log").is_file())

    def test_audit_variant_flags_reach_the_scoring_backend(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / "arbitrary-name.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for([video.name])), encoding="utf-8")
            code, backend = self.run_cli([
                "--audit", "--video", str(video), "--metadata", str(metadata),
                "--output", str(root / "out"), "--audit-mode", "ordered_role", "--detection-conditioned",
            ])
            self.assertEqual(code, 0)
            self.assertEqual(backend.call_args.kwargs["mode"], AblationMode.ORDERED_ROLE)
            self.assertTrue(backend.call_args.kwargs["condition_on_detection"])

    def test_batch_both_uses_same_run_id_and_gpu_list(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            names = ["video_000.mp4", "video_003.mp4", "video_1000.mp4"]
            for name in names:
                (root / name).touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for(names)), encoding="utf-8")
            code, backend = self.run_cli(["--both", "--video-dir", str(root), "--metadata", str(metadata), "--output", str(root / "out"), "--gpu", "0,2"])
            self.assertEqual(code, 0)
            self.assertEqual([call.args[0] for call in backend.call_args_list], ["vbench", "audit"])
            self.assertTrue(all(call.args[3] == [0, 2] for call in backend.call_args_list))
            self.assertTrue(all(call.args[-1] == 42 for call in backend.call_args_list))
            vbench_dir = next((root / "out").glob("spatial-relationship/vbench/*"))
            audit_dir = next((root / "out").glob("spatial-relationship/audit/*"))
            self.assertEqual(vbench_dir.name, audit_dir.name)

    def test_missing_semantic_field_fails_before_cuda(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / "video.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"videos": [{"video": video.name, "dimension_metadata": {"object_a": "cat"}}]}), encoding="utf-8")
            cuda = Mock()
            with patch("spatial_relationship.cli.check_cuda", cuda):
                code = main(["--audit", "--video", str(video), "--metadata", str(metadata)])
            self.assertEqual(code, 2)
            cuda.assert_not_called()

    def test_interruption_is_recorded_and_returns_130(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / "video.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for([video.name])), encoding="utf-8")
            environment = {
                "python": "test", "input_sha256": {}, "metadata_sha256": "mock",
                "model_weight_sha256": "mock", "upstream_sha": "mock",
            }
            with (
                patch("spatial_relationship.cli.check_cuda", return_value={"requested_gpu_ids": [0]}),
                patch("spatial_relationship.cli.weight_path", return_value=Path("/etc/hosts")),
                patch("spatial_relationship.cli.environment_record", return_value=environment),
                patch("spatial_relationship.cli.evaluate_backend_sharded", side_effect=KeyboardInterrupt),
            ):
                code = main(["--audit", "--video", str(video), "--metadata", str(metadata), "--output", str(root / "out")])
            self.assertEqual(code, 130)
            summary_path = next((root / "out").glob("spatial-relationship/audit/*/summary.json"))
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8"))["status"], "interrupted")
