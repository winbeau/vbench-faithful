import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from overall_consistency.cli import main


def metadata_for(names):
    return {"videos": [{"video": name, "prompt": "a woman playing guitar", "semantic_conditions": ["a woman", "a woman playing guitar"]} for name in names]}


def fake_results(mode, videos, metadata, gpu_ids, checkpoint, alpha, lambda_):
    return [{"video": str(video), "prompt": metadata[video.name]["prompt"], "backend": mode, "score": 0.4, "status": "succeeded", "failure_reason": None, "error": None, "diagnostics": {"alpha": alpha, "lambda": lambda_}} for video in videos]


class CliSmokeTests(unittest.TestCase):
    def run_cli(self, argv):
        with (
            patch("overall_consistency.cli.verify_upstream"),
            patch("overall_consistency.cli.checkpoint_path", return_value=Path("/etc/hosts")),
            patch("overall_consistency.cli.check_cuda", return_value={"requested_gpu_ids": [0, 2]}),
            patch("overall_consistency.cli._environment", return_value={"upstream_sha": "mock"}),
            patch("overall_consistency.cli.evaluate_sharded", side_effect=fake_results) as evaluate,
        ):
            code = main(argv)
        return code, evaluate

    def test_repair_mode_parses_config_and_writes_output(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "video.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for([video.name])), encoding="utf-8")
            code, evaluate = self.run_cli(["--mode", "repair", "--video", str(video), "--metadata", str(metadata), "--output", str(root / "out"), "--alpha", "0.25", "--lambda", "0.75"])
            self.assertEqual(code, 0)
            self.assertEqual(evaluate.call_args.args[0], "repair")
            self.assertEqual(evaluate.call_args.args[5:], (0.25, 0.75))
            result_path = next((root / "out").glob("overall-consistency/repair/*/results.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(result["score"], 0.4)
            self.assertEqual(result["diagnostics"]["lambda"], 0.75)

    def test_both_modes_share_run_id(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "video.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for([video.name])), encoding="utf-8")
            code, evaluate = self.run_cli(["--both", "--video", str(video), "--metadata", str(metadata), "--output", str(root / "out")])
            self.assertEqual(code, 0)
            self.assertEqual([call.args[0] for call in evaluate.call_args_list], ["official", "repair"])
            official = next((root / "out").glob("overall-consistency/official/*"))
            repair = next((root / "out").glob("overall-consistency/repair/*"))
            self.assertEqual(official.name, repair.name)

    def test_invalid_parameter_fails_before_cuda(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "video.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for([video.name])), encoding="utf-8")
            cuda = Mock()
            with patch("overall_consistency.cli.check_cuda", cuda):
                code = main(["--mode", "repair", "--video", str(video), "--metadata", str(metadata), "--alpha", "2"])
            self.assertEqual(code, 2)
            cuda.assert_not_called()

    def test_invalid_lambda_fails_before_cuda(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "video.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps(metadata_for([video.name])), encoding="utf-8")
            cuda = Mock()
            with patch("overall_consistency.cli.check_cuda", cuda):
                code = main(
                    ["--mode", "repair", "--video", str(video), "--metadata", str(metadata), "--lambda", "-1"]
                )
            self.assertEqual(code, 2)
            cuda.assert_not_called()


if __name__ == "__main__":
    unittest.main()
