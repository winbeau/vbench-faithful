import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subject_consistency.cli import main


def fake_results(backend, videos, metadata, gpu_ids, dino_config):
    return [
        {
            "video": str(video), "prompt": metadata.get(video.name, {}).get("prompt", ""),
            "backend": backend, "score": 0.75, "status": "succeeded",
            "failure_reason": None, "error": None,
            "diagnostics": {"num_frames": 3, "local_score": 0.8, "global_score": 0.7, "final_score": 0.75},
        }
        for video in videos
    ]


class CliSmokeTests(unittest.TestCase):
    def run_cli(self, argv):
        config = {"repo_or_dir": "/tmp/dino", "path": "/etc/hosts", "model": "dino_vitb16", "source": "local", "read_frame": False}
        environment = {"model_name": "DINO ViT-B/16", "upstream_sha": "mock", "input_sha256": {}}
        with (
            patch("subject_consistency.cli.verify_upstream"),
            patch("subject_consistency.cli.check_cuda", return_value={"requested_gpu_ids": [0]}),
            patch("subject_consistency.cli.build_dino_config", return_value=config),
            patch("subject_consistency.cli.environment_record", return_value=environment),
            patch("subject_consistency.cli.evaluate_backend_sharded", side_effect=fake_results) as backend,
        ):
            code = main(argv)
        return code, backend

    def test_help(self):
        with self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_single_video_without_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / "arbitrary.mp4"
            video.touch()
            code, backend = self.run_cli(["--audit", "--video", str(video), "--output", str(root / "out")])
            self.assertEqual(code, 0)
            self.assertEqual(backend.call_args.args[0], "audit")
            result_path = next((root / "out").glob("subject-consistency/audit/*/results.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(result["score"], 0.75)
            self.assertEqual(result["diagnostics"]["num_frames"], 3)

    def test_both_modes_share_run_id_and_dino_config(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / "video.mp4"
            video.touch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"videos": [{"video": video.name, "prompt": "a dog running"}]}), encoding="utf-8")
            code, backend = self.run_cli(["--both", "--video", str(video), "--metadata", str(metadata), "--output", str(root / "out")])
            self.assertEqual(code, 0)
            self.assertEqual([call.args[0] for call in backend.call_args_list], ["vbench", "audit"])
            official = next((root / "out").glob("subject-consistency/vbench/*"))
            repaired = next((root / "out").glob("subject-consistency/audit/*"))
            self.assertEqual(official.name, repaired.name)


if __name__ == "__main__":
    unittest.main()
