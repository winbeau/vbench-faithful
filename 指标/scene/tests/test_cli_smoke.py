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
            class FakeScorer:
                def score(self, image, label):
                    return 0.75
            with (
                patch("scene.cli.check_cuda", return_value={"requested_gpu_ids": [0]}),
                patch("scene.cli.build_scorer", return_value=FakeScorer()),
                patch("scene.metric.decode_video", return_value=([0], [object()])),
                patch("scene.metric.evaluate_global_or_environment_batch", return_value=[{"video": str(video), "prompt": "beach", "scene": "beach", "backend": "global", "score": 0.75, "status": "succeeded", "failure_reason": None, "error": None, "diagnostics": {"frame_scores": [0.75]} }]),
            ):
                code = main(["--mode", "global", "--video", str(video), "--metadata", str(metadata), "--output", str(root / "out")])
            self.assertEqual(code, 0)
            result_path = next((root / "out").glob("scene/global/*/results.json"))
            result = json.loads(result_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(result["score"], 0.75)


if __name__ == "__main__":
    unittest.main()
