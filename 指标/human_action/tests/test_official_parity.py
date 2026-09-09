import os
import unittest
from pathlib import Path

import numpy as np

from human_action.backends.vbench import (
    OfficialHumanActionEvaluator,
    official_decision,
    official_target_from_filename,
    import_official_module,
    verify_upstream,
)
from human_action.models import middle_sample_indices
from human_action.metric import evaluate_vbench_batch
from human_action.schemas import OfficialActionResult


def categories():
    values = [f"action {index}" for index in range(400)]
    values[7] = "cutting watermelon"
    return tuple(values)


class OfficialParityTests(unittest.TestCase):
    def test_locked_upstream_identity(self):
        state = verify_upstream()
        self.assertEqual(state.sha, "13dee903cc97e2633ed6e8f50dea61bc90717935")
        self.assertFalse(state.dirty)

    def test_filename_target_matches_locked_source_expression(self):
        self.assertEqual(
            official_target_from_filename("/tmp/A person is cutting watermelon_0-1.mp4"),
            "cutting watermelon",
        )
        self.assertEqual(official_target_from_filename("/tmp/neutral.mp4"), "neutral.mp4")

    def test_official_round_top5_threshold_and_exact_label(self):
        probs = np.zeros(400)
        probs[[7, 1, 2, 3, 4]] = [0.84996, 0.99, 0.98, 0.97, 0.96]
        matched, actions, rounded, accepted = official_decision(
            probs, categories(), "cutting watermelon"
        )
        self.assertTrue(matched)
        self.assertEqual(rounded[-1], 0.85)
        self.assertIn("cutting watermelon", accepted)
        self.assertEqual(len(actions), 5)

    def test_middle_sampling_matches_known_official_cases(self):
        self.assertEqual(middle_sample_indices(4, 8), (0, 2, 4, 6))
        self.assertEqual(middle_sample_indices(4, 2), (0, 1, 1, 1))

    def test_middle_sampling_matches_locked_upstream_helper(self):
        module, _ = import_official_module()
        upstream_sampler = module.load_video.__globals__["get_frame_indices"]
        for frame_count in (1, 2, 17, 64):
            with self.subTest(frame_count=frame_count):
                self.assertEqual(
                    middle_sample_indices(16, frame_count),
                    tuple(upstream_sampler(16, frame_count, sample="middle")),
                )

    def test_official_batch_isolates_corrupt_video(self):
        videos = [Path("/tmp/video_000.mp4"), Path("/tmp/video_001.mp4")]
        metadata = {
            video.name: {
                "prompt": "A person is cutting watermelon",
                "dimension_metadata": {"target_action": "cutting watermelon"},
            }
            for video in videos
        }

        class FakeEvaluator:
            def evaluate_video(self, video):
                if video.name == "video_000.mp4":
                    raise RuntimeError("corrupt video")
                return OfficialActionResult(
                    str(video), "video_001.mp4", "official_filename_parser", 16,
                    (), (), (), 0.85, False,
                )

        results = evaluate_vbench_batch(
            videos, metadata, "cuda:0", Path("/tmp/mock.pth"), categories(), evaluator=FakeEvaluator()
        )
        self.assertEqual([result["status"] for result in results], ["failed", "succeeded"])
        self.assertIn("corrupt video", results[0]["failure_reason"])

    @unittest.skipUnless(
        os.environ.get("VBENCH_AUDIT_REAL_HUMAN_ACTION_PARITY") == "1",
        "requires real video, CUDA, locked UMT weight",
    )
    def test_real_video_parity(self):
        import torch

        video = Path(os.environ["VBENCH_AUDIT_REAL_HUMAN_ACTION_PARITY_VIDEO"]).resolve()
        weight = Path(os.environ["VBENCH_AUDIT_UMT_WEIGHT"]).resolve()
        gpu = int(os.environ.get("VBENCH_AUDIT_REAL_HUMAN_ACTION_PARITY_GPU", "0"))
        self.assertTrue(torch.cuda.is_available())
        module, _ = import_official_module()
        device = torch.device(f"cuda:{gpu}")
        expected_score, expected_results = module.human_action(str(weight), [str(video)], device)
        torch.cuda.empty_cache()
        actual = OfficialHumanActionEvaluator(device, weight).evaluate_video(video)
        self.assertEqual(actual.matched, expected_results[0]["video_results"])
        self.assertEqual(float(actual.matched), expected_score)
