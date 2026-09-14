import unittest
from pathlib import Path

from scene.backends.vbench import (
    UPSTREAM_SHA,
    UPSTREAM_PATH,
    extract_scene_label,
    official_frame_match,
    sample_middle_indices,
    dataset_score,
    official_metadata_entry,
    OfficialVBenchSceneEvaluator,
    verify_upstream,
    video_score,
)


class OfficialParityTests(unittest.TestCase):
    def test_locked_upstream_identity(self):
        state = verify_upstream()
        self.assertEqual(state["sha"], UPSTREAM_SHA)
        self.assertFalse(state["dirty"])

    def test_metadata_target_extraction_preserves_official_nesting(self):
        item = {"auxiliary_info": {"scene": {"scene": {"scene": "beach"}}}}
        self.assertEqual(extract_scene_label(item), "beach")

    def test_lexical_matching_is_all_space_tokens_and_no_threshold(self):
        self.assertTrue(official_frame_match("art gallery", "an art gallery with paintings"))
        self.assertFalse(official_frame_match("art gallery", "an art studio"))
        self.assertTrue(official_frame_match("beach", "beachfront"))

    def test_sampling_is_16_middle_and_pads_short_video(self):
        self.assertEqual(sample_middle_indices(16, 4), list(range(4)) + [3] * 12)
        self.assertEqual(sample_middle_indices(16, 1), [0] * 16)

    def test_sampling_matches_upstream_floating_step_edge_case(self):
        self.assertEqual(
            sample_middle_indices(14, 122),
            [3, 12, 21, 29, 38, 47, 55, 64, 73, 82, 90, 99, 108, 117],
        )

    def test_frame_video_aggregation_is_continuous_mean(self):
        self.assertAlmostEqual(video_score([True, False, True, False]), 0.5)

    def test_dataset_aggregation_uses_global_frame_counts(self):
        self.assertAlmostEqual(
            dataset_score([
                {"success_frame_count": 1, "frame_count": 2},
                {"success_frame_count": 3, "frame_count": 4},
            ]),
            4 / 6,
        )

    def test_official_metadata_adapter_keeps_prompt_and_nested_label(self):
        entry = official_metadata_entry(
            Path("/tmp/video_000.mp4"),
            {"prompt": "beach", "dimension_metadata": {"scene": "beach"}},
        )
        self.assertEqual(entry["prompt_en"], "beach")
        self.assertEqual(entry["auxiliary_info"]["scene"]["scene"]["scene"], "beach")

    def test_official_wrapper_accepts_injected_compute_for_logic_parity(self):
        calls = []

        def compute(path, device, config):
            calls.append((path, device, config))
            return 0.5, [{"video_path": "/tmp/video_000.mp4", "video_results": 0.5, "frame_results": [1, 0], "frame_count": 2}]

        evaluator = OfficialVBenchSceneEvaluator(
            "cuda:0", upstream=UPSTREAM_PATH, compute=compute
        )
        result = evaluator.evaluate_video(
            Path("/tmp/video_000.mp4"),
            {"prompt": "beach", "dimension_metadata": {"scene": "beach"}},
        )
        self.assertEqual(result["video_results"], 0.5)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
