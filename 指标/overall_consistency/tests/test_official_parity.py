import math
import importlib
import sys
import unittest
from pathlib import Path

from overall_consistency.backends.vbench import (
    INPUT_SIZE,
    NUM_FRAMES,
    SAMPLE_MODE,
    UPSTREAM_SHA,
    UPSTREAM_PATH,
    OfficialEvaluator,
    official_entry,
    sample_middle_indices,
    verify_upstream,
)
from overall_consistency.metric import cosine_similarity, dataset_mean, normalize_feature


class FakeEncoder:
    model = object()
    tokenizer = object()


class FakeOfficialModule:
    def __init__(self):
        self.calls = []

    def overall_consistency(self, model, entries, tokenizer, device, sample):
        self.calls.append((model, entries, tokenizer, device, sample))
        return 0.25, [{"video_path": entries[0]["video_list"][0], "video_results": 0.25}]


class OfficialParityTests(unittest.TestCase):
    def test_locked_upstream_and_constants(self):
        state = verify_upstream()
        self.assertEqual(state["sha"], UPSTREAM_SHA)
        self.assertEqual((NUM_FRAMES, SAMPLE_MODE, INPUT_SIZE), (8, "middle", 224))

    def test_middle_sampling_uniformly_partitions_full_timeline(self):
        self.assertEqual(sample_middle_indices(8, 16), [0, 2, 4, 6, 8, 10, 12, 14])
        self.assertEqual(sample_middle_indices(8, 3), [0, 1, 2, 2, 2, 2, 2, 2])

    def test_sampling_matches_locked_upstream_helper(self):
        sys.path.insert(0, str(UPSTREAM_PATH))
        utils = importlib.import_module("vbench.utils")
        for length in (1, 3, 8, 16, 17, 101):
            self.assertEqual(
                sample_middle_indices(8, length),
                list(utils.get_frame_indices(8, length, sample="middle")),
            )

    def test_official_adapter_passes_full_prompt_and_middle_sampling(self):
        module = FakeOfficialModule()
        evaluator = OfficialEvaluator("cuda:0", Path("/unused"), module=module, encoder=FakeEncoder())
        metadata = {"prompt": "A woman playing guitar on a beach."}
        self.assertEqual(evaluator.evaluate_video(Path("/tmp/video_000.mp4"), metadata), 0.25)
        _, entries, _, device, sample = module.calls[0]
        self.assertEqual(entries[0]["prompt"], metadata["prompt"])
        self.assertEqual(device, "cuda:0")
        self.assertEqual(sample, "middle")

    def test_official_metadata_entry_preserves_prompt_verbatim(self):
        prompt = "  a deliberately spaced prompt  "
        self.assertEqual(official_entry(Path("/tmp/a.mp4"), {"prompt": prompt})["prompt"], prompt)

    def test_l2_normalization_cosine_and_dataset_mean(self):
        self.assertEqual(normalize_feature([3.0, 4.0]), (0.6, 0.8))
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)
        self.assertAlmostEqual(dataset_mean([0.1, 0.3, 0.8]), 0.4)

    def test_normalization_matches_locked_official_helpers(self):
        sys.path.insert(0, str(UPSTREAM_PATH))
        module = importlib.import_module("vbench.overall_consistency")
        import torch

        class Model:
            def encode_vision(self, frames, test):
                self.video_test = test
                return torch.tensor([[3.0, 4.0]])

            def encode_text(self, text):
                self.text = text
                return torch.tensor([[-3.0, 4.0]])

        model = Model()
        video = module.get_vid_features(model, torch.zeros(1))
        text = module.get_text_features(model, "full prompt", object(), {})
        self.assertTrue(model.video_test)
        self.assertEqual(model.text, "full prompt")
        self.assertEqual(tuple(video[0].tolist()), (0.6000000238418579, 0.800000011920929))
        self.assertEqual(tuple(text[0].tolist()), (-0.6000000238418579, 0.800000011920929))

    def test_zero_norm_and_nan_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_feature([0.0, 0.0])
        with self.assertRaises(ValueError):
            normalize_feature([math.nan, 1.0])


if __name__ == "__main__":
    unittest.main()
