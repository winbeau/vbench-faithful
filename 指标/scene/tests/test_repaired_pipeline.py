import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scene.metric import decode_video, evaluate_repaired_video


class FakeImage:
    size = (100, 80)

    def __init__(self, box=None):
        self.box = box

    def crop(self, box):
        return FakeImage(box)


class MappingScorer:
    def score(self, image, scene_text):
        self.last_scene_text = scene_text
        if image.box is None:
            return 0.9
        return {
            (0, 0, 50, 40): 0.8,
            (50, 0, 100, 40): 0.82,
            (0, 40, 50, 80): 0.78,
            (50, 40, 100, 80): 0.84,
        }[image.box]


class RepairedPipelineTests(unittest.TestCase):
    def test_decode_video_returns_source_indices_and_pil_like_rgb_frames(self):
        source_indices = [0, 3, 7]

        class FakeBatch:
            def asnumpy(self):
                return ["array-0", "array-3", "array-7"]

        class FakeReader:
            def __init__(self, path, width, height, num_threads):
                self.path = path
                self.width = width
                self.height = height

            def __len__(self):
                return 12

            def get_batch(self, indices):
                self.indices = indices
                return FakeBatch()

        class FakePILImage:
            def __init__(self, array):
                self.array = array

            def convert(self, mode):
                self.mode = mode
                return self

        fake_utils = SimpleNamespace(
            VideoReader=FakeReader,
            get_frame_indices=lambda count, length, sample: source_indices,
            Image=SimpleNamespace(fromarray=FakePILImage),
        )
        fake_decord = SimpleNamespace(
            bridge=SimpleNamespace(set_bridge=lambda bridge: self.assertEqual(bridge, "native"))
        )

        def import_module(name):
            return {"vbench.utils": fake_utils, "decord": fake_decord}[name]

        with (
            patch("scene.metric.verify_upstream"),
            patch("importlib.import_module", side_effect=import_module),
        ):
            indices, frames = decode_video(Path("/tmp/video.mp4"), num_frames=3)

        self.assertEqual(indices, source_indices)
        self.assertEqual([frame.array for frame in frames], ["array-0", "array-3", "array-7"])
        self.assertEqual([frame.mode for frame in frames], ["RGB", "RGB", "RGB"])

    def test_diagnostics_use_the_values_from_frame_scoring(self):
        scorer = MappingScorer()
        result = evaluate_repaired_video(
            Path("/tmp/video.mp4"),
            {"prompt": "a person at the beach", "dimension_metadata": {"scene": "beach"}},
            scorer,
            "environment_grounded",
            frames=[FakeImage()],
            sampled_frame_indices=[17],
        )
        diagnostic = result["diagnostics"]["frame_diagnostics"][0]
        self.assertEqual(scorer.last_scene_text, "beach")
        self.assertEqual(diagnostic["frame_index"], 17)
        self.assertEqual(diagnostic["global_score"], 0.9)
        self.assertEqual(diagnostic["regional_scores"], [0.8, 0.82, 0.78, 0.84])
        self.assertEqual(diagnostic["environment_mean"], 0.81)
        self.assertAlmostEqual(diagnostic["final_frame_score"], 0.729)
        self.assertEqual(diagnostic["scene_label"], "beach")
        self.assertEqual(diagnostic["mode"], "environment_grounded")
        self.assertEqual(result["diagnostics"]["sampled_frame_indices"], [17])

    def test_global_mode_uses_only_the_global_view_with_the_same_scorer(self):
        scorer = MappingScorer()
        result = evaluate_repaired_video(
            Path("/tmp/video.mp4"),
            {"dimension_metadata": {"scene": "beach"}},
            scorer,
            "global",
            frames=[FakeImage()],
            sampled_frame_indices=[17],
        )
        diagnostic = result["diagnostics"]["frame_diagnostics"][0]
        self.assertEqual(diagnostic["regional_scores"], [])
        self.assertEqual(diagnostic["environment_mean"], 1.0)
        self.assertEqual(diagnostic["final_frame_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
