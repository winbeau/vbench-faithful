import unittest
from pathlib import Path

from multiple_objects.backends.vbench import OFFICIAL_NUM_FRAMES, OFFICIAL_THRESHOLD, UPSTREAM_PATH, OfficialGrITDetector, inspect_upstream
from multiple_objects.conditions import parse_target_objects
from multiple_objects.metric import official_frame_decision


class OfficialParityTests(unittest.TestCase):
    def test_locked_upstream_and_constants(self):
        state = inspect_upstream()
        self.assertEqual(state["sha"], "13dee903cc97e2633ed6e8f50dea61bc90717935")
        self.assertFalse(state["dirty"])
        self.assertEqual(OFFICIAL_NUM_FRAMES, 16)
        self.assertEqual(OFFICIAL_THRESHOLD, 0.5)

    def test_target_and_frame_conjunction(self):
        metadata = {"auxiliary_info": {"object": "cat and dog"}}
        self.assertEqual(parse_target_objects(metadata), ("cat", "dog"))
        self.assertTrue(official_frame_decision(("cat", "dog"), [("cat", None), ("dog", None)]))
        self.assertFalse(official_frame_decision(("cat", "dog"), [("cat", None)]))
        self.assertFalse(official_frame_decision(("cat", "dog"), [("Cat", None), ("dog", None)]))

    def test_upstream_strict_threshold_boundary(self):
        targets = ("cat", "dog")
        for confidence, expected in ((0.49, False), (0.50, False), (0.51, True)):
            detections = [
                {"label": "cat", "confidence": 0.9},
                {"label": "dog", "confidence": confidence},
            ]
            self.assertEqual(official_frame_decision(targets, detections), expected)

    def test_upstream_calls_16_frame_video_and_and_split(self):
        source = (UPSTREAM_PATH / "vbench/multiple_objects.py").read_text(encoding="utf-8")
        self.assertIn("num_frames=16", source)
        self.assertIn("split(' and ')", source)
        self.assertIn("success_frame_count / frame_count", source)
        roi_source = (UPSTREAM_PATH / "vbench/third_party/grit_src/grit/modeling/roi_heads/grit_roi_heads.py").read_text(encoding="utf-8")
        self.assertIn("scores_per_stage =", roi_source)
        self.assertIn("scores = [(s * ps[:, None]) ** 0.5", roi_source)
        self.assertIn("filter_mask = scores > score_thresh", roi_source)
        self.assertIn("result.scores = scores", roi_source)
        self.assertIn("pred_instance.scores = (pred_instance.scores *", roi_source)

    def test_grit_field_extraction_preserves_index_alignment(self):
        class Tensor:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class Boxes:
            tensor = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        class Descriptions:
            data = ["cat", "dog"]

        class Instances:
            det_obj = Descriptions()
            pred_boxes = Boxes()
            scores = Tensor([0.25, 0.75])

            def has(self, name):
                return hasattr(self, name)

            def __len__(self):
                return 2

        threshold_scores = Tensor([0.49, 0.51])
        extracted = OfficialGrITDetector._extract(
            {"instances": Instances()}, threshold_scores
        )
        self.assertEqual([
            (item.label, item.official_threshold_score, item.box) for item in extracted
        ], [
            ("cat", 0.49, (1.0, 2.0, 3.0, 4.0)),
            ("dog", 0.51, (5.0, 6.0, 7.0, 8.0)),
        ])
        self.assertEqual(
            [item.final_instance_confidence for item in extracted], [0.25, 0.75]
        )
