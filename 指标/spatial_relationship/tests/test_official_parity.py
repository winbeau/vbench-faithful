import ast
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_relationship.backends.vbench import OfficialGritDetector, OfficialVBenchEvaluator, UPSTREAM_PATH, UpstreamState, detections_from_instances, evaluate_official, verify_upstream
from spatial_relationship.metric import evaluate_vbench_batch, parse_query
from spatial_relationship.relation import official_position_score


def load_locked_upstream_position_score():
    source_path = UPSTREAM_PATH / "vbench/spatial_relationship.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_position_score")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["get_position_score"]


class OfficialParityTests(unittest.TestCase):
    def test_locked_upstream_identity(self):
        state = verify_upstream()
        self.assertEqual(state.sha, "13dee903cc97e2633ed6e8f50dea61bc90717935")
        self.assertEqual(state.source_sha256, "26f09a111dac904d1a92ae5bd1c332a029cebd22610d62376a09d7d634950270")
        self.assertFalse(state.dirty)

    def test_local_bundle_remote_is_provenance_not_identity(self):
        local = UpstreamState("/tmp/upstream", "file:///bundle", "master", "13dee903cc97e2633ed6e8f50dea61bc90717935", False, (), "26f09a111dac904d1a92ae5bd1c332a029cebd22610d62376a09d7d634950270")
        with patch("spatial_relationship.backends.vbench.inspect_upstream", return_value=local):
            self.assertEqual(verify_upstream(Path("/tmp/upstream")).remote, "file:///bundle")

    def test_wrong_commit_is_rejected(self):
        wrong = UpstreamState("/tmp/upstream", "file:///bundle", "master", "wrong", False, (), "26f09a111dac904d1a92ae5bd1c332a029cebd22610d62376a09d7d634950270")
        with patch("spatial_relationship.backends.vbench.inspect_upstream", return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, "upstream SHA mismatch"):
                verify_upstream(Path("/tmp/upstream"))

    def test_missing_official_source_is_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(FileNotFoundError, "source is missing"):
                verify_upstream(Path(root))

    def test_geometry_matches_locked_upstream_source(self):
        upstream_score = load_locked_upstream_position_score()
        boxes = [
            ((0, 0, 2, 2), (8, 0, 10, 2)),
            ((8, 0, 10, 2), (0, 0, 2, 2)),
            ((0, 0, 10, 10), (6, 0, 16, 10)),
            ((0, 0, 2, 2), (0, 8, 2, 10)),
        ]
        for relation in ("on the left of", "on the right of", "on the top of", "on the bottom of"):
            for left, right in boxes:
                with self.subTest(relation=relation, left=left, right=right):
                    self.assertAlmostEqual(official_position_score(relation, left, right), upstream_score(relation, left, right))

    def test_plumbing_adapter_preserves_official_metadata_and_raw_result(self):
        video = Path("/tmp/video_000.mp4")
        metadata = {
            video.name: {
                "video": video.name,
                "prompt": "a cat on the left of a dog",
                "dimension_metadata": {"object_a": "cat", "object_b": "dog", "relationship": "on the left of"},
            }
        }
        expected = (0.75, [{"video_path": str(video.resolve()), "video_results": 0.75, "frame_results": [1, 0.5]}])

        def fake_compute(metadata_path, device, submodules):
            data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            relation = data[0]["auxiliary_info"]["spatial_relationship"]["spatial_relationship"]
            self.assertEqual(relation, {"object_a": "cat", "object_b": "dog", "relationship": "on the left of"})
            self.assertEqual(submodules, {"model_weight": "/tmp/mock.pth"})
            return expected

        raw, state = evaluate_official([video], metadata, parse_query, "cuda:0", Path("/tmp/mock.pth"), compute=fake_compute)
        self.assertIsNone(state)
        self.assertIs(raw, expected)

    def test_detector_confidence_is_preserved_when_instances_expose_scores(self):
        class Value:
            def __init__(self, value):
                self.value = value
            def item(self):
                return self.value
            def tolist(self):
                return list(self.value)

        class Tensor:
            def __init__(self, values):
                self.values = values
            def detach(self):
                return self
            def cpu(self):
                return self
            def __getitem__(self, index):
                return Value(self.values[index])

        class Instances:
            pred_object_descriptions = type("Descriptions", (), {"data": ["cat", "dog"]})()
            pred_boxes = type("Boxes", (), {"tensor": Tensor([[0, 0, 2, 2], [8, 0, 10, 2]])})()
            scores = Tensor([0.95, 0.51])
            @staticmethod
            def has(name):
                return name == "scores"

        self.assertEqual(
            detections_from_instances(Instances()),
            [("cat", [0, 0, 2, 2], 0.95), ("dog", [8, 0, 10, 2], 0.51)],
        )

    def test_detector_converts_sampled_tensor_frames_to_numpy(self):
        import numpy as np
        import torch

        class VideoTensor:
            def size(self):
                return (16, 3, 2, 2)
            def permute(self, *order):
                return self
            def numpy(self):
                return np.zeros((16, 2, 2, 3), dtype=np.uint8)

        class Instances:
            pred_object_descriptions = type("Descriptions", (), {"data": []})()
            pred_boxes = type("Boxes", (), {"tensor": torch.empty((0, 4))})()
            scores = torch.empty((0,))
            @staticmethod
            def has(name):
                return name == "scores"

        class Model:
            calls = 0
            def run_det_tensor(self, frame):
                self.calls += 1
                self_outer.assertIsInstance(frame, np.ndarray)
                return {"instances": Instances()}, None

        self_outer = self
        detector = OfficialGritDetector.__new__(OfficialGritDetector)
        detector.module = type("Module", (), {"load_video": staticmethod(lambda *_args, **_kwargs: VideoTensor()), "torch": torch})()
        detector.model = Model()
        utils = type("Utils", (), {"VideoReader": type("Reader", (), {"__init__": lambda self, *_args, **_kwargs: None, "__len__": lambda self: 16}), "get_frame_indices": staticmethod(lambda *args, **kwargs: list(range(16)))})()
        with patch("spatial_relationship.backends.vbench.importlib.import_module", return_value=utils):
            indices, predictions = detector.detect_video(Path('/tmp/sample.mp4'))
        self.assertEqual(detector.model.calls, 16)
        self.assertEqual(indices, list(range(16)))
        self.assertEqual(predictions, [[] for _ in range(16)])

    def test_official_batch_isolates_one_video_failure(self):
        videos = [Path("/tmp/video_000.mp4"), Path("/tmp/video_001.mp4")]
        metadata = {
            video.name: {
                "video": video.name,
                "prompt": "a cat on the left of a dog",
                "dimension_metadata": {"object_a": "cat", "object_b": "dog", "relationship": "on the left of"},
            }
            for video in videos
        }

        class FakeEvaluator:
            def evaluate_video(self, video, metadata_item, query):
                if video.name == "video_000.mp4":
                    raise RuntimeError("corrupt video")
                return 1.0, [{"video_path": str(video), "video_results": 1.0, "frame_results": [1.0]}]

        results = evaluate_vbench_batch(videos, metadata, "cuda:0", Path("/tmp/mock.pth"), evaluator=FakeEvaluator())
        self.assertEqual([item["status"] for item in results], ["failed", "succeeded"])
        self.assertIn("corrupt video", results[0]["failure_reason"])
        self.assertEqual(results[1]["score"], 1.0)

    @unittest.skipUnless(os.environ.get("VBENCH_AUDIT_REAL_SPATIAL_PARITY") == "1", "requires real video, CUDA, Detectron2, GRiT and official weight")
    def test_real_video_parity(self):
        import torch

        video = Path(os.environ["VBENCH_AUDIT_REAL_SPATIAL_PARITY_VIDEO"]).resolve()
        metadata_path = Path(os.environ["VBENCH_AUDIT_REAL_SPATIAL_PARITY_METADATA"]).resolve()
        weight = Path(os.environ["VBENCH_AUDIT_GRIT_WEIGHT"]).resolve()
        gpu_id = int(os.environ.get("VBENCH_AUDIT_REAL_SPATIAL_PARITY_GPU", "0"))
        self.assertTrue(video.is_file())
        self.assertTrue(metadata_path.is_file())
        self.assertTrue(weight.is_file())
        self.assertTrue(torch.cuda.is_available())

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        entries = payload.get("videos") if isinstance(payload, dict) else None
        self.assertIsInstance(entries, list)
        item = next(entry for entry in entries if Path(entry["video"]).name == video.name)
        metadata = {video.name: item}
        device = torch.device(f"cuda:{gpu_id}")

        reference_raw, state = evaluate_official([video], metadata, parse_query, device, weight)
        self.assertEqual(state.sha, "13dee903cc97e2633ed6e8f50dea61bc90717935")
        torch.cuda.empty_cache()
        adapter = OfficialVBenchEvaluator(device, weight)
        extracted_raw = adapter.evaluate_video(video, item, parse_query(item))

        self.assertAlmostEqual(float(reference_raw[0]), float(extracted_raw[0]), places=7)
        self.assertEqual(len(reference_raw[1]), len(extracted_raw[1]))
        for expected, actual in zip(reference_raw[1], extracted_raw[1]):
            self.assertEqual(Path(expected["video_path"]).resolve(), Path(actual["video_path"]).resolve())
            self.assertEqual(expected["frame_results"], actual["frame_results"])
            self.assertAlmostEqual(float(expected["video_results"]), float(actual["video_results"]), places=7)
