import ast
import json
import os
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from dynamic_degree.backends.vbench import (
    OfficialDynamicEvaluator,
    UPSTREAM_PATH,
    evaluate_official_reference,
    official_check_move,
    official_parameters,
    official_top5_mean,
    verify_upstream,
)
from dynamic_degree.metric import evaluate_vbench_batch
from dynamic_degree.schemas import OfficialVideoResult


def upstream_method(name):
    source_path = UPSTREAM_PATH / "vbench/dynamic_degree.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DynamicDegree")
    method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace[name]


class OfficialParityTests(unittest.TestCase):
    def test_locked_upstream_identity(self):
        state = verify_upstream()
        self.assertEqual(state.sha, "13dee903cc97e2633ed6e8f50dea61bc90717935")
        self.assertFalse(state.dirty)

    def test_top5_matches_locked_upstream_method(self):
        flow = np.zeros((16, 24, 2), dtype=np.float32)
        flow[..., 0] = np.arange(16 * 24, dtype=np.float32).reshape(16, 24) / 10.0
        upstream = upstream_method("get_score")
        image = torch.zeros((1, 3, 16, 24))
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1)[None]
        self.assertAlmostEqual(official_top5_mean(flow), upstream(None, image, flow_tensor), places=7)

    def test_parameters_and_check_move_match_locked_upstream(self):
        upstream_set = upstream_method("set_params")
        upstream_check = upstream_method("check_move")
        holder = types.SimpleNamespace()
        frame = torch.zeros((1, 3, 240, 320))
        upstream_set(holder, frame, 2)
        threshold, count_num = official_parameters(tuple(frame.shape), 2)
        self.assertEqual(holder.params, {"thres": threshold, "count_num": count_num})
        scores = [0.0]
        self.assertEqual(official_check_move(scores, threshold, count_num), upstream_check(holder, scores))
        self.assertTrue(official_check_move(scores, threshold, count_num))

    def test_reference_metadata_adapter(self):
        video = Path("/tmp/video_000.mp4")
        metadata = {video.name: {"video": video.name, "prompt": "a person running"}}

        def fake_compute(metadata_path, device, submodules):
            payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["dimension"], ["dynamic_degree"])
            self.assertEqual(payload[0]["video_list"], [str(video.resolve())])
            self.assertEqual(submodules, {"model": "/tmp/mock.pth"})
            return 1.0, [{"video_path": str(video), "video_results": True}]

        raw, state = evaluate_official_reference(
            [video], metadata, "cuda:0", Path("/tmp/mock.pth"), compute=fake_compute
        )
        self.assertIsNone(state)
        self.assertEqual(raw[0], 1.0)

    def test_official_batch_isolates_corrupt_video(self):
        videos = [Path("/tmp/video_000.mp4"), Path("/tmp/video_001.mp4")]
        metadata = {video.name: {"video": video.name, "prompt": "a person running"} for video in videos}

        class FakeEvaluator:
            def evaluate_video(self, video):
                if video.name == "video_000.mp4":
                    raise RuntimeError("corrupt video")
                return OfficialVideoResult(
                    video=str(video), sampled_source_frame_indices=(0, 2), timestamps=(0.0, 0.125),
                    timestamp_source="nominal_fps_diagnostic_only", source_fps=16.0,
                    sampling_interval=2, frame_shape=(64, 64), sampled_frame_count=2,
                    raw_flow_top5_mean=(8.0,), official_threshold=1.5,
                    official_moving_flags=(True,), official_moving_count=1,
                    official_count_num=0, official_video_boolean=True,
                )

        results = evaluate_vbench_batch(
            videos, metadata, "cuda:0", Path("/tmp/mock.pth"), evaluator=FakeEvaluator()
        )
        self.assertEqual([result["status"] for result in results], ["failed", "succeeded"])
        self.assertIn("corrupt video", results[0]["failure_reason"])

    @unittest.skipUnless(
        os.environ.get("VBENCH_AUDIT_REAL_DYNAMIC_PARITY") == "1",
        "requires real video, CUDA, locked RAFT Things weight",
    )
    def test_real_video_parity(self):
        video = Path(os.environ["VBENCH_AUDIT_REAL_DYNAMIC_PARITY_VIDEO"]).resolve()
        weight = Path(os.environ.get("VBENCH_AUDIT_RAFT_WEIGHT", "")).resolve()
        gpu_id = int(os.environ.get("VBENCH_AUDIT_REAL_DYNAMIC_PARITY_GPU", "0"))
        self.assertTrue(video.is_file())
        self.assertTrue(weight.is_file())
        self.assertTrue(torch.cuda.is_available())
        device = torch.device(f"cuda:{gpu_id}")
        evaluator = OfficialDynamicEvaluator(device, weight)
        extracted = evaluator.evaluate_video(video)
        captured_scores = []
        original_get_score = evaluator.dynamic.get_score

        def capture_score(image, flow):
            score = original_get_score(image, flow)
            captured_scores.append(score)
            return score

        evaluator.dynamic.get_score = capture_score
        direct = evaluator.dynamic.infer(str(video))
        self.assertEqual(extracted.official_video_boolean, direct)
        for expected, actual in zip(extracted.raw_flow_top5_mean, captured_scores):
            self.assertAlmostEqual(expected, actual, places=6)
        self.assertEqual(extracted.sampled_frame_count, len(captured_scores) + 1)
