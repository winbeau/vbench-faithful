import ast
import unittest
from pathlib import Path

from motion_smoothness.backends.vbench import inspect_upstream
from vbench_audit_core import upstream


class OfficialParityTests(unittest.TestCase):
    def test_locked_upstream_identity_and_config(self):
        state = inspect_upstream()
        self.assertEqual(state.sha, upstream.load_config().sha)
        self.assertFalse(state.dirty)
        self.assertEqual(state.remote, upstream.load_config().url)

    def test_upstream_contract_contains_midpoint_and_rgb_error(self):
        source = (upstream.resolve_upstream_path() / "vbench/motion_smoothness.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        text = source.replace(" ", "")
        self.assertIn("torch.tensor(1/2)", text)
        self.assertIn("cv2.absdiff", text)
        self.assertIn("norm=(255.0-vfi_score)/255.0", text)

    def test_official_extract_frame_is_exact_stride_two(self):
        source = (upstream.resolve_upstream_path() / "vbench/motion_smoothness.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        extract_frame = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "extract_frame"
        )
        range_call = next(
            node
            for node in ast.walk(extract_frame)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"
        )
        self.assertEqual(ast.literal_eval(range_call.args[2]), 2)
        compact = source.replace(" ", "")
        self.assertIn("extract_frame(frames,start_from=0)", compact)
        self.assertIn("extract_frame(ori_frames,start_from=1)", compact)

    def test_official_motion_smoothness_has_no_fps_resampling(self):
        source = (upstream.resolve_upstream_path() / "vbench/motion_smoothness.py").read_text(encoding="utf-8")
        compact = source.replace(" ", "").lower()
        self.assertNotIn("cap_prop_fps", compact)
        self.assertNotIn("fps/8", compact)
        self.assertNotIn("round(fps", compact)
