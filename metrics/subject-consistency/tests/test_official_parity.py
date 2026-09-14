import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from subject_consistency.backends.vbench import inspect_upstream, official_subject_consistency, verify_upstream
from subject_consistency.cli import _official_dataset_score
from subject_consistency.metric import evaluate_batch
from subject_consistency.models import OfficialDinoFeatureExtractor
from vbench_audit_core import upstream


class OfficialParityTests(unittest.TestCase):
    def test_locked_upstream_identity(self):
        state = verify_upstream()
        self.assertEqual(state.sha, upstream.load_config().sha)
        self.assertFalse(state.dirty)

    def test_original_formula_matches_adjacent_plus_first_anchor(self):
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F
        features = F.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]), dim=-1)
        previous = [max(0.0, F.cosine_similarity(features[i - 1], features[i], dim=0).item()) for i in range(1, 3)]
        first = [max(0.0, F.cosine_similarity(features[0], features[i], dim=0).item()) for i in range(1, 3)]
        expected = sum(0.5 * a + 0.5 * b for a, b in zip(previous, first)) / 2
        self.assertAlmostEqual(official_subject_consistency(features), expected)

    def test_negative_cosines_are_clamped(self):
        torch = pytest.importorskip("torch")
        features = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(official_subject_consistency(features), 0.25)

    def test_official_and_repair_share_one_injected_extractor(self):
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F
        video = Path("/tmp/video.mp4")
        features = F.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]), dim=-1)

        class Extractor:
            def __init__(self):
                self.calls = 0

            def extract(self, path):
                self.calls += 1
                self.asserted_path = path
                return features

        extractor = Extractor()
        official = evaluate_batch("vbench", [video], {}, "cuda:0", {}, extractor=extractor)
        repaired = evaluate_batch("audit", [video], {}, "cuda:0", {}, extractor=extractor)
        self.assertEqual(extractor.calls, 2)
        self.assertEqual(official[0]["score"], official_subject_consistency(features))
        self.assertNotEqual(official[0]["score"], repaired[0]["score"])

    def test_extractor_uses_all_frames_and_official_preprocessing(self):
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F
        calls = {}
        images = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])

        class Model:
            def to(self, device):
                calls["device"] = device
                return self

            def eval(self):
                return self

            def load_state_dict(self, state_dict, strict):
                calls["load_state_dict"] = (state_dict, strict)
                return self

            def __call__(self, image):
                return torch.cat([image.flatten(1), torch.ones((1, 1))], dim=1)

        torch_module = torch
        functional_module = F
        class Module:
            torch = torch_module
            F = functional_module

            @staticmethod
            def load_video(path, *args, **kwargs):
                calls["load_video"] = (path, args, kwargs)
                return images

            @staticmethod
            def dino_transform(size):
                calls["transform_size"] = size
                return lambda value: value

        config = {"repo_or_dir": "/tmp/dino", "path": "/tmp/dino.pth", "model": "dino_vitb16", "source": "local", "read_frame": False}
        with (
            patch("subject_consistency.backends.vbench.import_official_module", return_value=(Module, object())),
            patch.object(torch, "load", return_value={}),
            patch.object(torch.hub, "load", return_value=Model()) as load,
        ):
            extractor = OfficialDinoFeatureExtractor("cpu", config, Path("/tmp/upstream"))
            features = extractor.extract(Path("/tmp/video.mp4"))
        self.assertEqual(calls["load_video"], ("/tmp/video.mp4", (), {}))
        self.assertEqual(calls["transform_size"], 224)
        self.assertEqual(load.call_args.kwargs["model"], "dino_vitb16")
        self.assertFalse(load.call_args.kwargs["pretrained"])
        self.assertEqual(calls["load_state_dict"], ({}, True))
        self.assertEqual(features.shape, (3, 2))
        self.assertTrue(torch.allclose(features.norm(dim=-1), torch.ones(3)))

    def test_official_dataset_aggregation_preserves_single_multi_difference(self):
        results = [
            {"status": "succeeded", "score": 1.0, "diagnostics": {"num_frames": 2}},
            {"status": "succeeded", "score": 0.0, "diagnostics": {"num_frames": 4}},
        ]
        self.assertEqual(_official_dataset_score(results, 1), (0.25, 4))
        self.assertEqual(_official_dataset_score(results, 2), (0.25, 4))


    def test_detached_head_is_allowed(self):
        config = upstream.load_config()
        hashes = {name: spec.source_sha256 for name, spec in config.dimensions.items()}
        state = upstream.UpstreamState("/tmp/VBench", config.url, "", config.sha, False, (), hashes)
        with patch("vbench_audit_core.upstream.inspect_upstream", return_value=state):
            self.assertEqual(verify_upstream().source_type, "github")

    def test_local_bundle_with_wrong_sha_is_rejected(self):
        config = upstream.load_config()
        hashes = {name: spec.source_sha256 for name, spec in config.dimensions.items()}
        state = upstream.UpstreamState("/tmp/VBench", config.url, "", "0" * 40, False, (), hashes)
        with patch("vbench_audit_core.upstream.inspect_upstream", return_value=state):
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                verify_upstream()

    def test_local_bundle_with_dirty_worktree_is_rejected(self):
        config = upstream.load_config()
        hashes = {name: spec.source_sha256 for name, spec in config.dimensions.items()}
        state = upstream.UpstreamState("/tmp/VBench", config.url, "", config.sha, True, (), hashes)
        with patch("vbench_audit_core.upstream.inspect_upstream", return_value=state):
            with self.assertRaisesRegex(RuntimeError, "checkout is dirty"):
                verify_upstream()

    def test_non_git_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                inspect_upstream(Path(root))

if __name__ == "__main__":
    unittest.main()
