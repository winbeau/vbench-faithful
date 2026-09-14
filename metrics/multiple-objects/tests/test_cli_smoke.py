import unittest
from pathlib import Path

from multiple_objects.cli import build_parser


class CliSmokeTests(unittest.TestCase):
    def test_parser_modes_and_repair_options(self):
        args = build_parser().parse_args(["--audit", "--video", "/tmp/video_000.mp4", "--repair-candidate-threshold", "0.0", "--softmin-beta", "12"])
        self.assertTrue(args.audit)
        self.assertEqual(args.repair_candidate_threshold, 0.0)
        self.assertEqual(args.softmin_beta, 12.0)

    def test_multi_gpu_uses_shared_parser_contract(self):
        args = build_parser().parse_args(["--audit", "--video", "/tmp/video_000.mp4", "--gpu", "0,2"])
        self.assertEqual(args.gpu, "0,2")
        self.assertEqual(args.aggregation, "softmin")

    def test_module_entrypoint_exists(self):
        package_root = Path(__file__).parents[1] / "src/multiple_objects"
        self.assertTrue((package_root / "__main__.py").is_file())
