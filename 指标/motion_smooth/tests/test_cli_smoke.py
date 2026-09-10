import unittest

from motion_smoothness.cli import build_parser


class CliSmokeTests(unittest.TestCase):
    def test_parser_modes_and_parameters(self):
        args = build_parser().parse_args(["--audit", "--video", "/tmp/video_000.mp4", "--tail-quantile", "0.8", "--tail-weight", "0.1"])
        self.assertTrue(args.audit)
        self.assertEqual(args.tail_quantile, 0.8)
        self.assertEqual(args.tail_weight, 0.1)

