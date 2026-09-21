from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from background_consistency.cli import main


class CliSmokeTests(unittest.TestCase):
    def test_help(self):
        with self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_both_reports_missing_configuration_without_initializing_cuda(self):
        with tempfile.TemporaryDirectory() as root_value:
            root = Path(root_value)
            video = root / "arbitrary.mp4"
            video.touch()
            output = root / "output"
            code = main(["--both", "--video", str(video), "--output", str(output)])
            self.assertEqual(code, 1)
            official = next((output / "background-consistency/vbench").iterdir())
            audit = next((output / "background-consistency/audit").iterdir())
            self.assertEqual(official.name, audit.name)
            for destination in (official, audit):
                result = json.loads((destination / "results.json").read_text(encoding="utf-8"))[0]
                summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["score"])
                self.assertEqual(summary["status"], "failed")
                self.assertTrue((destination / "run.json").is_file())
