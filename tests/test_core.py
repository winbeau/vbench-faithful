import json
import tempfile
import unittest
from pathlib import Path

from vbench_audit_core.devices import parse_gpu, round_robin_shards
from vbench_audit_core.errors import InputError
from vbench_audit_core.inputs import enumerate_videos, load_metadata


class CoreContractTests(unittest.TestCase):
    def test_gpu_parser_and_round_robin(self):
        self.assertEqual(parse_gpu(None), [0])
        self.assertEqual(parse_gpu("0,2,4"), [0, 2, 4])
        self.assertEqual(round_robin_shards(list(range(8)), [0, 2, 4]), {0: [0, 3, 6], 2: [1, 4, 7], 4: [2, 5]})
        for value in ("", "0,0", "-1", "x", "0,"):
            with self.subTest(value=value):
                with self.assertRaises(InputError):
                    parse_gpu(value)

    def test_batch_order_and_validation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "video_1000.mp4").touch()
            (path / "video_003.mp4").touch()
            (path / "notes.json").touch()
            self.assertEqual([p.name for p in enumerate_videos(None, root)], ["video_003.mp4", "video_1000.mp4"])
            (path / "bad.mp4").touch()
            with self.assertRaises(InputError):
                enumerate_videos(None, root)

    def test_metadata_duplicate_and_lookup(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            video = path / "video_000.mp4"
            video.touch()
            metadata = path / "metadata.json"
            metadata.write_text(json.dumps({"videos": [{"video": "video_000.mp4", "prompt": "a cat"}]}), encoding="utf-8")
            loaded = load_metadata(metadata, [video])
            self.assertEqual(loaded["video_000.mp4"]["prompt"], "a cat")
            metadata.write_text(json.dumps({"videos": [{"video": "video_000.mp4"}, {"video": "video_000.mp4"}]}), encoding="utf-8")
            with self.assertRaises(InputError):
                load_metadata(metadata, [video])
