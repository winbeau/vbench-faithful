import unittest
from pathlib import Path

from human_action.backends.audit import parse_action_query
from human_action.backends.vbench import official_target_from_filename


def categories():
    return ("cutting watermelon",) + tuple(f"action {index}" for index in range(1, 400))


class FilenameInvarianceTests(unittest.TestCase):
    def test_audit_target_is_invariant_to_filename(self):
        metadata = {
            "prompt": "A person is cutting watermelon",
            "dimension_metadata": {"target_action": "cutting watermelon"},
        }
        targets = {
            parse_action_query(metadata, categories()).target_action
            for _ in (
                Path("A person is cutting watermelon_0-0.mp4"),
                Path("A person is brushing teeth_0-0.mp4"),
                Path("neutral.mp4"),
            )
        }
        self.assertEqual(targets, {"cutting watermelon"})

    def test_official_target_changes_with_filename(self):
        names = [
            "A person is cutting watermelon_0-0.mp4",
            "A person is brushing teeth_0-0.mp4",
            "neutral.mp4",
        ]
        self.assertEqual(
            [official_target_from_filename(name) for name in names],
            ["cutting watermelon", "brushing teeth", "neutral.mp4"],
        )

    def test_strict_prompt_fallback_and_invalid_target(self):
        query = parse_action_query({"prompt": "A person is cutting watermelon"}, categories())
        self.assertEqual(query.target_source, "prompt_exact_k400")
        with self.assertRaisesRegex(ValueError, "not an exact Kinetics-400"):
            parse_action_query({"prompt": "A person prepares fruit"}, categories())
