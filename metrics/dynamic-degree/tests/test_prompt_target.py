import unittest

from dynamic_degree.prompt_target import MotionTarget, parse_motion_target


class PromptTargetTests(unittest.TestCase):
    def test_subject_camera_generic_both_unknown(self):
        cases = {
            "a person is running through a field": MotionTarget.SUBJECT,
            "the camera pans across a quiet landscape": MotionTarget.CAMERA,
            "a scene with rapid movement": MotionTarget.GENERIC,
            "the camera follows a running dog": MotionTarget.BOTH,
            "a red vase on a table": MotionTarget.UNKNOWN,
            "": MotionTarget.UNKNOWN,
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(parse_motion_target(prompt).target, expected.value)

    def test_explicit_override_wins(self):
        decision = parse_motion_target("the camera pans", "subject")
        self.assertEqual(decision.target, MotionTarget.SUBJECT.value)
        self.assertEqual(decision.source, "explicit_override")

    def test_invalid_override_fails(self):
        with self.assertRaises(ValueError):
            parse_motion_target("anything", "not-a-target")

