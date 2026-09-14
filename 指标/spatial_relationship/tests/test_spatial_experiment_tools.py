from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from spatial_relationship.serialization import to_jsonable


REPO = Path(__file__).resolve().parents[3]


def load_script(name):
    path = REPO / 'scripts' / name
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SpatialExperimentToolTests(unittest.TestCase):
    def test_spatial_to_jsonable_preserves_numpy_values_and_zero_score(self):
        payload = {
            "frame_index": np.int64(7),
            "score": np.float32(0.0),
            "relation_valid": np.bool_(True),
            "nested": [np.float64(.25), {"boxes": np.asarray([[1, 2, 3, 4]])}],
        }
        converted = to_jsonable(payload)
        self.assertEqual(converted["frame_index"], 7)
        self.assertEqual(converted["score"], 0.0)
        self.assertIsInstance(converted["score"], float)
        self.assertIs(converted["relation_valid"], True)
        self.assertEqual(converted["nested"][1]["boxes"], [[1, 2, 3, 4]])
        self.assertEqual(json.loads(json.dumps(converted)), converted)

    def test_numpy_diagnostics_are_json_serializable(self):
        runner = load_script('run_spatial_experiments.py')
        encoded = json.dumps({'frame_index': np.int64(7), 'score': np.float32(.5)}, default=runner.json_default)
        self.assertEqual(json.loads(encoded), {'frame_index': 7, 'score': .5})

    def test_same_video_distinct_queries_are_distinct_keys_and_resume(self):
        runner = load_script('run_spatial_experiments.py')
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = root / 'sample.mp4'
            video.touch()
            rows = [
                {'base_id': 'b1', 'derived_id': 'd1', 'intervention_family': 'role_swap', 'intervention_level': level, 'subject_a': a, 'relation': 'left', 'subject_b': b, 'expected_relation': expected, 'video_path': str(video)}
                for level, a, b, expected in (('correct', 'cat', 'dog', True), ('swapped', 'dog', 'cat', False))
            ]
            manifest = root / 'manifest.jsonl'
            manifest.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
            normalized = runner.validate_manifest(runner.read_jsonl(manifest), manifest, 'role_swap', None)
            self.assertEqual(len({runner.manifest_key(row, 'role_preserving') for row in normalized}), 2)
            existing = {runner.manifest_key(normalized[0], 'role_preserving')}
            self.assertEqual(runner.pending_rows(normalized, existing, 'role_preserving'), [normalized[1]])

    def test_full_temporal_is_explicitly_not_ready(self):
        runner = load_script('run_spatial_experiments.py')
        with self.assertRaises(runner.VariantNotReady):
            runner.variant_mode('full_temporal')

    def test_detection_diagnostics_preserve_selected_pair_and_confidence(self):
        runner = load_script('run_spatial_experiments.py')
        diagnostics = {'frame_results': [{'frame_index': 0, 'subject_candidate_ids': [1], 'object_candidate_ids': [2], 'assigned_subject_id': 1, 'assigned_object_id': 2, 'selected_subject_box': [0, 0, 2, 2], 'selected_object_box': [8, 0, 10, 2], 'detections': [{'detection_id': 1, 'confidence': .9}, {'detection_id': 2, 'confidence': .8}]}]}
        result = runner.detection_diagnostics(diagnostics, 'cat', 'dog')
        self.assertTrue(result['detected_a'])
        self.assertTrue(result['detected_b'])
        self.assertEqual(result['detection_confidence'][0]['subject'], .9)
        self.assertEqual(result['selected_instance_pair'][0]['object_id'], 2)

    def test_temporal_summary_uses_base_group_and_handles_constant(self):
        summary = load_script('summarize_spatial_results.py')
        rows = [{'intervention_level': level, 'score': 1.0, 'failure_or_abstention': None, 'detected_a': True, 'detected_b': True} for level in (0, 25, 50, 75, 100)]
        result = summary.summarize_group('b1', 'temporal_persistence', 'role_preserving', rows, False)
        self.assertIsNone(result['spearman_rho'])
        self.assertEqual(result['spearman_reason'], 'constant_input')
        self.assertEqual(result['strict_order_rate'], 0.0)

    def test_directional_and_role_swap_pairing(self):
        summary = load_script('summarize_spatial_results.py')
        base = {'failure_or_abstention': None, 'detected_a': True, 'detected_b': True}
        directional = summary.summarize_group('b1', 'directional_inversion', 'role_preserving', [{**base, 'intervention_level': 'original', 'score': 1.0}, {**base, 'intervention_level': 'hflip', 'score': 0.0}], False)
        swapped = summary.summarize_group('b1', 'role_swap', 'role_preserving', [{**base, 'intervention_level': 'correct', 'score': .8}, {**base, 'intervention_level': 'swapped', 'score': .2}], False)
        self.assertEqual(directional['paired_score_gap'], 1.0)
        self.assertTrue(directional['inversion_pass'])
        self.assertAlmostEqual(swapped['correct_vs_swapped_gap'], .6)
        self.assertTrue(swapped['role_swap_correct'])

    def test_multi_instance_and_detection_control(self):
        summary = load_script('summarize_spatial_results.py')
        base = {'failure_or_abstention': None, 'detected_a': True, 'detected_b': True}
        result = summary.summarize_group('b1', 'multi_instance', 'role_preserving', [{**base, 'intervention_level': 'clean', 'score': .9}, {**base, 'intervention_level': 'distractor', 'score': .7}], detection_conditioned=False)
        self.assertTrue(result['target_pair_success'])
        self.assertAlmostEqual(result['distractor_robustness_gap'], .2)


if __name__ == '__main__':
    unittest.main()
