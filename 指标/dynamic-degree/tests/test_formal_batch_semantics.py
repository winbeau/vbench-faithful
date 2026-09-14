from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from dynamic_degree.formal_results import classify_evaluation_result


def _evidence():
    return {
        'apparent_intensity': 1.0,
        'camera_intensity': 0.4,
        'residual_intensity': 0.6,
        'apparent_coverage': 0.8,
        'camera_coverage': 0.7,
        'residual_coverage': 0.5,
        'component_provenance': {'source_decomposition': True, 'duration_persistence': True},
        'routing': {'target_type': 'UNKNOWN', 'selected_evidence_channel': None},
    }


def _stats_module():
    path = Path(__file__).resolve().parents[3] / 'scripts' / 'run_dynamic_counterfactual_statistics.py'
    spec = importlib.util.spec_from_file_location('dynamic_formal_statistics', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FormalBatchSemanticsTests(unittest.TestCase):
    def test_scalar_success(self):
        self.assertEqual(classify_evaluation_result({'score': 1.0, 'failure_reason': None, 'structured_evidence': None}), 'succeeded_scalar')

    def test_structured_null_score_success_including_unknown_target(self):
        self.assertEqual(classify_evaluation_result({'score': None, 'failure_reason': 'unknown_motion_target', 'structured_evidence': _evidence()}), 'succeeded_structured')

    def test_true_failure(self):
        self.assertEqual(classify_evaluation_result({'score': None, 'failure_reason': 'decode failure', 'structured_evidence': None}), 'failed_or_abstained')

    def test_two_dict_paired_intersection_and_missing_base(self):
        stats = _stats_module()
        self.assertEqual(stats.common_base_ids({'a': 1, 'b': 2}, {'b': 3, 'c': 4}), ['b'])

    def test_multi_dict_intersection(self):
        stats = _stats_module()
        self.assertEqual(stats.common_base_ids({'a': 1, 'b': 2}, {'b': 3, 'c': 4}, {'b': 5, 'd': 6}), ['b'])

    def test_variant_bootstrap_uses_same_per_base_values_as_point(self):
        stats = _stats_module()
        values = {'b1': 0.0, 'b2': 1.0, 'b3': 2.0}
        result = stats.bootstrap_mean(values, 5000, 42)
        self.assertEqual(result['valid_base_ids'], ['b1', 'b2', 'b3'])
        self.assertEqual(result['ci_input_base_ids'], ['b1', 'b2', 'b3'])
        self.assertEqual(result['valid_n'], 3)
        self.assertEqual(result['sampling_unit'], 'base_id')
        self.assertAlmostEqual(result['point_estimate'], 1.0)
        self.assertTrue(result['point_in_ci'])

    def test_constant_spearman_is_null_with_reason(self):
        stats = _stats_module()
        self.assertEqual(stats.safe_spearman([1, 2, 3], [1, 1, 1]), (None, 'constant_input'))

    def test_unsupported_statistic_stays_unsupported(self):
        stats = _stats_module()
        rows = []
        stats.add(rows, 'base', 'motion_coverage', 'official', 'coverage_mae', None, reason='unsupported')
        self.assertFalse(rows[0]['valid'])
        self.assertEqual(rows[0]['reason'], 'unsupported')


if __name__ == '__main__':
    unittest.main()
