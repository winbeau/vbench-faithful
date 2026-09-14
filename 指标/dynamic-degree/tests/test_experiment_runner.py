import unittest
from pathlib import Path
from unittest.mock import patch

from dynamic_degree.experiment_runner import VariantNotReady, evaluate_variant


class RunnerTests(unittest.TestCase):
    @patch('dynamic_degree.experiment_runner.evaluate_audit_batch')
    def test_repair_variants_are_dispatched_with_config(self, backend):
        backend.return_value = [{'backend': 'audit'}]
        for variant in ('time_only', 'source_only', 'duration_only', 'source_time', 'full'):
            self.assertEqual(evaluate_variant(Path('/tmp/x.mp4'), '', 'GENERIC', variant, device='cuda:0', model_weight=Path('/tmp/x'))['backend'], 'audit')

    @patch('dynamic_degree.experiment_runner.evaluate_vbench_batch')
    def test_official_uses_existing_backend(self, backend):
        backend.return_value = [{'backend': 'vbench'}]
        result = evaluate_variant(Path('/tmp/x.mp4'), 'prompt', 'GENERIC', 'official', device='cuda:0', model_weight=Path('/tmp/x'))
        self.assertEqual(result['backend'], 'vbench')
