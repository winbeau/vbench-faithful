import json
import tempfile
import unittest
from pathlib import Path

from dynamic_degree.evaluation_records import EvaluationRecordWriter, RecordValidationError


def record(**overrides):
    value = {
        'base_id': 'base_a', 'derived_id': 'derived_a', 'split': 'dev', 'dimension': 'dynamic_degree',
        'intervention_family': 'fps_resampling', 'intervention_level': 8, 'prompt': 'a person runs',
        'target_type': 'GENERIC', 'expected_relation': 'invariance', 'metric_name': 'dynamic_degree',
        'metric_variant': 'official', 'score': 1.0, 'seed': None, 'failure_or_abstention': None,
    }
    value.update(overrides)
    return value


class EvaluationRecordTests(unittest.TestCase):
    def test_success_failure_and_null_score(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EvaluationRecordWriter(Path(root) / 'records.jsonl')
            self.assertTrue(writer.append(record()))
            self.assertTrue(writer.append(record(derived_id='failure', score=None, failure_or_abstention='decode_error')))
            self.assertTrue(writer.append(record(derived_id='structured', metric_variant='full', score=None)))
            self.assertEqual(len(writer.path.read_text().splitlines()), 3)

    def test_resume_and_duplicate_skip(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'records.jsonl'
            self.assertTrue(EvaluationRecordWriter(path).append(record()))
            self.assertFalse(EvaluationRecordWriter(path).append(record()))

    def test_identity_is_not_physical_path(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EvaluationRecordWriter(Path(root) / 'records.jsonl')
            self.assertTrue(writer.append(record(derived_id='same_file_a')))
            self.assertTrue(writer.append(record(derived_id='same_file_b')))

    def test_variant_and_target_are_distinct_resume_keys(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EvaluationRecordWriter(Path(root) / 'records.jsonl')
            self.assertTrue(writer.append(record()))
            self.assertTrue(writer.append(record(metric_variant='full')))
            self.assertTrue(writer.append(record(target_type='SUBJECT')))

    def test_rejects_malformed_record(self):
        with tempfile.TemporaryDirectory() as root:
            writer = EvaluationRecordWriter(Path(root) / 'records.jsonl')
            with self.assertRaises(RecordValidationError):
                writer.append({'base_id': 'only'})

    def test_rejects_duplicate_existing_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'records.jsonl'
            path.write_text(json.dumps(record()) + '\n' + json.dumps(record()) + '\n')
            with self.assertRaises(RecordValidationError):
                EvaluationRecordWriter(path)
