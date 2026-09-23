"""Independently verify model-training receipts, scores and fixed DEV gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_vjepa_motion_probe import rows, sha


VIEWS = ('base', 'alternating', 'aperiodic', 'still', 'pan8', 'pan32', 'reversal32', 'still_jitter8')


def audit(root, probe_root, training_subdir='training-chunked'):
    root, probe_root = Path(root), Path(probe_root)
    training = root / training_subdir
    config_path = root / 'code/configs/dynamic-static-jitter/vjepa-anchored-v1.json'
    config = json.loads(config_path.read_text())
    summary = json.loads((training / 'summary.json').read_text())
    provenance = json.loads((training / 'provenance.json').read_text())
    assert provenance['config_sha256'] == sha(config_path)
    for path, checksum in provenance['code'].items():
        assert sha(root / 'code' / path) == checksum
    assert sha(probe_root / 'selection/sources.jsonl') == config['probe_sources_sha256']
    assert sha(probe_root / 'selection/pairs.jsonl') == config['probe_pairs_sha256']
    assert sha(training / 'scores.jsonl') == summary['scores_sha256']
    if (training / 'anchored.pt').exists():
        assert sha(training / 'anchored.pt') == summary['head_sha256']
    assert sha(root / 'code/scripts/counterfactual/train_vjepa_anchored_chunked.py') == provenance['trainer_sha256']
    curves = json.loads((training / 'training_curves.json').read_text())
    assert len(curves) == summary['optimizer_steps'] == config['optimization']['steps'] == 300
    assert [r['step'] for r in curves] == list(range(1, 301))
    for r in curves:
        c = config['loss']
        total = (.5 * r['natural_ordered'] + .5 * r['human_ties'] + c['invariance_weight'] * r['native_jitter']
                 + c['zero_weight'] * r['still_zero'] + c['pan_weight'] * (r['weak_pan'] + r['strong_pan'] + r['real_reversal']) / 3
                 + c['native_floor_weight'] * r['native_not_below_still'])
        assert abs(total - r['total']) < 1e-6
    features = []
    for p in sorted((root / 'features').glob('shard-*/features.jsonl')):
        c = json.loads((p.parent / 'completion.json').read_text())
        assert c['status'] == 'finished' and c['failed_sources'] == 0 and c['parity']['prior_base_feature_exact']
        assert sha(p) == c['ledger_sha256']
        features.extend(rows(p))
    assert len(features) == 1350 and len({(r['video_uid'], r['view']) for r in features}) == 1350
    source = [r for r in rows(probe_root / 'selection/sources.jsonl') if r['role'] in ('train', 'validation')]
    assert set((r['video_uid'], r['view']) for r in features) == {(r['video_uid'], v) for r in source for v in VIEWS[3:]}
    pairs = rows(probe_root / 'selection/pairs.jsonl')
    records = rows(training / 'scores.jsonl')
    assert len(records) == 540
    index = {(r['role'], r['arm'], r['video_uid']): r for r in records}
    assert len(index) == 540
    assert set(index) == {(r['role'], a, r['video_uid']) for r in source for a in ('old_joint', 'anchored')}
    maximum_error = 0.
    def same(actual, reference):
        nonlocal maximum_error
        error = float(np.abs(np.asarray(actual) - np.asarray(reference)).max())
        assert error < 1e-12, (actual, reference)
        maximum_error = max(maximum_error, error)
    for role, n in [('train', 210), ('validation', 60)]:
        cohort = [r for r in source if r['role'] == role]
        assert len(cohort) == n
        lookup = {r['video_uid']: i for i, r in enumerate(cohort)}
        pp = [r for r in pairs if r['role'] == role]
        ordered = [r for r in pp if r['label'] != .5]
        for arm in ('old_joint', 'anchored'):
            selected = [index[role, arm, r['video_uid']] for r in cohort]
            assert all(tuple(r['views']) == VIEWS for r in selected)
            q = np.array([r['latent'] for r in selected]); score = np.array([r['scores'] for r in selected])
            same(1 / (1 + np.exp(-q)), score)
            reference = summary['evaluations'][role][arm]
            delta = score[:, 1:3] - score[:, :1]
            same(delta.mean(), reference['native_jitter_delta']); same(abs(delta).mean(), reference['native_jitter_mae'])
            same(delta.min(), reference['native_largest_drop']); same(score[:, 0].std(), reference['native_base_std'])
            for j, view in enumerate(VIEWS):
                same(score[:, j].mean(), reference['mean_by_view'][view])
                same(np.quantile(score[:, j], .95), reference['p95_by_view'][view])
            signed = np.array([(score[lookup[r['a']], :3] - score[lookup[r['b']], :3]) * (2 * r['label'] - 1) for r in ordered])
            assert (signed > 0).sum(0).tolist() == reference['ordered_correct_base_alternating_aperiodic']
            for name, high, low in [('weak_pan_above_still_fraction', 4, 3), ('strong_pan_above_weak_fraction', 5, 4),
                                   ('reversal_above_still_fraction', 6, 3)]:
                same(np.mean(score[:, high] > score[:, low]), reference[name])
    g = config['acceptance']; v = summary['evaluations']['validation']['anchored']
    correct = v['ordered_correct_base_alternating_aperiodic']
    checks = {f'{name}_mean': v['mean_by_view'][name] <= g['validation_still_mean_max'] for name in ('still', 'still_jitter8')}
    checks.update({f'{name}_p95': v['p95_by_view'][name] <= g['validation_still_p95_max'] for name in ('still', 'still_jitter8')})
    for name in ('weak_pan_above_still', 'strong_pan_above_weak', 'reversal_above_still'):
        checks[name] = v[name + '_fraction'] >= g[name + '_fraction_min']
    checks.update(natural_ordering=correct[0] >= g['validation_native_ordered_correct_min'],
                  jitter_ordering=min(correct[1:]) >= correct[0] - g['max_cf_correct_loss_from_clean'],
                  native_invariance=v['native_jitter_mae'] <= g['native_jitter_mae_max'],
                  nonconstant=v['native_base_std'] >= g['native_score_std_min'])
    assert checks == summary['acceptance']['checks'] and all(checks.values()) == summary['acceptance']['all_passed']
    assert summary['encoder_updates'] == 0 and summary['trained_parameters'] == 51393
    assert not summary['posthoc_mapping_changed'] and not summary['acceptance']['formal_goal_complete']
    return {'integrity_checks': 'passed', 'extra_features_verified': 1350, 'score_values_verified': 4320,
            'optimizer_steps_verified': 300, 'parameter_count': 51393, 'encoder_updates': 0,
            'maximum_statistic_error': maximum_error, 'acceptance_independently_recomputed': checks,
            'model_acceptance_passed': all(checks.values()), 'summary_sha256': sha(training / 'summary.json'),
            'auditor_sha256': sha(__file__)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'probe-root', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    target = Path(args.output).resolve(); repo = Path(__file__).resolve().parents[2]
    if target == repo or any(target.is_relative_to(repo / name) for name in ('data', 'results', 'splits', 'runs')):
        raise ValueError('cannot write frozen trees')
    result = audit(args.root, args.probe_root)
    with target.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
