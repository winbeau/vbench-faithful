"""Independent CPU recomputation of the 450-source expansion; no model calls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit_vjepa_motion_probe import rows, sha


def interval(values, prompts, seed=20260926, replicates=20000):
    # Aggregate within prompts independently of the production summarizer.
    labels = sorted(set(prompts))
    buckets = [[v for v, p in zip(values, prompts) if p == label] for label in labels]
    sums = np.asarray([sum(b) for b in buckets], dtype=float)
    counts = np.asarray([len(b) for b in buckets], dtype=float)
    sample = np.random.default_rng(seed).integers(len(labels), size=(replicates, len(labels)))
    return np.quantile(sums[sample].sum(1) / counts[sample].sum(1), [.025, .975])


def audit(root, previous_root, human_source):
    root, previous_root = Path(root), Path(previous_root)
    config_path = root / 'code/configs/dynamic-static-jitter/vjepa-expansion450-v1.json'
    extension = json.loads(config_path.read_text())
    parent_path = root / 'code' / extension['parent_config']
    assert sha(parent_path) == extension['parent_config_sha256']
    parent = json.loads(parent_path.read_text())
    summary = json.loads((root / 'analysis/summary.json').read_text())
    assert sha(config_path) == summary['config_sha256']
    assert sha(root / 'inputs/inputs.jsonl') == summary['input_sha256']
    assert sha(previous_root / 'inputs/inputs.jsonl') == extension['previous_inputs_sha256']
    inputs = rows(root / 'inputs/inputs.jsonl')
    ledger = {r['evaluation_id']: r for r in inputs}
    assert len(inputs) == len(ledger) == 1800
    indexed = {(r['base_id'], r['family'], r['seed']): r for r in inputs}
    bases = sorted({r['base_id'] for r in inputs})
    assert len(bases) == 450 and len(indexed) == 1800
    output = {}
    for backend in ('origin', 'vjepa'):
        output[backend] = {}
        directories = sorted((root / 'scores' / backend).glob('shard-*'))
        assert len(directories) == 4
        for directory in directories:
            completion = json.loads((directory / 'completion.json').read_text())
            provenance = json.loads((directory / 'provenance.json').read_text())
            driver = json.loads((directory / 'expansion_driver.json').read_text())
            assert completion['status'] == 'finished' and completion['failed'] == 0
            assert sha(directory / 'scores.jsonl') == completion['scores_sha256']
            assert provenance['config_sha256'] == sha(config_path)
            assert provenance['input_sha256'] == sha(root / 'inputs/inputs.jsonl')
            assert sha(root / 'code/scripts/counterfactual/expand_vjepa_validation.py') == driver['driver_sha256'] == summary['driver_sha256']
            assert driver['training_updates'] == 0
            assert driver['parent_config_sha256'] == extension['parent_config_sha256']
            assert driver['frozen_scoring_script_sha256'] == extension['frozen_validation_script_sha256']
            for name, checksum in provenance['code'].items():
                assert sha(root / 'code' / name) == checksum
            if backend == 'vjepa':
                assert provenance['model']['heads'] == parent['heads']
                assert provenance['model']['pretest_loading_check']['feature_exact']
                assert provenance['model']['training_updates'] == 0
            else:
                assert provenance['model']['upstream']['sha'] == parent['origin_upstream_commit']
                assert not provenance['model']['upstream']['dirty']
                assert provenance['model']['raft_sha256'] == parent['raft_sha256']
            records = rows(directory / 'scores.jsonl')
            assert len(records) == completion['completed'] == completion['expected']
            for r in records:
                key = r['evaluation_id']
                assert key not in output[backend] and r['status'] == 'ok'
                assert r['input_sha256'] == ledger[key]['sha256']
                for field in ('base_id', 'family', 'seed', 'prompt_id', 'generator'):
                    assert r[field] == ledger[key][field]
                output[backend][key] = r
        assert set(output[backend]) == set(ledger)
    errors = []
    def same(actual, reference):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))
        assert error < 1e-12, (actual, reference, error)
        errors.append(error)
    for key, record in output['origin'].items():
        d = record['diagnostics']; sampling = d['sampling']; official = d['official']
        flow = np.array(official['raw_flow_top5_mean'])
        threshold = 6 * min(sampling['frame_shape']) / 256
        count = round(4 * sampling['sampled_frame_count'] / 16)
        assert len(flow) == 15 and np.isfinite(flow).all()
        assert sampling['source_fps'] == 8 and sampling['sampled_source_frame_indices'] == list(range(16))
        assert official['official_threshold'] == threshold and official['official_count_num'] == count
        assert float(np.count_nonzero(flow > threshold) >= count) == record['score']
        for arm in ('joint', 'natural_only'):
            v = output['vjepa'][key][arm]
            same(1 / (1 + np.exp(-v['latent'])), v['score'])
    def score(row, backend):
        key = row['evaluation_id']
        if backend == 'origin':
            return output['origin'][key]['score']
        if backend == 'joint_latent':
            return output['vjepa'][key]['joint']['latent']
        return output['vjepa'][key][backend]['score']
    for uid in bases:
        base, control = (indexed[uid, f, 0] for f in ('original', 'encoding_control'))
        a, b = base['evaluation_id'], control['evaluation_id']
        assert output['origin'][a]['diagnostics'] == output['origin'][b]['diagnostics']
        assert output['vjepa'][a]['feature_sha256'] == output['vjepa'][b]['feature_sha256']
        for arm in ('joint', 'natural_only'):
            assert output['vjepa'][a][arm] == output['vjepa'][b][arm]
    for group, n in [('all450', 450), ('initial90', 90), ('additional360', 360)]:
        chosen = [u for u in bases if group == 'all450' or indexed[u, 'original', 0]['construction_group'] == group]
        assert len(chosen) == n
        prompts = [indexed[u, 'original', 0]['prompt_id'] for u in chosen]
        changes = {}
        for backend in ('origin', 'joint', 'natural_only', 'joint_latent'):
            values = np.array([[score(indexed[u, f, s], backend) for f, s in
                                [('original', 0), ('local_texture_alternating', 1701), ('local_texture_alternating', 2904)]] for u in chosen])
            delta = values[:, 1:] - values[:, :1]
            changes[backend] = delta.mean(1)
            expected = summary['counterfactual'][group][backend]
            for name, value in [('base_mean', values[:, 0].mean()), ('cf_mean', values[:, 1:].mean()),
                                ('delta', delta.mean()), ('mae', abs(delta).mean()), ('base_std', values[:, 0].std()),
                                ('base_min', values[:, 0].min()), ('base_max', values[:, 0].max()),
                                ('largest_drop', delta.min()), ('maximum_absolute_change', abs(delta).max())]:
                same(value, expected[name])
            same(interval(delta.mean(1), prompts), expected['delta_prompt_ci95'])
            same(interval(abs(delta).mean(1), prompts), expected['mae_prompt_ci95'])
        gate = summary['counterfactual'][group]['fixed_scale_numerical_check']
        same(interval(changes['joint'] - .1 * changes['origin'], prompts), gate['margin_prompt_ci95'])
        if changes['origin'].mean() > 0:
            same(gate['allowed_delta'], .1 * changes['origin'].mean())
            assert gate['point_estimate_pass'] == (changes['joint'].mean() <= .1 * changes['origin'].mean())
        assert not gate['absolute_strength_calibrated'] and not gate['overall_goal_complete']
    assert sha(human_source) == parent['human_pairs_sha256']
    actual = []
    with Path(human_source).open() as handle:
        for r in csv.DictReader(handle):
            if r['dimension'] == 'dynamics_degree' and r['split'] == 'test' and r['video_a_uid'] in bases and r['video_b_uid'] in bases:
                actual.append(dict(a=r['video_a_uid'], b=r['video_b_uid'], prompt_id=r['prompt_id'], label=float(r['human_label'])))
    assert len(actual) == 450 and actual == rows(root / 'analysis/human_pairs.jsonl')
    ordered = [r for r in actual if r['label'] != .5]
    ties = [r for r in actual if r['label'] == .5]
    assert len(ordered) == 171 and len(ties) == 279
    prompts = [r['prompt_id'] for r in ordered]
    credits = {}
    for view, family, seed in [('base', 'original', 0), ('1701', 'local_texture_alternating', 1701),
                               ('2904', 'local_texture_alternating', 2904)]:
        credits[view] = {}
        for backend in ('origin', 'joint', 'natural_only'):
            margins = np.array([(score(indexed[r['a'], family, seed], backend) - score(indexed[r['b'], family, seed], backend)) * (2 * r['label'] - 1) for r in ordered])
            credit = (margins > 0).astype(float) + .5 * (margins == 0)
            credits[view][backend] = credit
            expected = summary['human_preferences_by_view'][view][backend]
            assert [int((margins > 0).sum()), int((margins == 0).sum()), int((margins < 0).sum())] == [expected['strict_correct'], expected['predicted_ties'], expected['incorrect']]
            same(credit.mean(), expected['concordance'])
            same(interval(credit, prompts), expected['concordance_prompt_ci95'])
            gaps = [abs(score(indexed[r['a'], family, seed], backend) - score(indexed[r['b'], family, seed], backend)) for r in ties]
            same(np.mean(gaps), expected['human_tie_mean_absolute_gap'])
        difference = credits[view]['joint'] - credits[view]['origin']
        expected = summary['human_preferences_by_view'][view]['joint_minus_origin']
        same(difference.mean(), expected['difference']); same(interval(difference, prompts), expected['prompt_ci95'])
    for seed in ('1701', '2904'):
        for backend in ('origin', 'joint', 'natural_only'):
            change = credits[seed][backend] - credits['base'][backend]
            expected = summary['human_response_change'][seed][backend]
            same(change.mean(), expected['concordance_change']); same(interval(change, prompts), expected['prompt_ci95'])
            assert [int((change > 0).sum()), int((change == 0).sum()), int((change < 0).sum())] == [expected['better_pairs'], expected['unchanged_pairs'], expected['worse_pairs']]
    previous_outputs = {backend: {(r['base_id'], r['family'], r['seed']): r
                        for p in (previous_root / 'scores' / backend).glob('shard-*/scores.jsonl')
                        for r in rows(p) if r['cohort'] == 'holdout'} for backend in ('origin', 'vjepa')}
    compared = 0; max_flow = 0.; max_head = 0.
    for identity, row in indexed.items():
        if identity not in previous_outputs['origin']:
            continue
        prior = previous_outputs['origin'][identity]; current = output['origin'][row['evaluation_id']]
        assert prior['input_sha256'] == row['sha256'] and prior['score'] == current['score']
        max_flow = max(max_flow, float(abs(np.array(prior['diagnostics']['official']['raw_flow_top5_mean']) - current['diagnostics']['official']['raw_flow_top5_mean']).max()))
        prior = previous_outputs['vjepa'][identity]; current = output['vjepa'][row['evaluation_id']]
        assert prior['feature_sha256'] == current['feature_sha256']
        for arm in ('joint', 'natural_only'):
            max_head = max(max_head, abs(prior[arm]['latent'] - current[arm]['latent']), abs(prior[arm]['score'] - current[arm]['score']))
        compared += 1
    assert compared == 720 and max_flow == 0 and max_head <= 1e-6
    return {'status': 'passed', 'origin_formula_verified': 1800, 'sigmoid_scores_verified': 3600,
            'encoding_controls_exact': 450, 'previous_inputs_rechecked': compared,
            'previous_raw_flow_max_abs_error': max_flow, 'previous_head_max_abs_error': max_head,
            'maximum_statistic_error': max(errors), 'human_pairs_verified': 450, 'human_views_verified': 3,
            'ordered_pairs': 171, 'human_ties': 279, 'bootstrap_intervals_independently_verified': True,
            'execution_snapshot_hashes_verified': True, 'summary_sha256': sha(root / 'analysis/summary.json'),
            'auditor_sha256': sha(__file__)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'previous-root', 'human-source', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    target = Path(args.output).resolve(); repo = Path(__file__).resolve().parents[2]
    if target == repo or any(target.is_relative_to(repo / name) for name in ('data', 'results', 'splits', 'runs')):
        raise ValueError('cannot write frozen trees')
    result = audit(args.root, args.previous_root, args.human_source)
    with target.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
