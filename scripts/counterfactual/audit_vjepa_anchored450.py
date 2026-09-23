"""Independent CPU verification of anchored450 identities and reported statistics."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .audit_vjepa_expansion import interval
from .audit_vjepa_motion_probe import rows, sha


def audit(root, previous, human_path):
    root, previous = Path(root), Path(previous)
    cp = root / 'code/configs/dynamic-static-jitter/vjepa-anchored450-v1.json'
    config = json.loads(cp.read_text())
    summary = json.loads((root / 'analysis/summary.json').read_text())
    assert sha(cp) == summary['config_sha256']
    assert config['head_sha256'] == summary['head_sha256']
    assert sha(root / 'code/scripts/counterfactual/summarize_vjepa_anchored450.py') == summary['summarizer_sha256']
    ip = previous / 'inputs/inputs.jsonl'
    assert sha(ip) == config['inputs_sha256'] == summary['inputs_sha256']
    inputs = rows(ip); ledger = {r['evaluation_id']: r for r in inputs}
    assert len(ledger) == len(inputs) == 1800
    index = {(r['base_id'], r['family'], r['seed']): r['evaluation_id'] for r in inputs}
    bases = sorted({r['base_id'] for r in inputs})
    views = [('original', 0), ('local_texture_alternating', 1701), ('local_texture_alternating', 2904)]
    assert len(bases) == 450
    assert set(index) == {(uid, family, seed) for uid in bases for family, seed in views + [('encoding_control', 0)]}
    assert len({r['prompt_id'] for r in inputs}) == 30
    old = {}
    for backend in ('origin', 'vjepa'):
        old[backend] = {}
        for part, checksum in config['previous_scores_sha256'][backend].items():
            path = previous / 'scores' / backend / part / 'scores.jsonl'
            assert sha(path) == checksum
            for r in rows(path):
                key = r['evaluation_id']
                assert key not in old[backend] and r['status'] == 'ok'
                assert r['input_sha256'] == ledger[key]['sha256']
                old[backend][key] = r
        assert set(old[backend]) == set(ledger)
    current = {}; maximum_old_error = 0.
    for n in range(4):
        path = root / 'scores' / f'shard-{n}'
        c = json.loads((path / 'completion.json').read_text())
        p = json.loads((path / 'provenance.json').read_text())
        assert c['status'] == 'finished' and c['failed'] == 0
        assert c['feature_exact_to_previous'] == c['completed'] == c['expected']
        assert sha(path / 'scores.jsonl') == c['scores_sha256']
        assert p['input_sha256'] == sha(ip) and p['config_sha256'] == sha(cp)
        assert p['head_sha256'] == config['head_sha256'] and p['old_head_sha256'] == config['old_head_sha256']
        assert p['training_updates'] == 0 and not p['score_mapping_changed']
        assert p['mode'] == 'fresh_decode_and_encoder_forward_every_input'
        assert p['shard'] == n and p['shards'] == 4
        assert p['pretest_loading_check']['feature_exact']
        assert max(p['pretest_loading_check']['latent_error'].values()) <= 1e-6
        assert p['encoder']['trainable_encoder_parameters'] == 0 and p['encoder']['strict_state_load']
        for name, checksum in p['code'].items():
            assert sha(root / 'code' / name) == checksum
        part = rows(path / 'scores.jsonl')
        assert len(part) == c['completed']
        expected = {r['evaluation_id'] for r in inputs if r['base_id'] in set(bases[n::4])}
        assert {r['evaluation_id'] for r in part} == expected == set(p['expected_ids'])
        for r in part:
            key = r['evaluation_id']; source = ledger[key]
            assert key not in current and r['status'] == 'ok'
            assert r['input_sha256'] == source['sha256'] and r['pixels_sha256'] == source['decoded_pixels_sha256']
            assert r['construction_status'] == source['status'] and r['construction_reason'] == source.get('reason')
            for field in ('base_id', 'family', 'seed', 'prompt_id', 'generator', 'construction_group'):
                assert r[field] == source[field]
            assert r['feature_sha256'] == old['vjepa'][key]['feature_sha256']
            error = abs(r['old_joint']['latent'] - old['vjepa'][key]['joint']['latent'])
            assert error <= 1e-6 and r['old_head_replay_error'] == error
            maximum_old_error = max(maximum_old_error, error)
            current[key] = r
    assert set(current) == set(ledger)
    errors = []
    def same(a, b):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(abs(a - b).max()); errors.append(error)
        assert error < 1e-12, (a, b)
    for key, r in current.items():
        for arm in ('anchored', 'old_joint'):
            same(1 / (1 + np.exp(-r[arm]['latent'])), r[arm]['score'])
        d = old['origin'][key]['diagnostics']; sampling = d['sampling']; original = d['official']
        flow = np.array(original['raw_flow_top5_mean'])
        assert len(flow) == 15 and np.isfinite(flow).all()
        threshold = 6 * min(sampling['frame_shape']) / 256
        count = round(4 * sampling['sampled_frame_count'] / 16)
        assert sampling['source_fps'] == 8 and sampling['sampled_source_frame_indices'] == list(range(16))
        assert float(np.count_nonzero(flow > threshold) >= count) == old['origin'][key]['score']
    def value(uid, family, seed, arm):
        key = index[uid, family, seed]
        if arm == 'origin':
            return old['origin'][key]['score']
        if arm == 'old_joint':
            return old['vjepa'][key]['joint']['score']
        return current[key]['anchored']['latent' if arm == 'anchored_latent' else 'score']
    for uid in bases:
        a, b = (index[uid, family, 0] for family in ('original', 'encoding_control'))
        assert current[a]['feature_sha256'] == current[b]['feature_sha256']
        assert current[a]['anchored'] == current[b]['anchored']
        for arm in ('origin', 'old_joint'):
            assert value(uid, 'original', 0, arm) == value(uid, 'encoding_control', 0, arm)
    pairs = rows(root / 'analysis/pairs.jsonl')
    assert len(pairs) == len({r['base_id'] for r in pairs}) == 450
    for r in pairs:
        for arm in ('origin', 'old_joint', 'anchored', 'anchored_latent'):
            same([value(r['base_id'], f, s, arm) for f, s in views], r[arm])
    for group, recorded in summary['counterfactual'].items():
        selected = [u for u in bases if group == 'all450' or
                    (ledger[index[u, 'original', 0]]['generator'] == group.split(':', 1)[-1] if group.startswith('generator:')
                     else ledger[index[u, 'original', 0]]['construction_group'] == group)]
        assert len(selected) == (450 if group == 'all450' else 90 if group == 'initial90' else 360 if group == 'additional360' else 150)
        prompts = [ledger[index[u, 'original', 0]]['prompt_id'] for u in selected]
        changes = {}
        for arm in ('origin', 'old_joint', 'anchored', 'anchored_latent'):
            v = np.array([[value(u, f, s, arm) for f, s in views] for u in selected])
            d = v[:, 1:] - v[:, :1]; changes[arm] = d.mean(1)
            expected = recorded[arm]
            assert expected['sources'] == len(selected)
            for k, actual in {'base_mean': v[:, 0].mean(), 'cf_mean': v[:, 1:].mean(), 'delta': d.mean(),
                'mae': abs(d).mean(), 'base_std': v[:, 0].std(), 'base_range': [v[:, 0].min(), v[:, 0].max()],
                'base_quantiles_05_25_50_75_95': np.quantile(v[:, 0], [.05, .25, .5, .75, .95]),
                'largest_drop': d.min(), 'largest_increase': d.max(), 'maximum_absolute_change': abs(d).max(),
                'delta_prompt_ci95': interval(d.mean(1), prompts), 'mae_prompt_ci95': interval(abs(d).mean(1), prompts)}.items():
                same(actual, expected[k])
            same(v[:, 1:].mean(0), [expected['cf_by_seed'][s] for s in ('1701', '2904')])
        g = recorded['relative_scale_invariance_check']; od = changes['origin'].mean(); nd = changes['anchored'].mean()
        assert g['origin_increased'] == (od > 0)
        if od > 0:
            same(.1 * od, g['allowed_delta']); same(nd / od, g['new_delta_over_origin'])
            assert g['point_estimate_pass'] == (nd <= .1 * od)
        same(interval(changes['anchored'] - .1 * changes['origin'], prompts), g['margin_prompt_ci95'])
        assert not g['overall_goal_complete'] and not g['absolute_physical_units_aligned']
    assert sha(human_path) == config['human_pairs_sha256']
    with Path(human_path).open() as f:
        human = [dict(a=r['video_a_uid'], b=r['video_b_uid'], prompt_id=r['prompt_id'], label=float(r['human_label']))
                 for r in csv.DictReader(f) if r['dimension'] == 'dynamics_degree' and r['split'] == 'test'
                 and r['video_a_uid'] in bases and r['video_b_uid'] in bases]
    assert human == rows(root / 'analysis/human_pairs.jsonl') and len(human) == 450
    ordered = [r for r in human if r['label'] != .5]; tied = [r for r in human if r['label'] == .5]
    assert len(ordered) == 171 and len(tied) == 279
    prompts = [r['prompt_id'] for r in ordered]; credits = {}
    for view, (family, seed) in zip(('base', '1701', '2904'), views):
        credits[view] = {}; reference = summary['human_preferences_by_view'][view]
        for arm in ('origin', 'old_joint', 'anchored'):
            m = np.array([(value(r['a'], family, seed, arm) - value(r['b'], family, seed, arm)) * (2 * r['label'] - 1) for r in ordered])
            credit = (m > 0).astype(float) + .5 * (m == 0); credits[view][arm] = credit
            e = reference[arm]
            assert e['ordered'] == 171 and e['human_ties'] == 279
            assert [(m > 0).sum(), (m == 0).sum(), (m < 0).sum()] == [e['strict_correct'], e['predicted_ties'], e['incorrect']]
            same(credit.mean(), e['concordance']); same(interval(credit, prompts), e['prompt_ci95'])
            gaps = [abs(value(r['a'], family, seed, arm) - value(r['b'], family, seed, arm)) for r in tied]
            same(np.mean(gaps), e['human_tie_mean_absolute_gap'])
        for arm in ('origin', 'old_joint'):
            diff = credits[view]['anchored'] - credits[view][arm]; e = reference['anchored_minus_' + arm]
            same(diff.mean(), e['difference']); same(interval(diff, prompts), e['prompt_ci95'])
            assert [(diff > 0).sum(), (diff == 0).sum(), (diff < 0).sum()] == [e['better_pairs'], e['unchanged_pairs'], e['worse_pairs']]
    for seed in ('1701', '2904'):
        for arm in ('origin', 'old_joint', 'anchored'):
            diff = credits[seed][arm] - credits['base'][arm]; e = summary['human_response_change'][seed][arm]
            same(diff.mean(), e['difference']); same(interval(diff, prompts), e['prompt_ci95'])
            assert [(diff > 0).sum(), (diff == 0).sum(), (diff < 0).sum()] == [e['better_pairs'], e['unchanged_pairs'], e['worse_pairs']]
    assert summary['coverage']['construction_status_counts'] == dict(Counter(r['status'] for r in inputs))
    assert summary['training_updates'] == 0 and not summary['mapping_changes'] and not summary['origin_recomputed']
    assert summary['previous_development_acceptance'] == 'failed_natural_ordering_18_of_23_required_19'
    assert not summary['formal_goal_complete'] and not summary['new_independent_holdout'] and not summary['calibration45_opened']
    return {'status': 'passed', 'inputs_verified': 1800, 'new_sigmoid_and_old_replay_scores_verified': 3600,
        'fresh_features_match_prior': 1800, 'old_head_max_latent_error': maximum_old_error,
        'origin_formula_recomputed_from_existing_raw_flows': 1800, 'encoding_controls_exact': 450,
        'human_pairs_verified': 450, 'human_views': 3, 'ordered_pairs': 171, 'human_ties': 279,
        'bootstrap_intervals_independently_verified': True, 'maximum_statistic_error': max(errors),
        'summary_sha256': sha(root / 'analysis/summary.json'), 'auditor_sha256': sha(__file__)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'previous-root', 'human-pairs', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    target = Path(args.output).resolve(); repo = Path(__file__).resolve().parents[2]
    if target == repo or any(target.is_relative_to(repo / n) for n in ('data', 'results', 'splits', 'runs')):
        raise ValueError('cannot write frozen trees')
    result = audit(args.root, args.previous_root, args.human_pairs)
    with target.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
