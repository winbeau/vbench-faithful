"""Independent receipt/statistics audit for the fixed Origin-scale training trial.

No model selection, TEST450 access, score transformation, or input mutation.
Four-prompt bootstrap intervals are descriptive development diagnostics only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .audit_vjepa_motion_probe import rows, sha


VIEWS = ('base', 'alternating', 'aperiodic', 'still', 'pan8', 'pan32', 'reversal32', 'still_jitter8')


def check_origin_record(record):
    assert record['status'] == 'ok'
    s, o = record['diagnostics']['sampling'], record['diagnostics']['official']
    assert s['source_fps'] == 8 and s['sampled_frame_count'] == 16
    assert s['sampled_source_frame_indices'] == list(range(16)) and s['sampling_interval'] == 1
    assert s['timestamps'] == [i / 8 for i in range(16)]
    threshold = 6 * min(s['frame_shape']) / 256
    assert threshold == o['official_threshold'] and o['official_count_num'] == 4
    raw = o['raw_flow_top5_mean']
    assert len(raw) == 15 and np.isfinite(raw).all()
    flags = [x > threshold for x in raw]
    assert flags == o['official_moving_flags'] and sum(flags) == o['official_moving_count']
    verdict = sum(flags) >= 4
    assert verdict == o['official_video_boolean'] and float(verdict) == record['score']
    return float(verdict)


def audit_origin(root, probe_root):
    root, probe_root = Path(root), Path(probe_root)
    cp = root / 'code/configs/dynamic-static-jitter/vjepa-aligned-v1.json'
    config = json.loads(cp.read_text())
    assert sha(probe_root / 'selection/sources.jsonl') == config['source_manifest_sha256']
    sources = [r for r in rows(probe_root / 'selection/sources.jsonl') if r['role'] in ('train', 'validation')]
    assert Counter(r['role'] for r in sources) == {'train': 210, 'validation': 60}
    reference = {r['video_uid']: r for r in sources}
    features = {r['video_uid']: r for p in (probe_root / 'features').glob('shard-*/features.jsonl')
                for r in rows(p) if r['view'] == 'base'}
    paths = sorted((root / 'origin-dev').glob('shard-*')); assert len(paths) == 4
    teacher = {}; receipts = []
    for i, path in enumerate(paths):
        c = json.loads((path / 'completion.json').read_text()); p = json.loads((path / 'provenance.json').read_text())
        assert c['status'] == 'finished' and c['failed'] == 0
        assert p['shard'] == i and p['shards'] == 4 and p['training_updates'] == 0
        assert p['config_sha256'] == sha(cp) and p['source_manifest_sha256'] == config['source_manifest_sha256']
        assert p['upstream']['sha'] == config['origin_upstream_commit'] and not p['upstream']['dirty']
        assert p['raft_sha256'] == config['raft_sha256']
        for file, checksum in p['code'].items():
            assert sha(root / 'code' / file) == checksum
        part = rows(path / 'scores.jsonl'); assert sha(path / 'scores.jsonl') == c['scores_sha256']
        assert len(part) == c['completed'] == c['expected'] == len(sources[i::4])
        assert [r['video_uid'] for r in part] == p['expected_uids'] == [r['video_uid'] for r in sources[i::4]]
        assert [r['reference_infer_parity'] for r in part] == [True] + [False] * (len(part) - 1)
        for record in part:
            check_origin_record(record)
            uid = record['video_uid']; assert uid in reference and uid not in teacher
            for key in ('role', 'prompt_id', 'generator', 'relative_video_path'):
                assert record[key] == reference[uid][key]
            assert record['video_sha256'] == features[uid]['video_sha256']
            teacher[uid] = record
        receipts.append(dict(shard=i, wall_seconds=c['wall_seconds'], started_utc=p['started_utc'],
                             completed_utc=c['completed_utc']))
    assert set(teacher) == set(reference)
    means = {role: float(np.mean([teacher[r['video_uid']]['score'] for r in sources if r['role'] == role]))
             for role in ('train', 'validation')}
    receipt = dict(status='passed', originals_verified=270, origin_means=means, upstream_infer_parity_cases=4,
                   teacher_use='only TRAIN210 population mean; DEV60 never used in loss', shards=receipts)
    return sources, teacher, receipt


def independent_gates(stats, parent, config, origin_mean):
    checks = {f'{v}_mean': stats['mean_by_view'][v] <= parent['validation_still_mean_max'] for v in ('still', 'still_jitter8')}
    checks.update({f'{v}_p95': stats['p95_by_view'][v] <= parent['validation_still_p95_max'] for v in ('still', 'still_jitter8')})
    for kind in ('weak_pan_above_still', 'strong_pan_above_weak', 'reversal_above_still'):
        checks[kind] = stats[kind + '_fraction'] >= parent[kind + '_fraction_min']
    c = stats['ordered_correct_base_alternating_aperiodic']
    checks.update(natural_ordering=c[0] >= parent['validation_native_ordered_correct_min'],
        jitter_ordering=min(c[1:]) >= c[0] - parent['max_cf_correct_loss_from_clean'],
        native_invariance=stats['native_jitter_mae'] <= parent['native_jitter_mae_max'],
        nonconstant=stats['native_base_std'] >= parent['native_score_std_min'],
        natural_scale_alignment=abs(stats['mean_by_view']['base'] - origin_mean) <= config['acceptance']['validation_native_mean_abs_error_vs_origin_max'])
    return checks


def descriptive_intervals(sources, scores, teacher):
    groups = sorted({r['prompt_id'] for r in sources})
    delta = scores[:, 1:3] - scores[:, :1]
    original = np.array([teacher[r['video_uid']]['score'] for r in sources])
    columns = np.column_stack((scores[:, 0], scores[:, 1:3].mean(1), delta.mean(1), abs(delta).mean(1), scores[:, 0] - original))
    blocks = np.array([columns[[r['prompt_id'] == p for r in sources]].mean(0) for p in groups])
    sample = np.random.default_rng(20260928).integers(0, len(groups), size=(20000, len(groups)))
    limits = np.quantile(blocks[sample].mean(1), [.025, .975], axis=0)
    return dict(method='20,000 prompt bootstrap, seed 20260928; descriptive only; not a selection gate',
                prompts=len(groups), intervals={name: limits[:, i].tolist() for i, name in enumerate(
                    ('base_mean', 'cf_mean', 'delta', 'mae', 'base_minus_origin_mean'))})


def audit(root, probe_root, anchored_root):
    root, probe_root, anchored_root = Path(root), Path(probe_root), Path(anchored_root)
    sources, teacher, teacher_audit = audit_origin(root, probe_root)
    config_path = root / 'code/configs/dynamic-static-jitter/vjepa-aligned-v1.json'
    config = json.loads(config_path.read_text())
    pc = root / 'code' / config['parent_config']; assert sha(pc) == config['parent_config_sha256']
    parent = json.loads(pc.read_text())
    training = root / 'training'; summary = json.loads((training / 'summary.json').read_text())
    provenance = json.loads((training / 'provenance.json').read_text())
    assert provenance['config_sha256'] == sha(config_path)
    for path, checksum in provenance['code'].items():
        assert sha(root / 'code' / path) == checksum
    assert sha(probe_root / 'selection/pairs.jsonl') == parent['probe_pairs_sha256']
    assert sha(anchored_root / 'training-chunked/summary.json') == config['parent_summary_sha256']
    prior = json.loads((anchored_root / 'training-chunked/summary.json').read_text())
    assert sha(anchored_root / 'training-chunked/scores.jsonl') == prior['scores_sha256']
    old = {r['video_uid']: r for r in rows(anchored_root / 'training-chunked/scores.jsonl') if r['arm'] == 'anchored'}
    assert sha(training / 'scores.jsonl') == summary['scores_sha256']
    assert sha(training / 'aligned.pt') == summary['head_sha256']
    assert provenance['origin_train_mean_target'] == summary['origin_train_mean_target'] == teacher_audit['origin_means']['train']
    assert provenance['teacher_training_sources'] == 210 and provenance['teacher_validation_sources_used_in_loss'] == 0
    assert summary['origin_means'] == teacher_audit['origin_means']
    curves = json.loads((training / 'training_curves.json').read_text())
    assert len(curves) == summary['optimizer_steps'] == config['optimization']['steps'] == 300
    assert [r['step'] for r in curves] == list(range(1, 301))
    for r in curves:
        c = parent['loss']; assert np.isfinite(list(r.values())).all()
        expected = (.5 * r['natural_ordered'] + .5 * r['human_ties'] + c['invariance_weight'] * r['native_jitter']
            + c['zero_weight'] * r['still_zero'] + c['pan_weight'] * (r['weak_pan'] + r['strong_pan'] + r['real_reversal']) / 3
            + c['native_floor_weight'] * r['native_not_below_still'] + config['loss']['origin_mean_weight'] * r['origin_train_mean'])
        assert abs(expected - r['total']) < 1e-6
    assert curves == rows(training / 'progress.jsonl')
    records = rows(training / 'scores.jsonl'); assert len(records) == 540
    index = {(r['role'], r['arm'], r['video_uid']): r for r in records}
    assert len(index) == 540 and set(index) == {(r['role'], a, r['video_uid']) for r in sources for a in ('previous_anchored', 'aligned')}
    maximum_error = 0.; prior_error = 0.; intervals = {}
    def same(a, b):
        nonlocal maximum_error
        error = float(np.abs(np.asarray(a) - np.asarray(b)).max()); assert error < 1e-12, (a, b)
        maximum_error = max(maximum_error, error)
    pairs = rows(probe_root / 'selection/pairs.jsonl')
    for role in ('train', 'validation'):
        cohort = [r for r in sources if r['role'] == role]
        lookup = {r['video_uid']: i for i, r in enumerate(cohort)}
        pp = [r for r in pairs if r['role'] == role]; ordered = [r for r in pp if r['label'] != .5]; tied = [r for r in pp if r['label'] == .5]
        for arm in ('previous_anchored', 'aligned'):
            selected = [index[role, arm, r['video_uid']] for r in cohort]
            assert all(tuple(r['views']) == VIEWS for r in selected)
            q = np.array([r['latent'] for r in selected]); score = np.array([r['scores'] for r in selected])
            assert np.isfinite(q).all() and np.isfinite(score).all(); same(1 / (1 + np.exp(-q)), score)
            if arm == 'previous_anchored':
                error = float(abs(q - np.array([old[r['video_uid']]['latent'] for r in cohort])).max())
                assert error <= 1e-6; prior_error = max(prior_error, error)
            reference = summary['evaluations'][role][arm]; change = score[:, 1:3] - score[:, :1]
            assert reference['sources'] == len(cohort) and reference['ordered_pairs'] == len(ordered) and reference['human_ties'] == len(tied)
            for j, view in enumerate(VIEWS):
                same(score[:, j].mean(), reference['mean_by_view'][view]); same(np.quantile(score[:, j], .95), reference['p95_by_view'][view])
            same(change.mean(), reference['native_jitter_delta']); same(abs(change).mean(), reference['native_jitter_mae'])
            same(change.min(), reference['native_largest_drop']); same(score[:, 0].std(), reference['native_base_std'])
            same([score[:, 0].min(), score[:, 0].max()], reference['native_base_range'])
            signed = np.array([(score[lookup[r['a']], :3] - score[lookup[r['b']], :3]) * (2 * r['label'] - 1) for r in ordered])
            assert (signed > 0).sum(0).tolist() == reference['ordered_correct_base_alternating_aperiodic']
            gaps = np.array([abs(score[lookup[r['a']], :3] - score[lookup[r['b']], :3]) for r in tied])
            same(gaps.mean(0), reference['human_tie_mean_gap_by_natural_view'])
            for name, high, low in [('weak_pan_above_still', 4, 3), ('strong_pan_above_weak', 5, 4), ('reversal_above_still', 6, 3)]:
                same(np.mean(score[:, high] > score[:, low]), reference[name + '_fraction'])
            for name, high, low in [('pan8_minus_still', 4, 3), ('pan32_minus_pan8', 5, 4), ('reversal32_minus_still', 6, 3)]:
                same((score[:, high] - score[:, low]).mean(), reference['mean_motion_margins'][name])
            same(reference['origin_native_mean'], teacher_audit['origin_means'][role])
            same(abs(score[:, 0].mean() - teacher_audit['origin_means'][role]), reference['native_mean_abs_error_vs_origin'])
            if role == 'validation':
                intervals[arm] = descriptive_intervals(cohort, score, teacher)
    checks = independent_gates(summary['evaluations']['validation']['aligned'], parent['acceptance'], config, teacher_audit['origin_means']['validation'])
    assert checks == summary['acceptance']['checks'] and all(checks.values()) == summary['acceptance']['all_passed']
    assert summary['initial_loading_max_abs_error'] <= 1e-6 and summary['saved_head_reload_max_abs_error'] <= 1e-6
    assert summary['trained_parameters'] == 51393 and summary['encoder_updates'] == 0
    for flag in ('posthoc_mapping_changed', 'individual_origin_labels_used', 'validation_teacher_used_in_loss', 'test450_used_for_training', 'calibration45_opened'):
        assert summary[flag] is False
    assert summary['test450_this_model'] == 'NOT RUN' and not summary['acceptance']['formal_goal_complete']
    controls = summary['user_requested_control_after_freeze']
    assert {r['case'] for r in controls} == {'still', 'pan8', 'pan32', 'local_jitter8', 'original'}
    old_controls = {r['case']: r for r in prior['user_requested_control_after_freeze']}
    for r in controls:
        same(r['aligned'], 1 / (1 + np.exp(-r['latent']))); same(r['previous_anchored'], old_controls[r['case']]['anchored'])
    import torch
    weights = torch.load(training / 'aligned.pt', map_location='cpu', weights_only=True)
    old_path = anchored_root / 'training-chunked/anchored.pt'; assert sha(old_path) == config['initial_head_sha256']
    initial = torch.load(old_path, map_location='cpu', weights_only=True)
    assert weights.keys() == initial.keys() and sum(w.numel() for w in weights.values()) == 51393
    changed = sum(not torch.equal(weights[k], initial[k]) for k in weights)
    assert changed > 0 and all(torch.isfinite(w).all() for w in weights.values())
    return dict(integrity_checks='passed', teacher_audit=teacher_audit, score_values_verified=4320, optimizer_steps_verified=300,
        maximum_statistic_error=maximum_error, previous_head_reproduction_max_abs_error=prior_error,
        checkpoint_tensors_changed=changed, checkpoint_tensors_total=len(weights), validation_descriptive_uncertainty=intervals,
        model_acceptance_passed=all(checks.values()), acceptance_independently_recomputed=checks,
        test450_this_model='NOT RUN', summary_sha256=sha(training / 'summary.json'), auditor_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'probe-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--anchored-root'); parser.add_argument('--origin-only', action='store_true')
    args = parser.parse_args()
    result = audit_origin(args.root, args.probe_root)[2] if args.origin_only else audit(args.root, args.probe_root, args.anchored_root)
    target = Path(args.output).resolve(); repo = Path(__file__).resolve().parents[2]
    if target == repo or any(target.is_relative_to(repo / name) for name in ('data', 'results', 'splits', 'runs')):
        raise ValueError('cannot write frozen trees')
    with target.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
