#!/usr/bin/env python3
"""Bind the completed matrix to local audit reports, weights and resource ledgers.

Uses explicitly named, non-secret artifacts in the documented repository layout.
No network access, model loading, annotation calls or changes to scores occur.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    hashes = {}

    def read(name):
        path = ROOT / name
        data = path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    score = read('output/deterministic/matrix-v1/report.json')
    tables = read('output/deterministic/matrix-v1/tables/manifest.json')
    controls = read('output/deterministic/matrix-v1/object-controls.json')
    visibility = read('data/gold/objects-visibility-v1/merged.report.json')
    if not all([score['execution_complete'], tables['execution_complete'],
                visibility['complete'], controls['outside_patch_and_equal_area_checks_passed']]):
        raise ValueError('final execution and control checks must be complete')
    matrix = read('data/deterministic/matrix-v1/manifest.json')
    checkpoints = read('output/audit-recovery/final-checkpoints.json')
    weights = read('output/audit-recovery/model-sha256.json')
    base_revision = read('output/audit-recovery/base-revision-verification.json')
    if not base_revision['passed']:
        raise ValueError('base weight revision not verified')
    training, inference, visual = {}, {}, {}
    for task, version in [('spatial', 'v8'), ('scene', 'v8'), ('action', 'v9'), ('objects', 'v6')]:
        name = task + '-' + version
        prefix = 'output/audit-recovery/training/' + name
        run = read(prefix + '/run_summary.json')
        manifest = read(prefix + '/run_manifest.json')
        check = checkpoints[task]
        if not (run['base_frozen'] and run['adapter_changed'] and check['final_equals_last_checkpoint']
                and check['final_step'] == run['max_steps']
                and check['train_sha256'] == manifest['data']['sha256']):
            raise ValueError('training identity failed: ' + task)
        prediction = 'action-v9' if task == 'action' else task
        plan = read('data/repair/matrix-v1/' + prediction + '.plan.json')
        inf = read('data/repair/matrix-v1/' + prediction + '.report.json')
        if not inf['complete']:
            raise ValueError('inference incomplete: ' + task)
        for key, value in plan['model_artifacts_sha256'].items():
            owner, filename = key.split('/', 1)
            files = weights['models']['base' if owner == 'base' else name]['files']
            if filename not in files or files[filename]['sha256'] != value:
                raise ValueError('inference/training artifact differs: ' + key)
        training[task] = {
            'version': version, 'train_samples': run['train_samples'], 'dev_samples': run['eval_samples'],
            'fixed_steps': run['max_steps'], 'train_metrics': run['train_metrics'],
            'peak_gpu_memory_mib': None, 'peak_memory_note': 'Not recorded during these training runs.',
            'base_frozen': run['base_frozen'], 'adapter_changed': run['adapter_changed'],
            'final_checkpoint_verification': check, 'git': manifest['git'],
            'environment': manifest['environment'], 'config': run['config'],
            'length_report': run['length_report'], 'parameter_inventory': {
                k: run['parameter_inventory'][k] for k in ['trainable_parameters', 'frozen_parameters']}}
        inference[task] = {**inf, 'artifact_sha256': plan['model_artifacts_sha256'],
                           'code_sha256': plan['inference_code_sha256']}
        cache = read('data/backend-cache/matrix-v1/' + task + '-test.manifest.json')
        visual[task] = {k: cache[k] for k in ['num_frames', 'upstream_sha', 'weights_sha256', 'split_sha256']}
        visual[task]['report'] = read('data/backend-cache/matrix-v1/' + task + '-test.report.json')
    api = {}
    for name, path in [
        ('scene-dev-evidence', 'data/gold/scene-evidence-v8/dev-evidence.report.json'),
        ('scene-test2-vision', 'output/audit-recovery/scene-test2-v8/gold-report.json'),
        ('scene-test2-evidence', 'output/audit-recovery/scene-test2-v8/evidence.report.json'),
        ('scene-synonym-evidence', 'output/audit-recovery/scene-matrix-syn-v1/evidence.report.json'),
        ('objects-visibility', 'data/gold/objects-visibility-v1/merged.report.json')]:
        report = read(path)
        api[name] = {'budget': report['budget'], 'usage': report['usage'],
                     'complete': report['complete'], 'human_gold': False}
    failed = read('output/audit-recovery/api-ledgers/BUDGET-scene-test2.json')
    source_audit = read('output/audit-recovery/training-near-sources.json')
    duplicate_audit = read('output/audit-recovery/test-video-duplicates.json')
    rules = read('output/upstream-parity/official-rules.json')
    umt = read('output/upstream-parity/umt-eight-videos.json')
    pixels = read('output/upstream-parity/mirror-pixels-v2.json')
    parity = {
        'official_rules': {k: rules[k] for k in ['passed', 'random_geometry_cases',
            'random_frame_cases', 'scene_cases', 'objects_cases', 'scope', 'upstream_commit']},
        'actual_umt': {**{k: umt[k] for k in ['passed', 'official_mean', 'selection',
            'upstream_commit', 'cache_sha256', 'weights_sha256', 'script_sha256']},
            'videos': len(umt['comparisons'])},
        'mirror_pixels': {'cases': len(pixels), 'passed': all(r['sampled_pixels_equal_exact_mirror'] for r in pixels),
            'axes': dict(Counter(r['axis'] for r in pixels)),
            'timing_policies': sorted({r['timing_policy'] for r in pixels})}}
    if not all(v['passed'] for v in parity.values()):
        raise ValueError('official parity failed')
    transforms = {task: read('data/backend-cache/matrix-v1/' + task + '-transforms.report.json')
                  for task in ['spatial', 'objects']}
    length_counts = {}
    path = ROOT / 'data/deterministic/matrix-v1/matrix.jsonl'
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    unique = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['transform'] == 'identity':
            unique[row['task'], row['original_prompt']] = len(row['original_prompt'].split())
    for task in training:
        counts = Counter(words for (dimension, _), words in unique.items() if dimension == task)
        length_counts[task] = {'unique_original_prompts': sum(counts.values()),
                               'word_count_histogram': dict(sorted(counts.items()))}
    api_totals = {
        'current_purposes_requests': sum(v['budget']['used_requests'] for v in api.values()),
        'successful_responses': sum(v['usage'].get('successful_requests', v['usage'].get('successful_responses', 0)) for v in api.values()),
        'prompt_tokens': sum(v['usage']['prompt_tokens'] for v in api.values()),
        'completion_tokens': sum(v['usage']['completion_tokens'] for v in api.values()),
        'currency_cost': None, 'historical_failed_test2_requests': failed['used_requests'],
        'note': 'Request reservations are not currency spending. Historical failed test2 is separate; older training annotation budgets are not included.'}
    result = {
        'execution_complete': True, 'matrix': matrix, 'training': training, 'inference': inference,
        'base_revision_verification': base_revision, 'model_file_sha256': weights,
        'visual_backends': visual, 'transform_coverage': transforms,
        'objects_visibility': visibility, 'api_by_purpose': api, 'api_totals': api_totals,
        'historical_failed_test2_budget': failed, 'source_audit': source_audit,
        'duplicate_video_audit': duplicate_audit, 'test_prompt_lengths': length_counts,
        'official_parity': parity, 'scorer_identity': {k: score[k] for k in [
            'scorer_git_commit', 'scorer_dirty_diff_sha256', 'scorer_source_sha256',
            'supporting_artifact_sha256', 'scoring_wall_seconds']},
        'scoring_input_sha256': score['input_sha256'], 'table_manifest': tables,
        'input_report_sha256': hashes, 'summarizer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reporting_git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'limitations': ['Model labels are not human gold.', 'Same-model multi-pass votes can share errors.',
                        'Training peak GPU memory and currency cost are unknown.',
                        'API aliases are not immutable provider model revisions.',
                        'Completed execution does not imply complete valid occlusion coverage.']}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': True, 'out': str(out), 'api_totals': api_totals}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
