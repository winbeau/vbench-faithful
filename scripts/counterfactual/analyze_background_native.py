"""Analyze the full native background cohort, including source/donor dependence."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from background_consistency.algorithms import METHODS
from .analyze_background_development import checked_rows
from .analyze_subject_background_auto import cluster_summary
from .analyze_subject_natural import score
from .common import ROOT, sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, write_json


def dependency_components(source_donor):
    """Keep any observations sharing either prompt in the same component.

    The development donor cycle does not produce independent source clusters.
    Components are conservative independent resampling units; with fewer than
    two, uncertainty is unestimable and is reported as null, never fabricated.
    """
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    for source, donor in source_donor.values():
        a, b = find(source), find(donor)
        parent[max(a, b)] = min(a, b)
    return {uid: find(source) for uid, (source, _) in source_donor.items()}


def native_cases(rows, manifests, methods=METHODS):
    cases = []
    for row in rows:
        if row['construction_status'] != 'accepted':
            continue
        uid = row['video_uid']
        for position in ('full', 'start', 'middle', 'end'):
            variants = row['variants']
            origin_clean = score(variants.get('clean', {}), 'official')
            origin_edit = score(variants.get(position+'/subject_blur', {}), 'official')
            for method in methods:
                clean = score(variants.get('clean', {}), method)
                edited = score(variants.get(position+'/subject_blur', {}), method)
                background = score(variants.get(position+'/background_switch', {}), method)
                signed = edited-clean if None not in (clean, edited) else None
                absolute = abs(signed) if signed is not None else None
                drop = clean-background if None not in (clean, background) else None
                improvement = (abs(origin_edit-origin_clean)-absolute
                               if None not in (origin_clean, origin_edit, absolute) else None)
                cases.append(dict(base_id=uid, prompt_id=row['base']['prompt_id'],
                    donor_prompt_id=manifests[uid]['donor']['prompt_id'],
                    generator=row['base']['generator'], method=method, position=position,
                    clean=clean, subject_blur=edited, background_switch=background,
                    foreground_signed_change=signed, foreground_absolute_change=absolute,
                    paired_improvement_vs_official=improvement, background_temporal_drop=drop,
                    background_drop_gt_foreground_abs=(float(drop > absolute)
                        if None not in (drop, absolute) else None)))
    return cases


def normalized_response(cases, method, *, resamples=10000, seed=20260920):
    """Contrast nuisance with sensitivity so output rescaling cannot repair it.

    Use full subject blur and the mean of all three scene-switch drops on
    exactly the same bases. Resample shared source/donor components jointly.
    Nonpositive background response makes the ratio undefined, not successful.
    """
    lookup = {(r['base_id'], r['method'], r['position']): r for r in cases}
    full = [r for r in cases if r['method'] == method and r['position'] == 'full']
    values, sources = {}, {}
    for row in full:
        uid = row['base_id']
        array = []
        for name in ('official', method):
            nuisance = lookup[(uid, name, 'full')]['foreground_absolute_change']
            background = [lookup[(uid, name, p)]['background_temporal_drop'] for p in ('start', 'middle', 'end')]
            if nuisance is None or None in background:
                break
            array.extend([nuisance, float(np.mean(background))])
        if len(array) == 4:
            values[uid] = array
            sources[uid] = (row['prompt_id'], row['donor_prompt_id'])
    result = {'construction_accepted': len(full), 'common_scored_bases': len(values),
              'ratio_definition': 'mean full foreground absolute change / mean start-middle-end background drop',
              'official_ratio': None, 'candidate_ratio': None, 'paired_ratio_improvement': None,
              'background_response_retention': None, 'paired_ratio_improvement_ci95': None,
              'background_response_retention_ci95': None, 'positive_denominator_bootstrap_fraction': None}
    if not values:
        return result
    components = dependency_components(sources)
    clusters = sorted(set(components.values()))
    sums = np.asarray([np.sum([value for uid, value in values.items() if components[uid] == cluster], axis=0)
                       for cluster in clusters])
    total = sums.sum(0)
    result['n_dependency_components'] = len(clusters)
    if total[1] <= 0 or total[3] <= 0:
        return result
    result.update(official_ratio=float(total[0]/total[1]), candidate_ratio=float(total[2]/total[3]),
                  paired_ratio_improvement=float(total[0]/total[1]-total[2]/total[3]),
                  background_response_retention=float(total[3]/total[1]))
    if len(clusters) > 1:
        draw = np.random.default_rng(seed).integers(0, len(clusters), (resamples, len(clusters)))
        boot = sums[draw].sum(1)
        positive = (boot[:, 1] > 0) & (boot[:, 3] > 0)
        result['positive_denominator_bootstrap_fraction'] = float(positive.mean())
        # Avoid reporting a conditional interval after silently dropping bad
        # resamples. Such candidates have not established a usable response.
        if positive.all():
            result['paired_ratio_improvement_ci95'] = np.quantile(boot[:, 0]/boot[:, 1]-boot[:, 2]/boot[:, 3], [.025, .975]).tolist()
            result['background_response_retention_ci95'] = np.quantile(boot[:, 3]/boot[:, 1], [.025, .975]).tolist()
    return result


def summarize_native(rows, manifests, *, methods=METHODS, resamples=10000):
    cases = native_cases(rows, manifests, methods)
    components = dependency_components({r['video_uid']: (r['base']['prompt_id'],
        manifests[r['video_uid']]['donor']['prompt_id']) for r in rows
        if r['construction_status'] == 'accepted'})
    report = dict(candidate_count=len(rows),
        status_counts=dict(Counter(r['status'] for r in rows)),
        construction_status_counts=dict(Counter(r['construction_status'] for r in rows)),
        rejection_reason_counts=dict(Counter(x for r in rows for x in r.get('rejection_reasons', []))),
        scored_version_count=sum(len(r.get('variants', {})) for r in rows),
        background_dependency_components=dict(Counter(components.values())),
        uncertainty={'foreground': 'source prompt cluster bootstrap',
            'background': 'connected components of source/donor prompt graph; retain all shared prompts together',
            'resamples': resamples, 'seed': 20260920,
            'limitation': 'Few independent components give imprecise development intervals. No held-out claim.'},
        by_position={})
    for position in ('full', 'start', 'middle', 'end'):
        report['by_position'][position] = {}
        for method in methods:
            selected = [r for r in cases if r['position'] == position and r['method'] == method]
            groups = {r['base_id']: r['prompt_id'] for r in selected}
            summary = {'construction_accepted': len(selected)}
            for key in ('clean', 'subject_blur', 'background_switch', 'foreground_signed_change',
                        'foreground_absolute_change', 'paired_improvement_vs_official',
                        'background_temporal_drop', 'background_drop_gt_foreground_abs'):
                values = {r['base_id']: r[key] for r in selected if r[key] is not None}
                if key in ('foreground_absolute_change', 'paired_improvement_vs_official',
                           'background_temporal_drop', 'background_drop_gt_foreground_abs'):
                    background_stat = key.startswith('background_')
                    summary[key] = cluster_summary(values, components if background_stat else groups,
                                                   resamples=resamples)
                    summary[key]['cluster_unit'] = 'source_donor_component' if background_stat else 'source_prompt'
                else:
                    array = list(values.values())
                    summary[key] = {'n_bases': len(array), 'mean': float(np.mean(array)) if array else None}
            report['by_position'][position][method] = summary
    report['scale_adjusted_response'] = {method: normalized_response(cases, method, resamples=resamples) for method in methods}
    return report, cases


def load_native(run_root, dataset, *, methods=METHODS):
    construction = json.loads((dataset/'run.json').read_text())
    index_path = dataset/'index.jsonl'
    if not construction.get('completed') or sha256_file(index_path) != construction['index_sha256']:
        raise ValueError('incomplete or modified construction')
    index = read_jsonl(index_path)
    expected = {r['video_uid']: r for r in read_jsonl(ROOT/'configs/background-repair/natural1720_manifest_v2.jsonl')
                if r['split'] == 'dev'}
    if len(index) != len(expected) or {r['video_uid'] for r in index} != set(expected):
        raise ValueError('incomplete or duplicate development construction population')
    manifests = {}
    for entry in index:
        path = artifact_path(dataset, entry['manifest'])
        if sha256_file(path) != entry['manifest_sha256']:
            raise ValueError('construction manifest hash mismatch')
        manifest = json.loads(path.read_text())
        uid = entry['video_uid']
        if manifest['base'] != expected[uid] or manifest['status'] != entry['status']:
            raise ValueError('construction identity mismatch')
        manifests[uid] = manifest
    rows, runs = [], []
    for shard in sorted(run_root.glob('shard*')):
        if not shard.is_dir():
            continue
        chunk, run = checked_rows(shard)
        if run['dataset_index_sha256'] != construction['index_sha256']:
            raise ValueError('score dataset mismatch')
        rows.extend(chunk); runs.append(run)
    if len(rows) != len(expected) or {r['video_uid'] for r in rows} != set(expected):
        raise ValueError('incomplete, duplicated or unexpected scored population')
    variants = {'clean', 'full/subject_blur'} | {
        p+'/'+k for p in ('start', 'middle', 'end') for k in ('subject_blur', 'background_switch')}
    for row in rows:
        uid = row['video_uid']
        if row['base'] != expected[uid] or row['construction_status'] != manifests[uid]['status']:
            raise ValueError('score identity mismatch')
        if row['status'] == 'completed' and (set(row['variants']) != variants
                or any(set(v['scores']) != set(methods) for v in row['variants'].values())):
            raise ValueError('incomplete score grid marked completed')
    return rows, manifests, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, manifests, runs = load_native(args.run, args.dataset)
    report, cases = summarize_native(rows, manifests)
    report['source_runs'] = runs
    report['dataset_index_sha256'] = sha256_file(args.dataset/'index.jsonl')
    output = new_output(args.output)
    write_json(output/'statistics.json', report)
    with (output/'native_per_case.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader(); writer.writerows(cases)
    print(json.dumps({'output': str(output), 'statuses': report['status_counts'],
                      'dependency_components': report['background_dependency_components']}))


if __name__ == '__main__':
    main()
