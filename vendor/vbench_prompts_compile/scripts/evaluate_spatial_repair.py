#!/usr/bin/env python3
"""Re-score frozen Spatial evidence with signed Repair and audit four-cell trends.

All planned rows remain in the output. Conditional direction tests use original
official entity identities and box geometry, independently of Repair predictions.
These are detector-geometry contracts, not human-labelled visual correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile import annotation_jobs as J, records as R
from vbench_prompts_compile.experiments import RELATIONS, cluster_summary, text_key
from vbench_prompts_compile.official_replay import spatial_scores, UPSTREAM_SHA
from vbench_prompts_compile.spatial_repair import frame_score
from vbench_prompts_compile.visual_transforms import mirror_detections
from score_matrix import SCHEMES, backend_target, native_entity_codec, origin_compiled, paired_rows, parsed_target, summarize
from cache_matrix_transforms import evidence_id
from render_matrix_tables import table, ci_text, target_set

CELLS = ('identity', 'evidence_mirror', 'prompt_swap', 'evidence_mirror_joint')
TOLERANCE = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index_unique(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError('duplicate identity: ' + str(identity))
        result[identity] = row
    return result


def reference_geometry(target, frame):
    """Independent center/area calculation; never use a Repair prediction/score."""
    if frame is None:
        return {'status': 'missing_frame', 'forward': 0., 'reverse': 0.}
    a = [(i, v['box']) for i, v in enumerate(frame) if v['label'] == target['object_a']]
    b = [(i, v['box']) for i, v in enumerate(frame) if v['label'] == target['object_b']]
    if not a or not b:
        return {'status': 'missing_entity', 'forward': 0., 'reverse': 0.}
    if any(len(box) != 4 or not all(math.isfinite(v) for v in box)
           or box[2] <= box[0] or box[3] <= box[1] for _, box in a + b):
        return {'status': 'invalid_box', 'forward': 0., 'reverse': 0.}
    relation = {v: k for k, v in RELATIONS.items()}[target['relationship']]
    horizontal = relation in {'left', 'right'}
    forward, reverse = 0., 0.
    for i, first in a:
        for j, second in b:
            if i == j:
                continue
            centers = [((first[k] + first[k + 2]) / 2, (second[k] + second[k + 2]) / 2) for k in (0, 1)]
            axis, other = (0, 1) if horizontal else (1, 0)
            offset = centers[axis][1] - centers[axis][0]
            if abs(offset) <= abs(centers[other][1] - centers[other][0]):
                continue
            overlap = math.prod(max(0, min(first[k+2], second[k+2]) - max(first[k], second[k])) for k in (0, 1))
            union = math.prod(first[k+2] - first[k] for k in (0, 1)) + math.prod(second[k+2] - second[k] for k in (0, 1)) - overlap
            iou = overlap / union
            weight = 1. if iou < .1 else .1 / iou
            if (offset > 0) == (relation in {'left', 'above'}):
                forward = max(forward, weight)
            else:
                reverse = max(reverse, weight)
    if len(a) != 1 or len(b) != 1 or a[0][0] == b[0][0]:
        status = 'multiple_instances'
    elif forward:
        status = 'unique_supported'
    elif reverse:
        status = 'unique_contradicted'
    else:
        status = 'off_axis_or_tied'
    return {'status': status, 'forward': forward, 'reverse': reverse}


def trend_record(cells, expected, family, axis, condition, identity):
    values = dict(zip(CELLS, cells))
    down = condition.endswith('supported')
    return {'family': family, 'axis': axis, 'condition': condition, 'observation_id': identity,
            **values, 'mirror_delta': cells[1] - cells[0],
            'mirror_trend_pass': cells[1] < cells[0] - TOLERANCE if down else cells[1] > cells[0] + TOLERANCE,
            'mirror_response_at_margin': cells[1] - cells[0] <= -.03 if down else cells[1] - cells[0] >= .03,
            'prompt_trend_pass': cells[2] < cells[0] - TOLERANCE if down else cells[2] > cells[0] + TOLERANCE,
            'joint_exact_recovery': abs(cells[3] - cells[0]) <= TOLERANCE,
            'four_cell_expected_values': expected,
            'four_cell_contract_pass': all(abs(a - b) <= TOLERANCE for a, b in zip(cells, expected))}


def trend_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        for axis in ('all', row['axis']):
            groups[(row['scheme'], row['condition'], axis)].append(row)
    result = []
    for (scheme, condition, axis), values in sorted(groups.items()):
        result.append({'scheme': scheme, 'condition': condition, 'axis': axis,
                       **{key: cluster_summary(values, key) for key in (*CELLS, 'mirror_delta', 'mirror_trend_pass',
                           'mirror_response_at_margin', 'prompt_trend_pass', 'joint_exact_recovery', 'four_cell_contract_pass')}})
    return result


def audit_cells(matrix, caches, transforms, paired):
    scores = index_unique(paired, lambda r: (r['relative_path'], r['scheme'], r['transform']))
    frame_rows, video_rows, geometry_rows = [], [], []
    counts = Counter()
    mirrored_verified = 0
    for row in matrix:
        if row['transform'] != 'identity' or not row['eligible']:
            continue
        path = row['relative_path']
        base = caches[path]
        frames = base.get('frame_detections', [])
        variants = {name: scores[path, 'Origin', name] for name in CELLS}
        axis = variants['evidence_mirror']['axis']
        changed = transforms[evidence_id(variants['evidence_mirror'])]
        expected_mirror = mirror_detections(frames, base['frame_size'], axis)
        if changed.get('frame_detections') != expected_mirror:
            raise ValueError('cached evidence is not an exact coordinate mirror: ' + path)
        if changed.get('source_video_sha256') != base['video_sha256']:
            raise ValueError('mirror source identity mismatch: ' + path)
        mirrored_verified += 1
        geometry = []
        for i in range(16):
            g = reference_geometry(row['official_target'], frames[i] if i < len(frames) else None)
            geometry.append(g)
            counts[g['status']] += 1
            geometry_rows.append({'relative_path': path, 'frame_index': i, 'family': row['family'], **g})
            if g['status'] not in {'unique_supported', 'unique_contradicted'}:
                continue
            expected = [g['forward'], g['reverse'], g['reverse'], g['forward']]
            for scheme in SCHEMES:
                values = [scores[path, scheme, c]['transformed_frame_scores'][i] for c in CELLS]
                frame_rows.append({'scheme': scheme, **trend_record(values, expected, row['family'], axis, g['status'], [path, i])})
        # One-sided throughout the clip: missing targets contribute zero, never
        # count as recovered positives. Mixed directions are retained separately.
        before = sum(g['forward'] for g in geometry) / 16
        reverse = sum(g['reverse'] for g in geometry) / 16
        if any(g['status'] in {'missing_frame', 'invalid_box'} for g in geometry):
            condition = 'incomplete_evidence'
        elif before > 0 and reverse == 0:
            condition = 'one_sided_supported'
        elif reverse > 0 and before == 0:
            condition = 'one_sided_contradicted'
        else:
            condition = 'mixed_directions' if before and reverse else 'no_directional_evidence'
        counts['videos_' + condition] += 1
        if condition.startswith('one_sided'):
            for scheme in SCHEMES:
                values = [scores[path, scheme, c]['transformed_score'] for c in CELLS]
                video_rows.append({'scheme': scheme, **trend_record(values, [before, reverse, reverse, before], row['family'], axis, condition, path)})
    return {'geometry_status_counts': dict(counts), 'exact_mirrors_verified': mirrored_verified,
            'frame_summary': trend_summary(frame_rows), 'video_summary': trend_summary(video_rows)}, frame_rows, video_rows, geometry_rows


def ablations(matrix, caches, transforms, predictions, codec, video_rows):
    groups = defaultdict(list)
    conditions = {row['observation_id']: row['condition'] for row in video_rows}
    for row in matrix:
        if row['transform'] != 'evidence_mirror' or not row['eligible']:
            continue
        path = row['relative_path']
        original, mirrored = caches[path], transforms[evidence_id(row)]
        for scheme in ('Repair-rule', 'Repair-model'):
            parsed = parsed_target('spatial', row, scheme, predictions, None)
            for name, normalize, geometry in [('legacy', False, 'official'), ('articles_only', True, 'official'),
                                               ('ordered_unsigned', True, 'ordered'), ('signed_repair_v2', True, 'signed')]:
                target = backend_target('spatial', parsed, codec, normalize_spatial=normalize)
                cells = []
                for evidence in (original, mirrored):
                    total = 0.
                    for frame in evidence.get('frame_detections', [])[:16]:
                        if frame is None or not target or not target.get('relationships'):
                            continue
                        parts = []
                        for triple in target['relationships']:
                            if geometry == 'official':
                                mapped = {'object_a': triple['subject'], 'object_b': triple['object'], 'relationship': RELATIONS[triple['relation']]}
                                parts.append(spatial_scores(mapped, [frame])[0])
                            else:
                                parts.append(frame_score(triple, frame, signed=geometry == 'signed'))
                        total += sum(parts) / len(parts)
                    cells.append(total / 16)
                result = {'family': row['family'], 'original': cells[0], 'mirrored': cells[1], 'delta': cells[1] - cells[0]}
                for condition in ['all', conditions.get(path, 'mixed_or_no_direction')]:
                    groups[(scheme, name, condition)].append(result)
    return [{'scheme': scheme, 'variant': variant, 'condition': condition,
             **{key: cluster_summary(rows, key) for key in ('original', 'mirrored', 'delta')}}
            for (scheme, variant, condition), rows in sorted(groups.items())]


def parsing_audit(matrix, predictions, codec):
    unique = {(r['prompt'], r['transform']): r for r in matrix if r['eligible'] and r['transform'] in {'identity', 'prompt_swap'}}
    rows = []
    for (_, transform), row in unique.items():
        target = parsed_target('spatial', row, 'Repair-model', predictions, None)
        expected = target_set('spatial', origin_compiled('spatial', row['official_target']))
        legacy = backend_target('spatial', target, codec)
        current = backend_target('spatial', target, codec, normalize_spatial=True)
        rows.append({'transform': transform, 'prompt': row['prompt'], 'raw_target': target,
                     'legacy_backend_target': legacy, 'backend_target': current,
                     'legacy_exact': target_set('spatial', legacy) == expected,
                     'current_exact': target_set('spatial', current) == expected})
    summary = {transform: {'n': sum(r['transform'] == transform for r in rows),
                          **{key: sum(r['transform'] == transform and r[key] for r in rows) for key in ('legacy_exact', 'current_exact')}}
               for transform in ('identity', 'prompt_swap')}
    return summary, rows


def physical_mirror_summary(paired, video_rows):
    """Detector reruns on the SAME preselected one-sided original videos.

    Geometry-derived conditions do not guarantee that a new detector run keeps
    the same objects/boxes. These are observed responses, not exact contracts.
    """
    conditions = {row['observation_id']: row['condition'] for row in video_rows}
    joint = {(r['relative_path'], r['scheme']): r for r in paired if r['transform'] == 'video_mirror_joint'}
    groups = defaultdict(list)
    for row in paired:
        if row['transform'] != 'video_mirror' or row['relative_path'] not in conditions:
            continue
        condition = conditions[row['relative_path']]
        other = joint[row['relative_path'], row['scheme']]
        delta = row['paired_delta']
        values = {'family': row['family'], 'original': row['original_score'], 'mirrored': row['transformed_score'],
                  'joint': other['transformed_score'], 'delta': delta,
                  'direction_response_at_margin': row['pair_complete'] and
                      (delta <= -.03 if condition.endswith('supported') else delta >= .03),
                  'joint_recovery_within_margin': other['pair_complete'] and abs(other['paired_delta']) <= .03}
        groups[row['scheme'], condition].append(values)
    return [{'scheme': scheme, 'condition': condition,
             **{key: cluster_summary(rows, key) for key in ('original', 'mirrored', 'joint', 'delta',
                 'direction_response_at_margin', 'joint_recovery_within_margin')}}
            for (scheme, condition), rows in sorted(groups.items())]


def compare_history(current, legacy, previous):
    old = {}
    with Path(previous).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['task'] == 'spatial':
                key = row['sample_id'], row['scheme']
                if key in old:
                    raise ValueError('duplicate historical identity')
                old[key] = row
    if len(old) != len(current) or len(legacy) != len(current):
        raise ValueError('historical planned denominator changed')
    for rows in (legacy, [r for r in current if r['scheme'] == 'Origin']):
        for row in rows:
            previous_row = old[row['sample_id'], row['scheme']]
            if row != previous_row:
                keys = [key for key in set(row) | set(previous_row) if row.get(key) != previous_row.get(key)]
                raise ValueError('historical parity failed: ' + str((row['sample_id'], row['scheme'], keys)))
    return {'legacy_all_scheme_rows_exact': len(legacy), 'origin_rows_exact': sum(r['scheme'] == 'Origin' for r in current)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--cache', required=True, help='original Spatial GRiT JSONL')
    p.add_argument('--transforms', required=True, help='frozen evidence/video mirror JSONL')
    p.add_argument('--repair', required=True, help='fixed final Spatial adapter predictions')
    p.add_argument('--previous-paired', required=True, help='matrix-v1 paired rows; Origin must match exactly')
    p.add_argument('--out', required=True, help='new output directory, never the historical directory')
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.resolve() == Path(args.previous_paired).resolve().parent:
        raise ValueError('refusing to overwrite the historical experiment')
    matrix = [r for r in J.read_jsonl(Path(args.matrix)) if r['task'] == 'spatial']
    if not matrix:
        raise ValueError('matrix contains no Spatial observations')
    index_unique(matrix, lambda r: r['sample_id'])
    caches = index_unique(J.read_jsonl(Path(args.cache)), lambda r: r['relative_path'])
    transforms = index_unique(J.read_jsonl(Path(args.transforms)), lambda r: r['evidence_id'])
    prediction_rows = J.read_jsonl(Path(args.repair))
    predictions = {key: r['target'] for key, r in index_unique(prediction_rows, lambda r: r['request_id']).items()}
    for row in matrix:
        if row['relative_path'] not in caches:
            raise ValueError('missing planned original record')
        if row['evidence_kind'] != 'original' and evidence_id(row) not in transforms:
            raise ValueError('unattempted planned transform')
        if row['eligible'] and text_key('spatial', row['prompt']) not in predictions:
            raise ValueError('missing planned adapter prediction')
    codec = native_entity_codec(json.loads(Path(args.metadata).read_text()))
    scored = paired_rows(matrix, {'spatial': caches}, transforms, predictions, None, codec)
    legacy = paired_rows(matrix, {'spatial': caches}, transforms, predictions, None, codec, spatial_backend='legacy-official')
    history = compare_history(scored, legacy, args.previous_paired)
    audit, frame_rows, video_rows, geometry_rows = audit_cells(matrix, caches, transforms, scored)
    parsing, parsing_rows = parsing_audit(matrix, predictions, codec)
    paths = [args.matrix, args.metadata, args.cache, args.transforms, args.repair, args.previous_paired]
    sources = ['scripts/evaluate_spatial_repair.py', 'scripts/score_matrix.py', 'scripts/render_matrix_tables.py',
               'scripts/build_matrix_plan.py', 'scripts/cache_matrix_transforms.py',
               'src/vbench_prompts_compile/spatial_repair.py', 'src/vbench_prompts_compile/official_replay.py',
               'src/vbench_prompts_compile/experiments.py', 'src/vbench_prompts_compile/records.py',
               'src/vbench_prompts_compile/sources.py', 'src/vbench_prompts_compile/visual_transforms.py']
    report = {'spatial_backend': 'repair-v2', 'upstream_sha': UPSTREAM_SHA, 'execution_complete': True,
              'matrix_rows': len(matrix), 'paired_rows': len(scored), 'history_parity': history, 'parsing': parsing,
              'summary': summarize(scored), **audit,
              'physical_mirror_conditional': physical_mirror_summary(scored, video_rows),
              'ablations': ablations(matrix, caches, transforms, predictions, codec, video_rows),
              'input_sha256': {path: sha(path) for path in paths},
              'source_sha256': {path: sha(ROOT / path) for path in sources},
              'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'git_diff_sha256': hashlib.sha256(subprocess.check_output(['git', '-C', str(ROOT), 'diff', 'HEAD'])).hexdigest(),
              'notes': ['New backend intervention on previously inspected data; no new blind-test or adapter-training claim.',
                        'All 16 frame slots retained; missing evidence scores zero, coverage is separate.',
                        'Conditional subsets use original official identities and detector geometry, not human gold.',
                        'Multiple instances/temporal direction mixtures and unlocalized targets remain in full results.',
                        '2000 source-family bootstrap draws, seed 20260919; floating equality tolerance 1e-10.',
                        'Physical mirror scores use previously re-run GRiT; detector equivariance is not assumed.']}
    out.mkdir(parents=True, exist_ok=True)
    R.write_jsonl(out / 'paired-rows.jsonl', scored)
    R.write_jsonl(out / 'frame-trends.jsonl', frame_rows)
    R.write_jsonl(out / 'video-trends.jsonl', video_rows)
    R.write_jsonl(out / 'geometry-audit.jsonl', geometry_rows)
    R.write_jsonl(out / 'parsing.jsonl', parsing_rows)
    report['output_sha256'] = {path.name: sha(path) for path in sorted(out.glob('*.jsonl'))}
    J.atomic_json(out / 'report.json', report)
    summaries = [{'Scheme': r['scheme'], 'Transform': r['transform'], 'N/planned': f"{r['eligible']}/{r['planned']}",
                  'Original': r['original_score']['estimate'], 'Transformed': r['transformed_score']['estimate'],
                  'Delta (95% family CI)': ci_text(r['paired_delta']), 'Coverage': r['transformed_coverage']['estimate']}
                 for r in report['summary']]
    table(out, 'scores', summaries, list(summaries[0]))
    for name in ('frame', 'video'):
        summaries = [{'Scheme': r['scheme'], 'Condition': r['condition'], 'Axis': r['axis'],
                      'N': r['identity']['n'], 'Families': r['identity']['families'],
                      **{key: r[key]['estimate'] for key in CELLS},
                      'Mirror delta (CI)': ci_text(r['mirror_delta']), 'Expected trend (CI)': ci_text(r['mirror_trend_pass']),
                      'Response >=0.03 (CI)': ci_text(r['mirror_response_at_margin']),
                      'Four-cell pass (CI)': ci_text(r['four_cell_contract_pass'])} for r in report[name + '_summary']]
        if summaries:
            table(out, name + '-trends', summaries, list(summaries[0]))
    summaries = [{'Scheme': r['scheme'], 'Variant': r['variant'], 'Condition': r['condition'],
                  'N': r['original']['n'], 'Original': r['original']['estimate'], 'Mirrored': r['mirrored']['estimate'],
                  'Delta (CI)': ci_text(r['delta'])} for r in report['ablations']]
    table(out, 'ablations', summaries, list(summaries[0]))
    summaries = [{'Scheme': r['scheme'], 'Original condition': r['condition'], 'N': r['original']['n'],
                  'Original': r['original']['estimate'], 'Video mirror': r['mirrored']['estimate'],
                  'Video mirror and prompt swap': r['joint']['estimate'], 'Delta (CI)': ci_text(r['delta']),
                  'Response >=0.03 (CI)': ci_text(r['direction_response_at_margin']),
                  'Joint within 0.03 (CI)': ci_text(r['joint_recovery_within_margin'])}
                 for r in report['physical_mirror_conditional']]
    if summaries:
        table(out, 'physical-mirror', summaries, list(summaries[0]))
    print(json.dumps({key: report[key] for key in ('matrix_rows', 'paired_rows', 'history_parity', 'parsing', 'geometry_status_counts')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
