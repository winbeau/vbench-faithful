"""Same-input stability, complete denominators, and non-constant response checks."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from .analyze_subject_background_auto import cluster_summary
from .common import sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json


def validate_integrity_receipt(receipt, run, protocol):
    if (receipt['index_sha256'] != run['dataset_index_sha256']
            or receipt['index_sha256'] != protocol['dataset_index_sha256']
            or receipt['outside_mask_changed_pixels'] != 0
            or receipt['bases'] != protocol['cohort_candidates']
            or receipt['verified_corrupted_frames'] <= 0):
        raise ValueError('full pixel replay does not cover frozen cohort')


def review_qualified_records(records, exclusions):
    accepted={r['base']['video_uid'] for r in records if r['construction_status']=='accepted'}
    if not set(exclusions)<=accepted or any(not isinstance(reason,str) or not reason for reason in exclusions.values()):
        raise ValueError('input review exclusions must identify numerically constructed inputs and reasons')
    return [{**r,'numeric_construction_status':r['construction_status'],'construction_status':'input_review_rejected',
             'status':'input_review_rejected','review_rejection_reason':exclusions[r['base']['video_uid']]}
            if r['base']['video_uid'] in exclusions else r for r in records]


def value(row, name, method):
    item = row.get('variants', {}).get(name, {}).get('scores', {}).get(method, {})
    number = item.get('score')
    if item.get('status') != 'succeeded' or number is None:
        return None
    if not math.isfinite(number) or not -1e-6 <= number <= 1.000001:
        raise ValueError('invalid succeeded score')
    return float(number)


def summarize(records, methods, *, resamples=10000, positions=('start','middle','end','full')):
    ids = [r['base']['video_uid'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate video')
    accepted = [r for r in records if r['construction_status'] == 'accepted']
    groups = {r['base']['video_uid']: r['base']['prompt_id'] for r in accepted}
    report = {'total_candidates': len(records), 'constructed_denominator': len(accepted),
              'status_counts': dict(Counter(r['status'] for r in records)), 'positions': {},
              'bootstrap': {'unit': 'source_prompt', 'resamples': resamples, 'seed': 20260920},
              'human_preferences': 'auxiliary_only_not_measured_in_this_run'}
    flat = []
    if not positions or len(set(positions)) != len(positions):
        raise ValueError('positions must be nonempty and unique')
    for position in positions:
        report['positions'][position] = {}
        for method in methods:
            cases = []
            for row in accepted:
                uid = row['base']['video_uid']
                a,b,c = (value(row,k,method) for k in ('clean',position+'/background_corrupt',position+'/subject_corrupt'))
                ao,bo = (value(row,k,'official') for k in ('clean',position+'/background_corrupt'))
                delta = abs(a-b) if a is not None and b is not None else None
                origin = abs(ao-bo) if ao is not None and bo is not None else None
                subject = a-c if a is not None and c is not None else None
                constant_zero = a == b == 0
                joint = bool(None not in (delta, origin) and not constant_zero and origin >= .1 and delta <= .01)
                case = {'video_uid': uid, 'prompt_id': groups[uid], 'method': method, 'position': position,
                        'clean': a, 'background': b, 'subject': c, 'origin_clean': ao, 'origin_background': bo,
                        'origin_abs_change': origin, 'repair_abs_change': delta, 'subject_signed_drop': subject,
                        'zero_to_zero': constant_zero, 'joint_success': joint, 'runtime_status': row['status']}
                cases.append(case); flat.append(case)
            paired = [r for r in cases if None not in (r['origin_abs_change'], r['repair_abs_change'])]
            def stats(key, rows=paired):
                return cluster_summary({r['video_uid']: r[key] for r in rows if r[key] is not None},
                                       groups, resamples=resamples)
            origin, repair = stats('origin_abs_change'), stats('repair_abs_change')
            subject_cases = [r for r in cases if r['subject_signed_drop'] is not None]
            response = stats('subject_signed_drop', subject_cases)
            complete = len(paired) == len(accepted) and bool(accepted)
            successful = sum(r['joint_success'] for r in cases)
            report['positions'][position][method] = {
                'paired_scored': len(paired), 'unscored': len(accepted)-len(paired), 'full_coverage': complete,
                'origin': origin, 'repair': repair, 'subject_response': response,
                'subject_scored': len(subject_cases),
                'subject_positive': sum(r['subject_signed_drop'] > 0 for r in subject_cases),
                'subject_larger_than_background': sum(r['subject_signed_drop'] > r['repair_abs_change']
                    for r in subject_cases if r['repair_abs_change'] is not None),
                'zero_to_zero': sum(r['zero_to_zero'] for r in cases),
                'origin_ge_01': sum(r['origin_abs_change'] >= .1 for r in paired),
                'origin_ge_02': sum(r['origin_abs_change'] >= .2 for r in paired),
                'joint_success_count': successful,
                'joint_success_fraction_full_constructed_denominator': successful/len(accepted) if accepted else None,
                'mean_target_on_complete_cohort': bool(complete and origin['mean'] >= .1 and repair['mean'] <= .01
                                                    and not any(r['zero_to_zero'] for r in cases)),
                'mean_target_on_scored_subset': bool(paired and origin['mean'] >= .1 and repair['mean'] <= .01),
            }
    return report, flat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('configs/subject-repair/stability_v3_protocol.json'))
    parser.add_argument('--integrity-receipt',type=Path)
    args = parser.parse_args()
    run = json.loads((args.run/'run.json').read_text())
    if not run.get('completed') or sha256_file(args.run/'scores.jsonl') != run['scores_sha256']:
        raise ValueError('run incomplete or scores changed')
    protocol = json.loads((args.run/'protocol.json').read_text())
    integrity = None
    if protocol.get('defer_pixel_replay_to_analysis'):
        if not args.integrity_receipt:
            raise ValueError('final analysis requires the independent full pixel replay receipt')
        integrity = json.loads(args.integrity_receipt.read_text())
        validate_integrity_receipt(integrity,run,protocol)
    if sha256_file(args.run/'protocol.json') != run['protocol_sha256']:
        # write_json canonicalizes the copy; compare the original protocol by
        # its object too, retaining both hashes in the provenance below.
        local = args.protocol
        if not local.is_file() or sha256_file(local) != run['protocol_sha256'] or json.loads(local.read_text()) != protocol:
            raise ValueError('protocol copy differs from frozen original')
    records = read_jsonl(args.run/'scores.jsonl')
    if len(records) != protocol['cohort_candidates']:
        raise ValueError('incomplete candidate population')
    positions = protocol.get('positions', ['start','middle','end','full'])
    primary_position = protocol.get('primary_intervention','start/background_corrupt').split('/')[0]
    if primary_position not in positions:
        raise ValueError('primary condition is absent from the declared positions')
    expected_variants = {'clean'} | {p+'/'+s for p in positions
                                   for s in ('background_corrupt','subject_corrupt')}
    candidate = protocol.get('candidate_name', 'tracked')
    methods = ['official','direct_zero','direct_exclude',candidate+'_zero',candidate+'_exclude']
    for row in records:
        if row['status'] == 'completed' and (set(row['variants']) != expected_variants or
            any(set(v['scores']) != set(methods) for v in row['variants'].values())):
            raise ValueError('completed row has incomplete score grid')
    report, flat = summarize(records, methods[1:], positions=positions)
    if protocol.get('input_review_exclusions'):
        qualified=review_qualified_records(records,protocol['input_review_exclusions'])
        reviewed_report,_=summarize(qualified,methods[1:], positions=positions)
        if reviewed_report['constructed_denominator'] != protocol['expected_review_qualified']:
            raise ValueError('pre-score review population changed')
        report['review_qualified']=reviewed_report
        report['input_review_exclusions']=protocol['input_review_exclusions']
        for case in flat:
            case['review_eligible']=case['video_uid'] not in protocol['input_review_exclusions']
    report['provenance'] = {'scores_sha256': run['scores_sha256'], 'protocol_sha256': run['protocol_sha256'],
                            'run_sha256': sha256_file(args.run/'run.json'), 'analysis_sha256': sha256_file(Path(__file__))}
    report['maximum_origin_reference_error'] = run['maximum_origin_reference_error']
    report['wall_seconds'] = run['wall_seconds']
    if integrity is not None:
        report['integrity'] = integrity
        report['provenance']['integrity_receipt_sha256'] = sha256_file(args.integrity_receipt)
    report['localization'] = {}
    for name in sorted(expected_variants):
        diags = [r['variants'][name]['localizer_diagnostics'] for r in records if name in r['variants']]
        report['localization'][name] = {'variants': len(diags), 'frames': sum(d['num_frames'] for d in diags),
            'direct_empty_frames': sum(d['direct_empty_frames'] for d in diags),
            'tracked_empty_frames': sum(d['tracked_empty_frames'] for d in diags),
            'gap_prompt_count': sum(d['gap_prompt_count'] for d in diags)}
    out = new_output(args.output); write_json(out/'statistics.json', report)
    with (out/'per_case.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    print(json.dumps({'status_counts':report['status_counts'],'primary_position':primary_position,
                      primary_position:report['positions'][primary_position],
                      'origin_parity':report['maximum_origin_reference_error']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
