#!/usr/bin/env python3
"""Three text interfaces over identical frozen evidence, paired by source video.

Origin preserves official unsigned geometry and same-label box pooling. Spatial
Repair defaults to ordered signed geometry and article-safe entity serialization;
--spatial-backend legacy-official reproduces the earlier interface-only experiment.
Missing scores are zero in the complete planned denominator. Declared eligibility
is determined before predictions; full-plan and eligible-domain summaries coexist.
No unverified visual endpoint is counted as a validated sensitivity success.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile import records as R
from vbench_prompts_compile.experiments import RELATIONS, cluster_summary, metadata_index, scene_rule_key, scene_rule_caption, text_key
from vbench_prompts_compile.official_replay import action_score, scene_scores, spatial_scores, UPSTREAM_SHA
from vbench_prompts_compile.sources import K400Vocabulary, load_k400
from vbench_prompts_compile import spatial_repair
from vbench_prompts_compile import objects_repair
from vbench_prompts_compile.action_repair import compile_action, LEXICON_SHA256
from build_matrix_plan import ACTION_SYNONYMS
from cache_matrix_transforms import evidence_id

SCHEMES = ('Origin', 'Repair-rule', 'Repair-model')


def native_entity_codec(metadata):
    """Invert the declared text canonicalizer over the full public label inventory.

    This is a text-side serialization adapter, independent of the test targets,
    predictions, detections and labels. Ambiguous inverses are never guessed.
    """
    index = metadata_index(metadata)
    inventory = set()
    for value in index['spatial'].values():
        if isinstance(value, dict):
            inventory.update([value['object_a'], value['object_b']])
    for value in index['objects'].values():
        if isinstance(value, str):
            inventory.update(name.strip() for name in value.split(' and '))
    inverse = defaultdict(set)
    for name in inventory:
        inverse[R.canonical_entity_name(name)].add(name)
    aliases = {name: next(iter(values)) for name, values in inverse.items() if len(values)==1}
    # A literal native label is already serialized correctly.
    aliases.update({name: name for name in inventory})
    return {'aliases': aliases, 'inventory': sorted(inventory),
            'ambiguous': {k: sorted(v) for k,v in inverse.items() if len(v)>1}}


def backend_target(task, target, codec, *, normalize_spatial=False):
    if not isinstance(target, dict) or task not in {'objects','spatial'}:
        return target
    aliases = (codec or {}).get('aliases', {})
    if task == 'objects':
        return {**target, 'entities': [aliases.get(v,v) for v in target.get('entities',[])]}
    name = (lambda value: spatial_repair.entity_name(value, aliases)) if normalize_spatial else (lambda value: aliases.get(value, value))
    return {**target, 'relationships': [
        {**triple, 'subject': name(triple['subject']),
         'object': name(triple['object'])}
        for triple in target.get('relationships',[])]}


def rule_target(task, prompt, vocab):
    if task == 'spatial':
        for relation, phrase in RELATIONS.items():
            if phrase in prompt.lower():
                a, b = prompt.lower().split(phrase, 1)
                return {'relationships': [{'subject': R.canonical_entity_name(a.strip()), 'relation': relation,
                    'object': R.canonical_entity_name(b.split(',')[0].strip().rstrip('.'))}]}
        return {'relationships': []}
    if task == 'objects':
        entities = [R.canonical_entity_name(p.strip()) for p in prompt.lower().split(',')[0].rstrip('.').split(' and ')]
        return {'entities': entities} if len(entities) >= 2 and all(entities) else {'entities': []}
    if task == 'action':
        phrase = re.sub(r'^a person is\s+', '', prompt.strip(), flags=re.I).rstrip('.')
        aliases = {v: vocab.resolve(k) for k, v in ACTION_SYNONYMS.items()}
        # Resolve a whole named class before interpreting an explicit conjunction.
        full = aliases.get(phrase.lower()) or vocab.resolve(phrase)
        if full:
            return {'actions': [full]}
        actions = [aliases.get(p.lower()) or vocab.resolve(p) or 'other' for p in phrase.split(' and ')]
        return {'actions': list(dict.fromkeys(actions))}
    raise ValueError(task)


def origin_compiled(task, target):
    if target is None:
        return None
    if task == 'spatial':
        reverse = {v:k for k,v in RELATIONS.items()}
        return {'relationships': [{'subject':target['object_a'], 'relation':reverse[target['relationship']], 'object':target['object_b']}]}
    return {('actions' if task == 'action' else 'entities'): [target] if task == 'action' else target.split(' and ')}


def parsed_target(task, row, scheme, predictions, vocab):
    if scheme == 'Origin':
        return origin_compiled(task, row['official_target'])
    if scheme == 'Repair-rule':
        return rule_target(task, row['prompt'], vocab)
    return predictions.get(text_key(task, row['prompt']))


def score_one(row, scheme, evidence, predictions, vocab, codec=None, *, spatial_backend='repair-v2', action_interface='repair-v2.1', objects_backend='repair-v2'):
    if spatial_backend not in {'repair-v2', 'legacy-official'}:
        raise ValueError('unknown spatial backend: ' + str(spatial_backend))
    if action_interface not in {'repair-v2.1', 'repair-v2', 'legacy-model'}:
        raise ValueError('unknown Action interface: ' + str(action_interface))
    if objects_backend not in {'repair-v2', 'legacy-official'}:
        raise ValueError('unknown Objects backend: ' + str(objects_backend))
    task = row['task']
    frames = 1 if task == 'action' else 16
    result = {'score':0., 'coverage':0., 'abstention':0., 'missing':1., 'target':None,
              'status':'missing_evidence', 'frame_scores':[0.] * frames}
    if not row['eligible']:
        return {**result, 'status':'ineligible'}
    good = evidence.get('status') in {'ok','partial'}
    if task == 'scene':
        captions = evidence.get('frame_captions', []) if good else []
        labels, scores, covered, abstained = [], [], 0, 0
        for i in range(16):
            caption = captions[i] if i < len(captions) else None
            label = None
            if isinstance(caption, str):
                if scheme == 'Repair-model':
                    label = predictions.get(text_key('scene', row['prompt'], caption))
                else:
                    key = row['official_target']
                    if scheme == 'Repair-rule':
                        key, caption = scene_rule_key(key), scene_rule_caption(caption)
                    label = 'supported' if scene_scores(key, [caption])[0] else 'contradicted'
            labels.append(label)
            covered += label in R.SCENE_LABELS
            abstained += label == 'insufficient'
            scores.append(float(label == 'supported'))
        return {**result, 'score':sum(scores)/16, 'coverage':covered/16, 'abstention':abstained/16,
            'missing':1-covered/16, 'target':labels, 'status':'ok' if covered==16 else 'partial', 'frame_scores':scores}
    target = parsed_target(task, row, scheme, predictions, vocab)
    if task == 'action' and scheme == 'Repair-model' and action_interface in {'repair-v2.1', 'repair-v2'}:
        resolution = compile_action(row['prompt'], target, vocab, scope_guard=action_interface == 'repair-v2.1')
        result['action_resolution'] = resolution
        target = resolution['target']
    result['target'] = target
    if not isinstance(target, dict):
        result['status'] = 'missing_parse'; return result
    key = {'spatial':'relationships','objects':'entities','action':'actions'}[task]
    values = target.get(key)
    if not isinstance(values, list) or not values:
        result['status'] = 'empty_parse'; return result
    if scheme != 'Origin':
        target = backend_target(task, target, codec, normalize_spatial=spatial_backend=='repair-v2')
        values = target[key]
    result['backend_target'] = target
    if not good:
        return result
    if task == 'objects' and scheme != 'Origin' and objects_backend == 'repair-v2':
        try:
            return {**result, **objects_repair.video_scores(values, evidence.get('frame_detections', []))}
        except (ValueError, TypeError) as error:
            return {**result, 'status': 'invalid_objects_evidence', 'errors': [str(error)]}
    if task == 'action':
        if not evidence.get('top5'):
            return result
        scores = [float(action_score(label, evidence['top5'])) if label != 'other' else 0. for label in values]
        return {**result, 'score':sum(scores)/len(scores), 'coverage':1., 'missing':0.,
            'abstention':values.count('other')/len(values), 'status':'ok', 'frame_scores':[sum(scores)/len(scores)],
            'known_class_score':sum(s for l,s in zip(values,scores) if l!='other')/max(1,sum(l!='other' for l in values))}
    evidence_frames = evidence.get('frame_detections' if task == 'spatial' else 'frame_labels', [])
    scores, covered, errors = [], 0, []
    for i in range(16):
        frame = evidence_frames[i] if i < len(evidence_frames) else None
        if frame is None:
            scores.append(0.); errors.append('missing_frame'); continue
        try:
            if task == 'objects':
                value = float(all(name in frame for name in values))
            else:
                components = []
                for triple in values:
                    if scheme != 'Origin' and spatial_backend == 'repair-v2':
                        components.append(spatial_repair.frame_score(triple, frame))
                    else:
                        mapped = {'object_a':triple['subject'], 'object_b':triple['object'], 'relationship':RELATIONS[triple['relation']]}
                        components.append(spatial_scores(mapped, [frame])[0])
                value = sum(components)/len(components)
            scores.append(value); covered += 1
        except (TypeError, ValueError, KeyError, ZeroDivisionError) as error:
            scores.append(0.); errors.append(type(error).__name__)
    return {**result, 'score':sum(scores)/16, 'coverage':covered/16, 'missing':1-covered/16,
            'status':'ok' if covered==16 else 'partial', 'frame_scores':scores, 'errors':errors}


def paired_rows(matrix, caches, transforms, predictions, vocab, codec=None, visibility=None, *, spatial_backend='repair-v2', action_interface='repair-v2.1', objects_backend='repair-v2'):
    originals = {(r['task'],r['relative_path']):r for r in matrix if r['transform']=='identity'}
    baseline = {}
    output = []
    for row in matrix:
        base_key = (row['task'],row['relative_path'])
        original = originals[base_key]
        original_evidence = caches.get(row['task'],{}).get(row['relative_path'],{})
        evidence = original_evidence if row['evidence_kind']=='original' else transforms.get(evidence_id(row),{})
        for scheme in SCHEMES:
            bk = (*base_key,scheme)
            if bk not in baseline:
                baseline[bk] = score_one(original, scheme, original_evidence, predictions, vocab, codec, spatial_backend=spatial_backend, action_interface=action_interface, objects_backend=objects_backend)
            before = baseline[bk]
            after = score_one(row, scheme, evidence, predictions, vocab, codec, spatial_backend=spatial_backend, action_interface=action_interface, objects_backend=objects_backend)
            covered = before['coverage']==1 and after['coverage']==1
            informative = before['abstention']<1 and after['abstention']<1
            delta = after['score'] - before['score']
            expected = row.get('expected','identity')
            invariant = expected in {'invariant','identity'} or row['transform']=='identity' or row.get('level')==0
            review = (visibility or {}).get(evidence_id(row), {}) if row['task']=='objects' and row.get('occlusion_control')=='target' and row.get('level')==1 else {}
            endpoint = bool(review.get('endpoint_invisible_verified'))
            removal = bool(review.get('removal_verified'))
            verified = bool(row.get('human_expectation_verified')) or removal
            objects_v2 = row['task']=='objects' and scheme!='Origin' and objects_backend=='repair-v2'
            # Unconfirmed detections are not verified negatives. Keep score
            # suppression separate from a fully observed/decisive rejection.
            decisive = not objects_v2 or before['abstention']==after['abstention']==0
            result = {**row, 'scheme':scheme, 'original_score':before['score'], 'transformed_score':after['score'],
                'paired_delta':delta, 'original_coverage':before['coverage'], 'transformed_coverage':after['coverage'],
                'original_abstention':before['abstention'], 'transformed_abstention':after['abstention'],
                'base_positive':before['score']>0, 'pair_complete':covered,
                'invariance_applicable':invariant, 'invariance_success':invariant and covered and informative and abs(delta)<=.03,
                'raw_drop_at_least_margin':covered and delta<=-.03,
                'expectation_verified':verified,
                'endpoint_invisible_verified':endpoint, 'removal_verified':removal,
                'verification_method':review.get('method'), 'visibility_status':review.get('status'),
                'endpoint_all_rejected':endpoint and after['coverage']==1 and after['score']==0 and (not objects_v2 or after['abstention']==0),
                'endpoint_negative_frame_fraction':max(0.,after['coverage']-after['score']-(after['abstention'] if objects_v2 else 0.)) if endpoint else 0.,
                'validated_sensitivity_success':verified and covered and decisive and before['score']>0 and delta<=-.03,
                'original_target':before['target'], 'transformed_target':after['target'],
                'original_backend_target':before.get('backend_target'), 'transformed_backend_target':after.get('backend_target'),
                'original_status':before['status'], 'transformed_status':after['status'],
                'cache_status':evidence.get('status','missing'),
                'transformed_frame_scores':after['frame_scores'], 'original_frame_scores':before['frame_scores']}
            if objects_v2:
                result['original_objects_resolution'] = before.get('objects_resolution')
                result['transformed_objects_resolution'] = after.get('objects_resolution')
                result['endpoint_no_positive'] = endpoint and after['coverage']==1 and after['score']==0
            if row['task']=='action':
                result['original_known_class_score'] = before.get('known_class_score')
                result['transformed_known_class_score'] = after.get('known_class_score')
                if scheme == 'Repair-model' and action_interface in {'repair-v2.1', 'repair-v2'}:
                    result['original_action_resolution'] = before.get('action_resolution')
                    result['transformed_action_resolution'] = after.get('action_resolution')
            if row['task']=='action' and row.get('expected_actions') is not None:
                value = after['target'] or {}
                result['protocol_correct'] = set(value.get('actions',[])) == set(row['expected_actions'])
            output.append(result)
    return output


def unique_protocol_inputs(rows):
    """Prompt-only protocol claims must not count repeated videos as new texts."""
    unique={}
    for row in rows:
        key=text_key('action',row['prompt'])
        if key in unique and unique[key]['protocol_correct']!=row['protocol_correct']:
            raise ValueError('one prompt has inconsistent deterministic protocol outcomes')
        unique[key]={**row,'family':'action-text:'+key}
    return list(unique.values())


def summarize(rows, *, rounds=2000):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['task'],row['scheme'],row['transform'])].append(row)
    summary = []
    for (task, scheme, transform), values in sorted(grouped.items()):
        eligible = [r for r in values if r['eligible']]
        item = {'task':task, 'scheme':scheme, 'transform':transform, 'planned':len(values), 'eligible':len(eligible),
            'ineligible':len(values)-len(eligible), 'families':len({r['family'] for r in eligible}),
            'status':dict(Counter(r['transformed_status'] for r in eligible)),
            'verified_expectation':sum(r['expectation_verified'] for r in eligible),
            'human_verified_expectation':sum(bool(r.get('human_expectation_verified')) for r in eligible),
            'vision_model_verified_invisible_endpoints':sum(r['endpoint_invisible_verified'] for r in eligible),
            'vision_model_verified_isolated_removals':sum(r['removal_verified'] for r in eligible)}
        for key in ['original_score','transformed_score','paired_delta','original_coverage','transformed_coverage',
                    'original_abstention','transformed_abstention']:
            item[key] = cluster_summary(eligible,key,rounds=rounds)
        item['full_plan_original_score'] = cluster_summary(values,'original_score',rounds=rounds)
        item['full_plan_transformed_score'] = cluster_summary(values,'transformed_score',rounds=rounds)
        complete = [r for r in eligible if r['pair_complete']]
        item['complete_pairs'] = len(complete)
        for key in ['original_score','transformed_score','paired_delta']:
            item['complete_pair_'+key] = cluster_summary(complete,key,rounds=rounds)
        inv = [r for r in eligible if r['invariance_applicable']]
        item['invariance_success'] = cluster_summary(inv,'invariance_success',rounds=rounds)
        positive = [r for r in eligible if r['base_positive']]
        item['raw_drop_given_positive_base'] = cluster_summary(positive,'raw_drop_at_least_margin',rounds=rounds)
        verified = [r for r in positive if r['expectation_verified']]
        item['validated_sensitivity_success'] = cluster_summary(verified,'validated_sensitivity_success',rounds=rounds)
        endpoints = [r for r in eligible if r['endpoint_invisible_verified']]
        item['verified_endpoint_all_rejected'] = cluster_summary(endpoints,'endpoint_all_rejected',rounds=rounds)
        item['verified_endpoint_negative_frame_fraction'] = cluster_summary(endpoints,'endpoint_negative_frame_fraction',rounds=rounds)
        item['full_plan_verified_endpoint_success'] = cluster_summary(eligible,'endpoint_all_rejected',rounds=rounds)
        protocol = unique_protocol_inputs([r for r in eligible if 'protocol_correct' in r])
        item['action_protocol_unique_prompts'] = len(protocol)
        item['action_protocol_correct'] = cluster_summary(protocol,'protocol_correct',rounds=rounds)
        summary.append(item)
    return summary


def main(argv=None):
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', required=True)
    p.add_argument('--metadata', required=True, help='full public VBench metadata; native entity serialization inventory')
    p.add_argument('--visibility', help='independent Objects endpoint model consensus JSONL')
    p.add_argument('--cache', action='append', default=[], help='task=original JSONL')
    p.add_argument('--transforms', action='append', default=[], help='transformed evidence JSONL, repeatable')
    p.add_argument('--predictions', action='append', default=[], help='adapter prediction JSONL; Action v9 for primary')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--allow-incomplete', action='store_true', help='explicit diagnostic; stamp unfinished execution in report')
    p.add_argument('--spatial-backend', choices=['repair-v2','legacy-official'], default='repair-v2',
                   help='Repair geometry/serialization only; Origin always uses frozen official rules')
    p.add_argument('--action-interface', choices=['repair-v2.1','repair-v2','legacy-model'], default='repair-v2.1',
                   help='Repair-model v9 plus declared text contract, or frozen legacy targets; no score copying')
    p.add_argument('--objects-backend', choices=['repair-v2','legacy-official'], default='repair-v2',
                   help='Repair same-label adjacent-frame box confirmation; Origin keeps single-frame labels')
    args=p.parse_args(argv)
    matrix=J.read_jsonl(Path(args.matrix))
    if not matrix:
        raise ValueError('empty matrix')
    cache_paths=dict(v.split('=',1) for v in args.cache)
    paths=[args.matrix,args.metadata,*cache_paths.values(),*args.transforms,*args.predictions]
    if args.visibility:
        paths.append(args.visibility)
    if any(not Path(path).is_file() for path in paths):
        raise ValueError('specified artifact file is missing')
    incomplete=[]
    if set(cache_paths)!={'spatial','scene','action','objects'}:
        incomplete.append('not all four original caches provided')
    for kind, files, flag in [('cache',list(cache_paths.values()),'complete'),
                              ('transforms',args.transforms,'all_attempted'),
                              ('predictions',args.predictions,'complete'),
                              ('visibility',[args.visibility] if args.visibility else [],'complete')]:
        for path in files:
            report_path=Path(path).with_suffix('.report.json')
            if not report_path.exists() or not json.loads(report_path.read_text()).get(flag):
                incomplete.append(kind+':'+path)
    if len(args.transforms)<2 or len(args.predictions)<4:
        incomplete.append('missing transform/prediction task artifacts')
    if incomplete and not args.allow_incomplete:
        raise ValueError('unfinished execution artifacts: '+str(incomplete))
    caches={task:{r['relative_path']:r for r in J.read_jsonl(Path(path))} for task,path in cache_paths.items()}
    transforms={r['evidence_id']:r for p in args.transforms for r in J.read_jsonl(Path(p))}
    visibility={r['evidence_id']:r for r in J.read_jsonl(Path(args.visibility))} if args.visibility else {}
    for key,review in visibility.items():
        if review.get('transform_record_sha256'):
            actual=hashlib.sha256(json.dumps(transforms.get(key),sort_keys=True).encode()).hexdigest()
            if actual!=review['transform_record_sha256']:
                raise ValueError('visibility labels refer to different transformed evidence: '+key)
    codec=native_entity_codec(json.loads(Path(args.metadata).read_text()))
    predictions={}
    for path in args.predictions:
        for row in J.read_jsonl(Path(path)):
            key=row.get('request_id', text_key(row['task'],row['prompt'],row.get('caption')))
            if key in predictions:
                raise ValueError('duplicate prediction identity; do not combine different adapters for one task')
            predictions[key]=row.get('target')
    required_predictions=set()
    for row in matrix:
        evidence=caches.get(row['task'],{}).get(row['relative_path'])
        if evidence is None:
            incomplete.append('missing original record:'+row['task']+':'+row['relative_path'])
        if row['evidence_kind']!='original' and evidence_id(row) not in transforms:
            incomplete.append('unattempted evidence transform:'+evidence_id(row))
        if not row['eligible']:continue
        if row['task']=='scene':
            required_predictions.update(text_key('scene',row['prompt'],caption) for caption in (evidence or {}).get('frame_captions',[]) if isinstance(caption,str))
        else:required_predictions.add(text_key(row['task'],row['prompt']))
    if missing:=required_predictions-set(predictions):
        incomplete.append('missing matrix text predictions:'+str(len(missing)))
    if incomplete and not args.allow_incomplete:
        raise ValueError('unfinished execution artifacts: '+str(incomplete[:20]))
    output=paired_rows(matrix,caches,transforms,predictions,load_k400(),codec,visibility,
                       spatial_backend=args.spatial_backend,action_interface=args.action_interface,objects_backend=args.objects_backend)
    summary=summarize(output)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    R.write_jsonl(out/'paired-rows.jsonl',output)
    sources=['scripts/score_matrix.py','scripts/build_matrix_plan.py','scripts/cache_matrix_transforms.py',
             'src/vbench_prompts_compile/experiments.py','src/vbench_prompts_compile/official_replay.py',
             'src/vbench_prompts_compile/spatial_repair.py',
             'src/vbench_prompts_compile/objects_repair.py',
             'src/vbench_prompts_compile/action_repair.py','src/vbench_prompts_compile/action_lexicon.py',
             'src/vbench_prompts_compile/records.py','src/vbench_prompts_compile/sources.py']
    supporting={str(Path(p).with_suffix(suffix)) for p in paths for suffix in ['.plan.json','.manifest.json','.report.json'] if Path(p).with_suffix(suffix).is_file()}
    supporting.add(str(Path(args.matrix).parent/'manifest.json'))
    report={'upstream_sha':UPSTREAM_SHA, 'input_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        'scorer_source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'supporting_artifact_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(supporting) if Path(p).is_file()},
        'scorer_git_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'scorer_dirty_diff_sha256':hashlib.sha256(subprocess.check_output(['git','-C',str(ROOT),'diff','HEAD'])).hexdigest(),
        'required_unique_matrix_text_inputs':len(required_predictions),'scoring_wall_seconds':time.monotonic()-started,
        'execution_complete':not incomplete, 'incomplete_artifacts':incomplete,
        'spatial_backend':args.spatial_backend,
        'objects_backend':args.objects_backend,
        'objects_repair_definition':'fixed adapter/rule entities plus adjacent same-label box confirmation, IoU >= 0.5; unconfirmed is abstention, not verified absence' if args.objects_backend=='repair-v2' else 'historical single-frame label conjunction',
        'action_interface':args.action_interface, 'action_lexicon_sha256':LEXICON_SHA256,
        'action_vocabulary_sha256':hashlib.sha256((ROOT/'data/raw/k400/labels.json').read_bytes()).hexdigest(),
        'action_repair_definition':'fixed adapter plus declared text equivalence contract; v2.1 restricts overrides to the declared prompt grammar; not learned synonym generalization' if args.action_interface in {'repair-v2','repair-v2.1'} else 'historical fixed adapter target',
        'protocol':'fixed 0.03 margins; 2000 paired source-family bootstrap draws; zero for missing outputs; Origin official unsigned geometry; Spatial Repair '+args.spatial_backend,
        'entity_codec':codec,
        'summary':summary, 'note':'Paired score change is not human-verified correctness. Supplied Objects labels are independent vision-model consensus, not human gold. Unverified endpoints remain in full-plan denominators.'}
    J.atomic_json(out/'report.json',report)
    columns=['task','scheme','transform','planned','eligible','families','original_score','transformed_score','paired_delta',
             'delta_ci_low','delta_ci_high','original_coverage','transformed_coverage','original_abstention','transformed_abstention']
    flat=[]
    for r in summary:
        row={k:r[k] for k in columns[:6]}
        for k in ['original_score','transformed_score','paired_delta','original_coverage','transformed_coverage','original_abstention','transformed_abstention']:
            row[k]=r[k]['estimate']
        ci=r['paired_delta']['ci'] or [None,None]
        row['delta_ci_low'],row['delta_ci_high']=ci
        flat.append(row)
    with (out/'main-table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(flat)
    lines=['| Dimension | Scheme | Transform | Eligible / planned | Original | Transformed | Paired delta (95% family CI) | Coverage |',
           '| --- | --- | --- | ---: | ---: | ---: | --- | ---: |']
    def fmt(x):return 'N/A' if x is None else f'{x:.4f}'
    for r in flat:
        lines.append(f"| {r['task']} | {r['scheme']} | {r['transform']} | {r['eligible']} / {r['planned']} | {fmt(r['original_score'])} | {fmt(r['transformed_score'])} | {fmt(r['paired_delta'])} [{fmt(r['delta_ci_low'])}, {fmt(r['delta_ci_high'])}] | {fmt(r['transformed_coverage'])} |")
    (out/'main-table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'planned_matrix_rows':len(matrix),'scored_rows':len(output),'summary_rows':len(summary),'out':str(out)}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
