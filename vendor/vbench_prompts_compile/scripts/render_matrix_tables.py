#!/usr/bin/env python3
"""Render reproducible CSV, Markdown and LaTeX tables from frozen scores.

Parsing targets are public benchmark metadata, not human annotations. Scene
classification uses separately identified model-labelled scorecards. Paired
video scores describe metric behaviour, not perceptual quality improvements.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile import records as R
from vbench_prompts_compile.experiments import cluster_summary, text_key
from vbench_prompts_compile.sources import load_k400
from vbench_prompts_compile.action_repair import compile_action
from score_matrix import SCHEMES, origin_compiled, parsed_target, backend_target, native_entity_codec
from cache_backend_outputs import sha


def fmt(value):
    if value is None:return 'N/A'
    if isinstance(value,float):return f'{value:.4f}'
    return str(value)


def ci_text(value):
    interval=value.get('ci')
    return fmt(value['estimate'])+(f' [{fmt(interval[0])}, {fmt(interval[1])}]' if interval else ' [N/A]')


def table(out,name,rows,columns):
    with (out/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');w.writeheader();w.writerows([{k:r.get(k) for k in columns} for r in rows])
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines.extend('| '+' | '.join(fmt(r.get(k)).replace('|','\\|') for k in columns)+' |' for r in rows)
    (out/(name+'.md')).write_text('\n'.join(lines)+'\n')
    def tex(s):
        return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
    lines=[r'\begin{longtable}{'+'l'*len(columns)+'}',r'\toprule',
        ' & '.join(tex(k) for k in columns)+r' \\',r'\midrule\endhead']
    lines.extend(' & '.join(tex(fmt(r.get(k))) for k in columns)+r' \\' for r in rows)
    lines.extend([r'\bottomrule',r'\end{longtable}'])
    (out/(name+'.tex')).write_text('\n'.join(lines)+'\n')


def target_set(task,target,*,canonicalize=False):
    if not isinstance(target,dict):return set()
    name=R.canonical_entity_name if canonicalize else str
    if task=='spatial':
        return {(name(r['subject']),r['relation'],name(r['object']))
                for r in target.get('relationships',[])}
    if task=='objects':return {name(v) for v in target.get('entities',[])}
    return set(target.get('actions',[]))


def parsing_table(matrix,predictions,codec=None,*,spatial_backend='legacy-official',action_interface='legacy-model'):
    unique={(r['task'],r['original_prompt']):r for r in matrix if r['transform']=='identity' and r['task']!='scene'}
    vocab=load_k400();details=[];summary=[]
    for (task,prompt),row in sorted(unique.items()):
        reference=origin_compiled(task,row['official_target'])
        expected=target_set(task,reference)
        semantic_reference=target_set(task,reference,canonicalize=True)
        for scheme in SCHEMES:
            parsed=parsed_target(task,row,scheme,predictions,vocab)
            compiled=parsed if scheme=='Origin' else backend_target(task,parsed,codec,normalize_spatial=spatial_backend=='repair-v2')
            if task=='action' and scheme=='Repair-model' and action_interface in {'repair-v2','repair-v2.1'}:
                compiled=compile_action(prompt,parsed,vocab,scope_guard=action_interface=='repair-v2.1')['target']
            actual=target_set(task,compiled)
            normalized=target_set(task,parsed,canonicalize=True)
            tp=len(actual&expected);denom=len(actual)+len(expected)
            details.append({'task':task,'scheme':scheme,'prompt':prompt,'family':row['family'],
                'eligible':row['eligible'],'expected':sorted(expected),'predicted':sorted(actual),'raw_target':parsed,
                'valid':parsed is not None,'exact':parsed is not None and actual==expected,
                'set_f1':2*tp/denom if denom else float(parsed is not None),
                'normalized_semantic_f1':2*len(normalized&semantic_reference)/(len(normalized)+len(semantic_reference)) if normalized or semantic_reference else float(parsed is not None),
                'prompt_words':len(prompt.split())})
    for task in sorted({r['task'] for r in details}):
        for scheme in SCHEMES:
            all_rows=[r for r in details if r['task']==task and r['scheme']==scheme]
            values=[r for r in all_rows if r['eligible']]
            summary.append({'Task':task,'Scheme':scheme,'Families':len(values),'Planned':len(all_rows),
                'Strict set F1 (CI)':ci_text(cluster_summary(values,'set_f1')),
                'Normalized semantic F1':cluster_summary(values,'normalized_semantic_f1')['estimate'],
                'Exact (CI)':ci_text(cluster_summary(values,'exact')),
                'Valid':sum(r['valid'] for r in values)/len(values) if values else None})
    return summary,details


def original_table(paired):
    rows=[r for r in paired if r['transform']=='identity']
    origins={(r['task'],r['relative_path']):r for r in rows if r['scheme']=='Origin'}
    groups=defaultdict(list)
    for row in rows:
        base=origins[(row['task'],row['relative_path'])]
        value={**row,'vs_origin':row['original_score']-base['original_score'],
               'common_coverage':min(row['original_coverage'],base['original_coverage'])}
        for generator in ['all',row['relative_path'].split('/')[1]]:
            groups[(row['task'],generator,row['scheme'])].append(value)
    result=[]
    for (task,generator,scheme),planned in sorted(groups.items()):
        values=[r for r in planned if r['eligible']]
        result.append({'Task':task,'Generator':generator,'Scheme':scheme,'N':len(values),'Planned':len(planned),
            'Families':len({r['family'] for r in values}),
            'Score (CI)':ci_text(cluster_summary(values,'original_score')),
            'Delta vs Origin (CI)':ci_text(cluster_summary(values,'vs_origin')),
            'Coverage':sum(r['common_coverage'] for r in values)/len(values) if values else None,
            'Abstain':sum(r['original_abstention'] for r in values)/len(values) if values else None})
    return result


def macro_f1(rows,scheme='Repair-model'):
    labels=['supported','contradicted','insufficient']
    counts=[[0,0,0] for _ in labels]
    for row in rows:
        truth=row['expected'];pred=row['schemes'][scheme]['label']
        for i,label in enumerate(labels):
            counts[i][0]+=truth==label and pred==label
            counts[i][1]+=truth!=label and pred==label
            counts[i][2]+=truth==label and pred!=label
    if any(tp+fn==0 for tp,fp,fn in counts):return None
    return sum(2*tp/(2*tp+fp+fn) for tp,fp,fn in counts)/len(labels)


def scene_macro_f1(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['family']].append(row)
    value=macro_f1(rows);distribution=[]
    if len(groups)>=2 and value is not None:
        rng=random.Random(20260919);blocks=list(groups.values())
        for _ in range(2000):
            sampled=[r for block in rng.choices(blocks,k=len(blocks)) for r in block]
            if (score:=macro_f1(sampled)) is not None:distribution.append(score)
    distribution.sort()
    return {'estimate':value,'ci':[distribution[int(.025*(len(distribution)-1))],distribution[int(.975*(len(distribution)-1))]] if distribution else None,
            'n':len(rows),'families':len(groups),'bootstrap_valid':len(distribution),'bootstrap_rounds':2000}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',required=True);p.add_argument('--paired',required=True);p.add_argument('--score-report',required=True)
    p.add_argument('--metadata',required=True,help='same full public entity inventory used by score_matrix')
    p.add_argument('--predictions',action='append',required=True)
    p.add_argument('--scene-card',action='append',default=[],help='name=scorecard JSON')
    p.add_argument('--object-controls',help='matched background and detector-side-effect report')
    p.add_argument('--out',required=True);p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args(argv);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    matrix=J.read_jsonl(Path(args.matrix));paired=J.read_jsonl(Path(args.paired));report=json.loads(Path(args.score_report).read_text())
    complete=bool(report.get('execution_complete')) and len(paired)==3*len(matrix)
    if not complete and not args.allow_incomplete:raise ValueError('incomplete matrix; use --allow-incomplete for a labelled draft')
    predictions={}
    for path in args.predictions:
        for row in J.read_jsonl(Path(path)):
            if row['request_id'] in predictions:raise ValueError('multiple adapters for one request')
            predictions[row['request_id']]=row['target']
    codec=native_entity_codec(json.loads(Path(args.metadata).read_text()))
    spatial_backend=report.get('spatial_backend','legacy-official')
    action_interface=report.get('action_interface','legacy-model')
    objects_backend=report.get('objects_backend','legacy-official')
    parsing,details=parsing_table(matrix,predictions,codec,spatial_backend=spatial_backend,action_interface=action_interface)
    table(out,'table-a-parsing',parsing,list(parsing[0]));R.write_jsonl(out/'parsing-details.jsonl',details)
    originals=original_table(paired);table(out,'table-b-originals',originals,list(originals[0]));original_table_rows=len(originals)
    transformed=[];visibility=[];conditional=[];protocol=[];compact=[];checks=[]
    for row in report['summary']:
        transformed.append({'Task':row['task'],'Scheme':row['scheme'],'Transform':row['transform'],
            'N/planned':str(row['eligible'])+'/'+str(row['planned']),'Families':row['families'],
            'Original':row['original_score']['estimate'],'Transformed':row['transformed_score']['estimate'],
            'Delta (CI)':ci_text(row['paired_delta']),'Coverage':row['transformed_coverage']['estimate'],
            'Abstain':row['transformed_abstention']['estimate']})
        if 'complete_pairs' in row:
            conditional.append({'Task':row['task'],'Scheme':row['scheme'],'Transform':row['transform'],
                'Complete/planned':str(row['complete_pairs'])+'/'+str(row['planned']),
                'Before':row['complete_pair_original_score']['estimate'],'After':row['complete_pair_transformed_score']['estimate'],
                'Delta (CI)':ci_text(row['complete_pair_paired_delta'])})
        if row['task']=='action' and row['transform']!='identity':
            protocol.append({'Scheme':row['scheme'],'Transform':row['transform'],'Videos':row['eligible'],
                'Unique prompts':row['action_protocol_correct']['n'],
                'Target set correct (CI)':ci_text(row['action_protocol_correct']),
                'Score invariant (CI)':ci_text(row['invariance_success']),
                'Abstain':row['transformed_abstention']['estimate']})
        if row['transform']!='identity':
            checks.append({'Task':row['task'],'Scheme':row['scheme'],'Transform':row['transform'],
                'Invariance N':row['invariance_success']['n'],'Invariant (CI)':ci_text(row['invariance_success']),
                'Positive-base N':row['raw_drop_given_positive_base']['n'],
                'Observed complete-pair drop (CI)':ci_text(row['raw_drop_given_positive_base']),
                'Verified positive N':row['validated_sensitivity_success']['n'],
                'Verified drop (CI)':ci_text(row['validated_sensitivity_success'])})
            transform=row['transform'].replace('_occlusion_',' mask ').replace('_',' ')
            compact.append({'Dim':row['task'],'Condition':transform,'Scheme':row['scheme'],
                'N/families':str(row['eligible'])+'/'+str(row['families']),
                'Before':row['original_score']['estimate'],'After':row['transformed_score']['estimate'],
                'Delta (95% CI)':ci_text(row['paired_delta'])})
        if row['task']=='objects' and row['transform']=='target_occlusion_1.0':
            visibility.append({'Scheme':row['scheme'],'Planned':row['planned'],
                'Invisible':row['vision_model_verified_invisible_endpoints'],
                'Isolated':row['vision_model_verified_isolated_removals'],
                'Positive isolated':row['validated_sensitivity_success']['n'],
                'All rejected (CI)':ci_text(row['verified_endpoint_all_rejected']),
                'Negative frames (CI)':ci_text(row['verified_endpoint_negative_frame_fraction']),
                'Verified success / full plan (CI)':ci_text(row['full_plan_verified_endpoint_success']),
                'Isolated positive-base drop (CI)':ci_text(row['validated_sensitivity_success'])})
    table(out,'table-c-transforms',transformed,list(transformed[0]))
    if conditional:table(out,'table-f-complete-pairs',conditional,list(conditional[0]))
    if protocol:table(out,'table-g-action-protocol',protocol,list(protocol[0]))
    if compact:table(out,'paper-main',compact,list(compact[0]))
    if visibility:table(out,'table-d-visibility',visibility,list(visibility[0]))
    if checks:table(out,'table-k-metamorphic-checks',checks,list(checks[0]))
    scene=[];scene_synonyms=[];scene_f1={}
    for item in args.scene_card:
        name,path=item.split('=',1);card=json.loads(Path(path).read_text());s=card['summary']
        originals=[r for r in card['rows'] if r['condition']!='synonym' and r['expected'] is not None]
        synonym_rows=[r for r in card['rows'] if r['condition']=='synonym']
        scene_f1[name]=scene_macro_f1(originals)
        for scheme in SCHEMES:
            common=s['primary_common_binary'][scheme];syn=s['synonyms'][scheme]
            scene.append({'Dataset':name,'Scheme':scheme,'Binary N':s['official_defined_observations'],'Source blocks':s['source_components'],
                'Binary accuracy (CI)':ci_text(common['binary_accuracy_full_denominator']),
                'Three-way N':len(originals) if scheme=='Repair-model' else None,
                'Three-way macro F1 (CI)':ci_text(scene_f1[name]) if scheme=='Repair-model' else 'N/A'})
            scene_synonyms.append({'Dataset':name,'Scheme':scheme,'Pairs':syn['correct_and_consistent']['n'],
                'Source blocks':syn['correct_and_consistent']['families'],
                'Binary correct and consistent (CI)':ci_text(syn['correct_and_consistent']),
                'Exact-label consistent (CI)':ci_text(syn['raw_consistent']),
                'Copy original (CI)':ci_text(syn['copy_original_correct_and_consistent']),
                'Always supported':sum(r['expected']=='supported' for r in synonym_rows)/len(synonym_rows) if synonym_rows else None,
                'Always not-supported':sum(r['expected'] in {'contradicted','insufficient'} for r in synonym_rows)/len(synonym_rows) if synonym_rows else None})
    if scene:table(out,'table-e-scene-classification',scene,list(scene[0]))
    if scene_synonyms:table(out,'table-h-scene-synonyms',scene_synonyms,list(scene_synonyms[0]))
    axes=defaultdict(list)
    for row in paired:
        if row['task']=='spatial' and row['eligible'] and row.get('axis'):
            axes[(row['axis'],row['scheme'],row['transform'])].append(row)
    axis_rows=[]
    for (axis,scheme,condition),values in sorted(axes.items()):
        axis_rows.append({'Axis':axis,'Scheme':scheme,'Condition':condition,'N':len(values),
            'Families':len({r['family'] for r in values}),'Before':cluster_summary(values,'original_score')['estimate'],
            'After':cluster_summary(values,'transformed_score')['estimate'],'Delta (CI)':ci_text(cluster_summary(values,'paired_delta'))})
    if axis_rows:table(out,'table-i-spatial-axes',axis_rows,list(axis_rows[0]))
    if args.object_controls:
        controls=json.loads(Path(args.object_controls).read_text());comparisons=[]
        for row in controls['matched_background_contrasts']:
            comparisons.append({'Scheme':row['scheme'],'Linear fraction':row['linear_fraction'],
                'Matched/planned':str(row['target']['n'])+'/'+str(row['planned_videos']),
                'Families':row['target']['families'],'Before':row['original']['estimate'],
                'Target mask':row['target']['estimate'],'Background mask':row['background']['estimate'],
                'Target minus background (CI)':ci_text(row['target_minus_background'])})
        if comparisons:table(out,'table-j-matched-controls',comparisons,list(comparisons[0]))
    J.atomic_json(out/'scene-macro-f1.json',scene_f1)
    paths=[args.matrix,args.metadata,args.paired,args.score_report,*args.predictions,*[v.split('=',1)[1] for v in args.scene_card]]
    if args.object_controls:paths.append(args.object_controls)
    J.atomic_json(out/'manifest.json',{'execution_complete':complete,'spatial_backend':spatial_backend,'action_interface':action_interface,'objects_backend':objects_backend,'input_sha256':{v:sha(v) for v in paths},'renderer_sha256':sha(__file__),
        'notes':['Parsing references are official structural metadata, not human gold.','Scene labels are model consensus.','Source-family bootstrap: 2000 draws, seed 20260919.','Scores and score gains are not measures of human correctness.']})
    doc=[r'\documentclass[10pt]{article}',r'\usepackage[paperwidth=21in,paperheight=11in,margin=0.4in]{geometry}',
         r'\usepackage{booktabs,longtable}',r'\begin{document}',r'\small',
         r'\noindent VBench deterministic matrix. '+('Complete execution.' if complete else 'DRAFT: transformations incomplete.'),
         r' Scores are metric outputs; annotations are model labels, not human gold.']
    for path in sorted(out.glob('table-*.tex')):
        doc.extend([r'\section*{'+path.stem.replace('-',' ')+r'}',r'\input{'+path.name+r'}',r'\clearpage'])
    doc.append(r'\end{document}');(out/'tables.tex').write_text('\n'.join(doc)+'\n')
    if compact:
        (out/'paper.tex').write_text('\n'.join([r'\documentclass{article}',r'\usepackage[a4paper,landscape,margin=0.5in]{geometry}',
            r'\usepackage{booktabs,longtable}',r'\begin{document}\footnotesize',
            r'Paired original/transformed scores; 95\% source-family bootstrap intervals. Missing outputs score zero; coverage and conditional results are in the appendix.',
            r'\input{paper-main.tex}',r'\end{document}'])+'\n')
    print(json.dumps({'out':str(out),'complete':complete,'parsing_rows':len(parsing),'original_rows':original_table_rows,'transform_rows':len(transformed),'scene_macro_f1':scene_f1}))
    return 0


if __name__=='__main__':raise SystemExit(main())
