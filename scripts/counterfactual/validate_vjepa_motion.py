"""Frozen V-JEPA head validation on real VBench 1.0 original/8px videos.

select: metadata only; prepare: collect verified construction and natural inputs;
score: one isolated GPU/backend shard; summarize: read labels only after all
predictions are complete. No training, downloads or score rescaling.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .static_jitter import ROOT, digest
from .vjepa_motion_probe import read_rows, write_json, write_rows, fresh_output, gpu_precheck


def check_sha(path, expected):
    if digest(Path(path)) != expected:
        raise ValueError(f'identity mismatch: {path}')


def select_metadata(pool, reserved, probe, dev):
    test = [dict(r) for r in reserved if r['split'] == 'test']
    prompts = {r['prompt_id'] for r in test}
    forbidden = {r['prompt_id'] for r in probe + dev}
    if prompts & forbidden:
        raise ValueError('holdout prompt leakage')
    native = [dict(r, base_id=r['video_uid']) for r in test if r['relative_video_path'].endswith('.mp4')]
    excluded = [dict(r, status='NOT SCORED', reason='GIF outside fixed native MP4/16-frame/8-FPS protocol; no invented timestamps')
                for r in test if not r['relative_video_path'].endswith('.mp4')]
    natural = [dict(r, base_id=r['video_uid']) for r in pool if r['dimension']=='dynamics_degree' and r['split']=='test'
               and r['prompt_id'] in prompts and r['relative_video_path'].endswith('.mp4')]
    if (len(test),len(prompts),len(native),len(excluded),len(natural)) != (120,30,90,30,450):
        raise ValueError('predeclared holdout population changed')
    if not {r['video_uid'] for r in native} <= {r['video_uid'] for r in natural}:
        raise ValueError('reserved sources not in official natural universe')
    if set(Counter(r['prompt_id'] for r in native).values()) != {3} or set(Counter(r['prompt_id'] for r in natural).values()) != {15}:
        raise ValueError('prompt/generator balance changed')
    return tuple(sorted(items,key=lambda r:r['video_uid']) for items in (native,natural,excluded))


def select(args, config):
    for path,key in [(args.pool,'pool_sha256'),(args.reserved,'reserved_sources_sha256'),
                     (args.probe_sources,'probe_sources_sha256'),(args.dev_sources,'dev32_sources_sha256')]:
        check_sha(path,config[key])
    with Path(args.pool).open() as handle:
        native,natural,excluded=select_metadata(list(csv.DictReader(handle)),read_rows(args.reserved),
                                              read_rows(args.probe_sources),read_rows(args.dev_sources))
    out=fresh_output(args.output)
    write_rows(out/'counterfactual_sources.jsonl',native)
    write_rows(out/'natural_sources.jsonl',natural)
    write_rows(out/'excluded_sources.jsonl',excluded)
    write_json(out/'construction.json',config['construction'])
    write_json(out/'selection.json',{'config_sha256':digest(Path(args.config)), 'selector_sha256':digest(Path(__file__)),
        'files':{p.name:digest(p) for p in sorted(out.iterdir())}, 'test_prompts':30,
        'counterfactual_sources':90,'natural_sources':450,'excluded_sources':30,
        'scores_read':False,'human_label_values_read':False,'test_media_opened':False,
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    print((out/'selection.json').read_text(),flush=True)


def verify_selection(path, config_path):
    path=Path(path)
    receipt=json.loads((path/'selection.json').read_text())
    check_sha(config_path,receipt['config_sha256'])
    for name,sha in receipt['files'].items():
        check_sha(path/name,sha)


def prepare(args, config):
    from .official_video_jitter import native_video

    verify_selection(args.selection,args.config)
    check_sha(args.dev_manifest,config['dev32_inputs_sha256'])
    native=read_rows(Path(args.selection)/'counterfactual_sources.jsonl')
    natural=read_rows(Path(args.selection)/'natural_sources.jsonl')
    out=fresh_output(args.output)
    rows=[]
    for row in read_rows(args.dev_manifest):
        rows.append(dict(row,cohort='dev32',evaluation_id='dev32:'+row['candidate_id'],counterfactual_primary=True))
    manifests=sorted(Path(args.construction).glob('shard-*/candidates.jsonl'))
    if len(manifests)!=4:
        raise ValueError('four complete construction shards required')
    test_rows=[]
    for manifest in manifests:
        completion=json.loads((manifest.parent/'completion.json').read_text())
        provenance=json.loads((manifest.parent/'construction.json').read_text())
        check_sha(manifest,completion['manifest_sha256'])
        check_sha(Path(args.selection)/'construction.json',provenance['config_sha256'])
        if completion['status']!='finished' or completion['counts'].get('construction_failed',0):
            raise ValueError('incomplete/failed construction')
        test_rows.extend(read_rows(manifest))
    expected={(r['video_uid'],fam,seed) for r in native for fam,seed in
              [('original',0),('encoding_control',0),('local_texture_alternating',1701),('local_texture_alternating',2904)]}
    if len(test_rows)!=360 or {(r['base_id'],r['family'],r['seed']) for r in test_rows}!=expected:
        raise ValueError('holdout construction coverage mismatch')
    rows.extend(dict(r,cohort='holdout',evaluation_id='holdout:'+r['candidate_id'],counterfactual_primary=True) for r in test_rows)
    primary={r['video_uid'] for r in native}
    root=Path(args.video_root).resolve()
    for source in natural:
        if source['video_uid'] in primary:
            continue
        path=(root/source['relative_video_path']).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('source escapes official root')
        frames,pts,fps=native_video(path,1e-6)
        if len(frames)!=16 or abs(fps-8)>1e-6 or frames.shape[1]!=frames.shape[2]:
            raise ValueError('natural source outside native-time protocol; never silently replace')
        rows.append(dict(source,cohort='holdout',evaluation_id='natural:'+source['video_uid'],
            candidate_id='natural:'+source['video_uid'],counterfactual_primary=False,video=str(path),
            family='original',seed=0,amplitude=0,sha256=digest(path),status='qualified',reason=None,
            pixel_exact_to_intended=True,native_timeline_preserved=True,decoded_shape=list(frames.shape),pts=pts,fps=fps,
            decoded_pixels_sha256=hashlib.sha256(frames.tobytes()).hexdigest()))
    if len(rows)!=config['cohorts']['total_scoring_inputs'] or len({r['evaluation_id'] for r in rows})!=len(rows):
        raise ValueError('combined coverage mismatch')
    for row in rows:
        check_sha(row['video'],row['sha256'])
        if not row['pixel_exact_to_intended'] or not row['native_timeline_preserved'] or row['decoded_shape'][0]!=16 or abs(row['fps']-8)>1e-6:
            raise ValueError('input timeline/pixel identity failure')
    old_base=[r for p in (Path(args.probe_root)/'features').glob('shard-*/features.jsonl') for r in read_rows(p) if r.get('view')=='base']
    old_hash={r['video_sha256'] for r in old_base}
    originals=[r for r in rows if r['family']=='original']
    if old_hash & {r['sha256'] for r in originals}:
        raise ValueError('byte-identical media overlap with probe train/validation')
    rows.sort(key=lambda r:(r['base_id'],r['evaluation_id']))
    write_rows(out/'inputs.jsonl',rows)
    write_json(out/'completion.json',{'status':'finished','inputs':len(rows),'originals':len(originals),
        'manifest_sha256':digest(out/'inputs.jsonl'),'config_sha256':digest(Path(args.config)),
        'prepare_sha256':digest(Path(__file__)),'selection_sha256':digest(Path(args.selection)/'selection.json'),
        'construction_manifests':{str(p):digest(p) for p in manifests},
        'cross_probe_byte_overlap':0,'construction_status_counts':dict(Counter(r['status'] for r in rows)),
        'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    print((out/'completion.json').read_text(),flush=True)


def execution_identity():
    files=[Path(__file__),ROOT/'configs/upstream.toml']
    for pattern in ['packages/audit-core/src/**/*.py','packages/audit-models/src/**/*.py','metrics/dynamic-degree/src/**/*.py']:
        files.extend(ROOT.glob(pattern))
    files.extend(Path(__file__).parent/name for name in ('vjepa_motion_probe.py','official_video_jitter.py','local_texture_jitter.py','static_jitter.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))}


def score(args,config):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import cv2
    import torch
    from .official_video_jitter import native_video

    start=time.monotonic()
    gpu=gpu_precheck()
    torch.set_num_threads(3); cv2.setNumThreads(1)
    if not 0<=args.shard<args.shards:
        raise ValueError('invalid shard')
    receipt=json.loads((Path(args.inputs).parent/'completion.json').read_text())
    check_sha(args.inputs,receipt['manifest_sha256']); check_sha(args.config,receipt['config_sha256'])
    for name,sha in config['implementation_sha256'].items():
        check_sha(ROOT/name,sha)
    rows=read_rows(args.inputs)
    bases=sorted({r['base_id'] for r in rows})
    selected=set(bases[args.shard::args.shards])
    rows=[r for r in rows if r['base_id'] in selected]
    out=fresh_output(args.output)
    model_identity={}
    if args.backend=='origin':
        from dynamic_degree.backends.vbench import OfficialDynamicEvaluator,official_result_payload
        check_sha(args.raft_weight,config['raft_sha256'])
        origin=OfficialDynamicEvaluator(torch.device('cuda:0'),Path(args.raft_weight),Path(args.upstream))
        if origin.upstream_state.sha!=config['origin_upstream_commit'] or origin.upstream_state.dirty:
            raise ValueError('official source is not the clean pinned checkout')
        model_identity={'upstream':asdict(origin.upstream_state),'raft_sha256':digest(Path(args.raft_weight))}
    else:
        from vbench_audit_models.vjepa import FrozenVJEPA
        from dynamic_degree.learned_probe import MotionProbe
        probe=Path(args.probe_root)
        probe_config=ROOT/'configs/dynamic-static-jitter/vjepa-probe-v1.json'
        check_sha(probe_config,config['probe_config_sha256'])
        pc=json.loads(probe_config.read_text()); hc=pc['head']
        torch.backends.cuda.matmul.allow_tf32=False
        encoder=FrozenVJEPA(probe/'assets/vjepa2',probe/'assets/vjepa2_1_vitb_dist_vitG_384.pt',pc)
        heads={}
        for arm,sha in config['heads'].items():
            path=probe/'training'/f'{arm}.pt'; check_sha(path,sha)
            head=MotionProbe(hc['input_dim'],hc['hidden_dim'],hc['mlp_dim'])
            head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
            heads[arm]=head.eval().requires_grad_(False).to('cuda:0')
        model_identity={'encoder':encoder.identity,'heads':config['heads'],'training_updates':0}
        # Pre-existing training clip: check adapter identity before new-test inference.
        reference=next(r for r in read_rows(probe/'features/shard-0/features.jsonl') if r['view']=='base')
        path=Path(args.video_root)/reference['relative_video_path']
        check_sha(path,reference['video_sha256']); check_sha(reference['feature_path'],reference['feature_sha256'])
        frames,_,_=native_video(path,1e-6)
        expected=np.load(reference['feature_path'],allow_pickle=False)
        actual=encoder.encode(frames)
        if not np.array_equal(actual,expected):
            raise ValueError('frozen encoder differs from training feature cache')
        prior={r['arm']:r for r in read_rows(probe/'training/scores.jsonl') if r['video_uid']==reference['video_uid']}
        parity={}
        with torch.inference_mode():
            tensor=torch.from_numpy(actual).to('cuda:0',torch.float32)[None]
            for arm,head in heads.items():
                error=abs(float(head(tensor)[0])-prior[arm]['latent'][0]); parity[arm]=error
                if error>config['score']['head_reload_tolerance']:
                    raise ValueError('head inference differs from frozen training result')
        model_identity['pretest_loading_check']={'feature_exact':True,'head_max_errors':parity,'uid':reference['video_uid']}
        (out/'features').mkdir()
    write_json(out/'provenance.json',{'config_sha256':digest(Path(args.config)),'input_sha256':digest(Path(args.inputs)),
        'code':execution_identity(),'model':model_identity,'gpu':gpu,'backend':args.backend,'shard':args.shard,'shards':args.shards,
        'torch':torch.__version__,'numpy':np.__version__,'opencv':cv2.__version__,
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'expected_ids':[r['evaluation_id'] for r in rows]})
    failed=0
    with (out/'scores.jsonl').open('x') as handle:
        for i,row in enumerate(rows):
            result={k:row[k] for k in ('evaluation_id','base_id','cohort','family','seed','prompt_id','generator','counterfactual_primary')}
            try:
                video=Path(row['video']); check_sha(video,row['sha256'])
                result['input_sha256']=row['sha256']
                if args.backend=='origin':
                    measured=origin.evaluate_video(video)
                    result.update(score=float(measured.official_video_boolean),diagnostics=official_result_payload(measured))
                    if i==0:
                        if bool(origin.dynamic.infer(str(video)))!=measured.official_video_boolean:
                            raise ValueError('official reference infer parity failure')
                        result['reference_infer_parity']=True
                else:
                    frames,pts,fps=native_video(video,1e-6)
                    if (hashlib.sha256(frames.tobytes()).hexdigest()!=row['decoded_pixels_sha256']
                            or frames.shape[0]!=16 or abs(fps-8)>1e-6 or not np.allclose(pts,row['pts'],atol=1e-6,rtol=0)):
                        raise ValueError('native decoded pixels or timestamps changed')
                    features=encoder.encode(frames)
                    path=out/'features'/f"{row['evaluation_id'].replace(':','_')}.npy"
                    with path.open('xb') as f: np.save(f,features,allow_pickle=False)
                    result.update(feature_path=str(path),feature_sha256=digest(path))
                    with torch.inference_mode():
                        tensor=torch.from_numpy(features).to('cuda:0',torch.float32)[None]
                        for arm,head in heads.items():
                            q=float(head(tensor)[0]); value=float(1/(1+np.exp(-q)))
                            if not np.isfinite(q): raise ValueError('nonfinite score')
                            result[arm]={'latent':q,'score':value}
                result['status']='ok'
            except Exception as exc:
                failed+=1; result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            handle.write(json.dumps(result,allow_nan=False)+'\n'); handle.flush()
            if (i+1)%10==0 or i+1==len(rows):
                print(json.dumps({'backend':args.backend,'shard':args.shard,'completed':i+1,'expected':len(rows),'failed':failed}),flush=True)
            if i==0 and failed:
                raise RuntimeError(result['error'])
    write_json(out/'completion.json',{'status':'finished' if not failed else 'failed','expected':len(rows),'completed':len(rows),
        'failed':failed,'scores_sha256':digest(out/'scores.jsonl'),'wall_seconds':time.monotonic()-start,
        'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    if failed: raise RuntimeError('incomplete scoring: retain failures, do not silently omit')


def cluster_interval(values,prompts,seed,replicates):
    groups=sorted(set(prompts)); rng=np.random.default_rng(seed)
    totals=np.array([np.asarray(values)[np.asarray(prompts)==p].sum() for p in groups])
    counts=np.array([sum(q==p for q in prompts) for p in groups])
    draws=rng.integers(len(groups),size=(replicates,len(groups)))
    estimates=totals[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return np.quantile(estimates,[.025,.975]).tolist()


def paired_stats(pairs,config):
    output={}; changes={}; prompts=[r['prompt_id'] for r in pairs]
    a=config['analysis']
    for backend in ('origin','joint','natural_only','joint_latent'):
        values=np.array([r[backend] for r in pairs],float)
        difference=values[:,1:]-values[:,:1]; delta=difference.mean(axis=1); changes[backend]=delta
        output[backend]={'base_mean':float(values[:,0].mean()),'cf_mean':float(values[:,1:].mean()),
            'delta':float(delta.mean()),'mae':float(abs(difference).mean()),'base_std':float(values[:,0].std()),
            'base_min':float(values[:,0].min()),'base_max':float(values[:,0].max()),
            'cf_by_seed':dict(zip(('1701','2904'),values[:,1:].mean(axis=0).tolist())),
            'maximum_absolute_change':float(abs(difference).max()),'largest_drop':float(difference.min()),
            'delta_prompt_ci95':cluster_interval(delta,prompts,a['bootstrap_seed'],a['bootstrap_replicates']),
            'mae_prompt_ci95':cluster_interval(abs(difference).mean(axis=1),prompts,a['bootstrap_seed'],a['bootstrap_replicates'])}
    o,r=output['origin']['delta'],output['joint']['delta']
    output['fixed_scale_numerical_check']={'origin_increased':o>0,'allowed_delta':.1*o if o>0 else None,
        'ratio':r/o if o>0 else None,'point_estimate_pass':bool(r<=.1*o) if o>0 else None,
        'margin_prompt_ci95':cluster_interval(changes['joint']-.1*changes['origin'],prompts,a['bootstrap_seed'],a['bootstrap_replicates']),
        'absolute_strength_calibrated':False,'overall_goal_complete':False}
    return output


def human_statistics(pairs,lookup,config):
    ordered=[r for r in pairs if r['label']!=.5]; ties=[r for r in pairs if r['label']==.5]
    out={}
    for backend in ('origin','joint','natural_only'):
        margins=np.array([(lookup[r['a']][backend]-lookup[r['b']][backend])*(2*r['label']-1) for r in ordered])
        gaps=np.array([abs(lookup[r['a']][backend]-lookup[r['b']][backend]) for r in ties])
        credit=(margins>0).astype(float)+.5*(margins==0)
        a=config['analysis']
        out[backend]={'ordered':len(ordered),'strict_correct':int((margins>0).sum()),'predicted_ties':int((margins==0).sum()),
            'incorrect':int((margins<0).sum()),'concordance':float(credit.mean()) if len(ordered) else None,
            'concordance_prompt_ci95':cluster_interval(credit,[r['prompt_id'] for r in ordered],a['bootstrap_seed'],a['bootstrap_replicates']) if len(ordered) else None,
            'human_ties':len(ties),'human_tie_mean_absolute_gap':float(gaps.mean()) if len(ties) else None}
    if ordered:
        credits={}
        for backend in ('origin','joint','natural_only'):
            margins=np.array([(lookup[r['a']][backend]-lookup[r['b']][backend])*(2*r['label']-1) for r in ordered])
            credits[backend]=(margins>0).astype(float)+.5*(margins==0)
        out['joint_minus_origin']={'difference':float((credits['joint']-credits['origin']).mean()),
            'prompt_ci95':cluster_interval(credits['joint']-credits['origin'],[r['prompt_id'] for r in ordered],config['analysis']['bootstrap_seed'],config['analysis']['bootstrap_replicates'])}
    return out


def summarize(args,config):
    inputs=read_rows(args.inputs); ledger={r['evaluation_id']:r for r in inputs}
    if len(inputs)!=848 or len(ledger)!=848: raise ValueError('full fixed input cohort required')
    results={}; receipts={}
    for backend in ('origin','vjepa'):
        merged={}; receipts[backend]=[]
        shards=sorted((Path(args.scores)/backend).glob('shard-*'))
        if len(shards)!=4: raise ValueError('four completed shards required per backend')
        for shard in shards:
            c=json.loads((shard/'completion.json').read_text()); p=json.loads((shard/'provenance.json').read_text())
            if c['status']!='finished' or c['failed'] or c['completed']!=c['expected']: raise ValueError('incomplete scores')
            check_sha(shard/'scores.jsonl',c['scores_sha256']); check_sha(args.config,p['config_sha256']); check_sha(args.inputs,p['input_sha256'])
            for r in read_rows(shard/'scores.jsonl'):
                key=r['evaluation_id']
                if key in merged or key not in ledger or r['status']!='ok' or r['input_sha256']!=ledger[key]['sha256']:
                    raise ValueError('duplicate/missing/changed prediction')
                merged[key]=r
            receipts[backend].append({'shard':str(shard),'completion':c,'provenance_sha256':digest(shard/'provenance.json')})
        if set(merged)!=set(ledger): raise ValueError('not all expected inputs scored')
        results[backend]=merged
    def value(row):
        key=row['evaluation_id']; v=results['vjepa'][key]
        return {'origin':results['origin'][key]['score'],'joint':v['joint']['score'],
                'natural_only':v['natural_only']['score'],'joint_latent':v['joint']['latent']}
    pair_groups={}; controls=0
    for cohort,n in [('dev32',32),('holdout',90)]:
        pairs=[]
        bases=sorted({r['base_id'] for r in inputs if r['cohort']==cohort and r['counterfactual_primary']})
        if len(bases)!=n: raise ValueError('source coverage changed')
        for uid in bases:
            selected=[r for r in inputs if r['base_id']==uid]
            def get(family,seed=0):
                return next(r for r in selected if r['family']==family and r['seed']==seed)
            base,control=get('original'),get('encoding_control'); cf=[get('local_texture_alternating',s) for s in (1701,2904)]
            b,c=value(base),value(control)
            if b!=c or results['vjepa'][base['evaluation_id']]['feature_sha256']!=results['vjepa'][control['evaluation_id']]['feature_sha256']:
                raise ValueError('zero-edit encoding control failed')
            controls+=1
            row={'base_id':uid,'prompt_id':base['prompt_id'],'generator':base['generator'],
                 'quality_flags':[r.get('reason') for r in cf],
                 **{key:[b[key],*[value(r)[key] for r in cf]] for key in b}}
            pairs.append(row)
        pair_groups[cohort]=pairs
    # No test human label values are inspected until the complete predictions above are fixed.
    check_sha(args.human_pairs,config['human_pairs_sha256'])
    originals={r['base_id']:r for r in inputs if r['cohort']=='holdout' and r['family']=='original'}
    if len(originals)!=450: raise ValueError('incomplete natural holdout')
    human=[]
    with Path(args.human_pairs).open() as handle:
        for r in csv.DictReader(handle):
            if r['dimension']!='dynamics_degree' or r['split']!='test': continue
            if r['video_a_uid'] not in originals or r['video_b_uid'] not in originals: continue
            label=float(r['human_label'])
            if label not in (0,.5,1): raise ValueError('unexpected human label')
            human.append({'a':r['video_a_uid'],'b':r['video_b_uid'],'prompt_id':r['prompt_id'],'label':label})
    if len(human)!=450: raise ValueError('natural preference pair coverage changed')
    lookup={uid:value(row) for uid,row in originals.items()}
    natural_stats=human_statistics(human,lookup,config)
    cf_sources={r['base_id'] for r in pair_groups['holdout']}
    both=[r for r in human if r['a'] in cf_sources and r['b'] in cf_sources]
    cf_natural={'base':human_statistics(both,lookup,config)}
    for seed in (1701,2904):
        cf_lookup={r['base_id']:value(r) for r in inputs if r['cohort']=='holdout' and r['seed']==seed}
        cf_natural[str(seed)]=human_statistics(both,cf_lookup,config)
    out=fresh_output(args.output)
    write_rows(out/'pairs.jsonl',[dict(r,cohort=cohort) for cohort,group in pair_groups.items() for r in group])
    write_rows(out/'human_pairs.jsonl',human)
    write_json(out/'summary.json',{'status':'completed_frozen_model_validation_not_absolute_strength_certification',
        'config_sha256':digest(Path(args.config)),'input_sha256':digest(Path(args.inputs)),'analysis_sha256':digest(Path(__file__)),
        'coverage':{'inputs_per_backend':848,'dev32_sources':32,'holdout_cf_sources':90,'holdout_natural_sources':450,
                    'excluded_gif_sources':30,'encoding_controls_exact':controls,'runtime_failures':0},
        'counterfactual':{cohort:paired_stats(group,config) for cohort,group in pair_groups.items()},
        'natural_preferences':natural_stats,'cf_preference_overlap':{'pairs':len(both),'results':cf_natural},
        'shards':receipts,'absolute_strength_calibration':'NOT RUN','motion_type_human_review':'NOT RUN',
        'no_training_or_mapping_changes':True,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    print((out/'summary.json').read_text(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('select')
    for name in ('pool','reserved','probe-sources','dev-sources'): s.add_argument('--'+name,required=True)
    s=sub.add_parser('prepare')
    for name in ('selection','construction','dev-manifest','video-root','probe-root'): s.add_argument('--'+name,required=True)
    s=sub.add_parser('score')
    for name in ('inputs','probe-root','video-root','raft-weight','upstream'): s.add_argument('--'+name,required=True)
    s.add_argument('--backend',choices=['origin','vjepa'],required=True)
    s.add_argument('--shard',type=int,required=True); s.add_argument('--shards',type=int,default=4)
    s=sub.add_parser('summarize')
    for name in ('inputs','scores','human-pairs'): s.add_argument('--'+name,required=True)
    for s in sub.choices.values(): s.add_argument('--output',required=True)
    args=p.parse_args(); config=json.loads(Path(args.config).read_text())
    if config['protocol']!='dynamic-vjepa-frozen-official-validation-v1': raise ValueError('unexpected protocol')
    globals()[args.command](args,config)


if __name__=='__main__': main()
