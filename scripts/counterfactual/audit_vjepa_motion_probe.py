"""Independent CPU audit of fixed V-JEPA probe outputs, with prompt bootstrap.

No model calls, training, calibration, checkpoint selection or input mutation.
Only four validation prompts: intervals are descriptive, not general validation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def generator_confound(source, pairs):
    """Post-hoc diagnostic, not a new video model or a checkpoint criterion."""
    by_uid={r['video_uid']:r for r in source}
    generators=sorted({r['generator'] for r in source})
    eye=np.eye(len(generators))
    train=[r for r in pairs if r['role']=='train']
    design=np.array([eye[generators.index(by_uid[r['a']]['generator'])]
                     - eye[generators.index(by_uid[r['b']]['generator'])] for r in train])
    target=np.array([2*r['label']-1 for r in train])
    weights=np.where(target!=0,.5/(target!=0).sum(),.5/(target==0).sum())
    coefficients=np.linalg.lstsq(design*np.sqrt(weights[:,None]),target*np.sqrt(weights),rcond=None)[0]
    outcome={}
    for role in ('train','validation'):
        ordered=[r for r in pairs if r['role']==role and r['label']!=.5]
        correct=sum((coefficients[generators.index(by_uid[r['a']]['generator'])]
                     - coefficients[generators.index(by_uid[r['b']]['generator'])])*(2*r['label']-1)>0 for r in ordered)
        outcome[role]={'correct':int(correct),'ordered':len(ordered)}
    return {'status':'post-hoc descriptive confound check, no model inference',
            'fit':'train-only ordered/tied balanced least-squares, generator identity only',
            'coefficients':dict(zip(generators,coefficients.tolist())), 'counts':outcome,
            'limitation':'rules out this simple generator-bias explanation only, not other appearance confounds'}


def verify_heads(root):
    """Re-load saved heads and replay all 1620 learned scores from fixed tokens."""
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from dynamic_degree.learned_probe import MotionProbe
    from .vjepa_motion_probe import gpu_precheck

    start=time.monotonic()
    gpu=gpu_precheck()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=Path(root)
    source=rows(root/'selection/sources.jsonl')
    predictions={(r['arm'],r['video_uid']):r for r in rows(root/'training/scores.jsonl')}
    features={(r['video_uid'],r['view']):r for path in (root/'features').glob('shard-*/features.jsonl') for r in rows(path)}
    summary=json.loads((root/'training/summary.json').read_text())
    head=json.loads((root/'training/head.json').read_text())['architecture']
    models={}
    for arm in ('natural_only','joint'):
        path=root/'training'/f'{arm}.pt'
        assert sha(path)==summary['head_sha256'][arm]
        model=MotionProbe(head['input_dim'],head['hidden_dim'],head['mlp_dim'])
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        models[arm]=model.to('cuda:0').eval().requires_grad_(False)
    max_error, checked=0.,0
    for role in ('train','validation'):
        cohort=[r for r in source if r['role']==role]
        data=np.empty((len(cohort)*3,4608,768),dtype=np.float16)
        for i,r in enumerate(cohort):
            for j,view in enumerate(('base','alternating','aperiodic')):
                record=features[r['video_uid'],view]
                assert sha(record['feature_path'])==record['feature_sha256']
                data[3*i+j]=np.load(record['feature_path'],allow_pickle=False)
        x=torch.from_numpy(data).to('cuda:0',torch.float32)
        for arm,model in models.items():
            with torch.inference_mode():
                values=model(x).reshape(len(cohort),3).cpu().numpy()
            target=np.array([predictions[arm,r['video_uid']]['latent'] for r in cohort])
            max_error=max(max_error,float(abs(values-target).max()))
            checked+=values.size
        del x,data
    assert checked==1620 and max_error<=1e-6
    return {'status':'passed','learned_scores_checked':checked,'max_latent_error':max_error,
            'exact':max_error==0,'head_state_load':'weights_only=True, strict=True',
            'encoder_calls':0,'gpu':gpu,'wall_seconds':time.monotonic()-start,
            'auditor_sha256':sha(__file__),'summary_sha256':sha(root/'training/summary.json')}


def audit(root):
    root = Path(root)
    source = rows(root / 'selection/sources.jsonl')
    pairs = rows(root / 'selection/pairs.jsonl')
    receipt = json.loads((root / 'selection/selection.json').read_text())
    summary = json.loads((root / 'training/summary.json').read_text())
    scores = rows(root / 'training/scores.jsonl')
    assert sha(root / 'selection/sources.jsonl') == receipt['sources_sha256']
    assert sha(root / 'selection/pairs.jsonl') == receipt['pairs_sha256']
    assert sha(root / 'training/scores.jsonl') == summary['scores_sha256']
    relevant = [r for r in source if r['role'] in ('train', 'validation')]
    index = {(r['arm'], r['video_uid']): r for r in scores}
    assert len(scores) == len(index) == 810
    assert set(index) == {(arm, r['video_uid']) for arm in ('joint','natural_only','untrained') for r in relevant}
    maximum_error = 0.
    prompt_records = defaultdict(dict)
    for role in ('train', 'validation'):
        cohort = [r for r in relevant if r['role'] == role]
        role_pairs = [r for r in pairs if r['role'] == role]
        ordered = [r for r in role_pairs if r['label'] != .5]
        tied = [r for r in role_pairs if r['label'] == .5]
        for arm in ('joint','natural_only','untrained'):
            q = np.array([index[arm,r['video_uid']]['latent'] for r in cohort])
            s = 1/(1+np.exp(-q))
            assert np.isfinite(q).all()
            maximum_error = max(maximum_error,float(abs(s-np.array([index[arm,r['video_uid']]['relative_sigmoid'] for r in cohort])).max()))
            margins = np.array([(np.array(index[arm,r['a']]['latent'])-index[arm,r['b']]['latent'])*(2*r['label']-1) for r in ordered])
            gaps = np.array([np.array(index[arm,r['a']]['latent'])-index[arm,r['b']]['latent'] for r in tied])
            ref = summary['evaluations'][role][arm]
            assert (margins>0).sum(axis=0).tolist() == ref['ordered_correct_by_view']
            assert np.allclose(margins.mean(axis=0),ref['mean_signed_ordered_margin_by_view'],atol=1e-12,rtol=0)
            assert np.allclose(abs(gaps).mean(axis=0),ref['mean_absolute_tie_gap_by_view'],atol=1e-12,rtol=0)
            for name, values in [('latent',q),('relative_sigmoid',s)]:
                delta = values[:,1:]-values[:,:1]
                checks = {'base_mean': values[:,0].mean(), 'cf_mean': values[:,1:].mean(),
                          'signed_change': delta.mean(), 'mean_absolute_change': abs(delta).mean(),
                          'base_std': values[:,0].std(), 'base_min': values[:,0].min(),
                          'base_max': values[:,0].max(), 'max_absolute_change': abs(delta).max(),
                          'p95_absolute_change': np.quantile(abs(delta),.95)}
                for key,value in checks.items():
                    error=abs(float(value)-ref[name][key]); maximum_error=max(maximum_error,error)
                    assert error<1e-12
            if role == 'validation':
                for prompt in sorted({r['prompt_id'] for r in cohort}):
                    mask = [r['prompt_id']==prompt for r in cohort]
                    omask = [r['prompt_id']==prompt for r in ordered]
                    delta = q[mask,1:]-q[mask,:1]
                    display_delta = s[mask,1:]-s[mask,:1]
                    prompt_records[prompt][arm] = {'ordered': sum(omask), 'correct': int((margins[omask,0]>0).sum()),
                        'latent_abs_change': float(abs(delta).mean()), 'sigmoid_change': float(display_delta.mean()),
                        'sigmoid_abs_change': float(abs(display_delta).mean())}
    prompts = sorted(prompt_records)
    rng = np.random.default_rng(20260925)
    distributions = defaultdict(list)
    for _ in range(2000):
        chosen = [prompts[i] for i in rng.integers(len(prompts),size=len(prompts))]
        for arm in ('natural_only','joint'):
            records = [prompt_records[p][arm] for p in chosen]
            count = sum(r['ordered'] for r in records)
            if count:
                distributions[arm+'.ordered_accuracy'].append(sum(r['correct'] for r in records)/count)
            for key in ('latent_abs_change','sigmoid_change','sigmoid_abs_change'):
                distributions[arm+'.'+key].append(float(np.mean([r[key] for r in records])))
        count = sum(prompt_records[p]['joint']['ordered'] for p in chosen)
        if count:
            distributions['joint_minus_natural.ordered_accuracy'].append(sum(
                prompt_records[p]['joint']['correct']-prompt_records[p]['natural_only']['correct'] for p in chosen)/count)
        distributions['joint_minus_natural.latent_abs_change'].append(float(np.mean([
            prompt_records[p]['joint']['latent_abs_change']-prompt_records[p]['natural_only']['latent_abs_change'] for p in chosen])))
    features = []
    for p in sorted((root/'features').glob('shard-*/features.jsonl')):
        completion = json.loads((p.parent/'completion.json').read_text())
        assert sha(p) == completion['ledger_sha256'] and completion['failed_sources']==0
        features.extend(rows(p))
    assert len(features)==810 and len({(r['video_uid'],r['view']) for r in features})==810
    base=[r for r in features if r['view']=='base']
    digest_roles=defaultdict(set)
    for r in base:
        digest_roles[r['video_sha256']].add(r['role'])
    assert not any(len(v)>1 for v in digest_roles.values())
    assert {r['video_uid'] for r in base}=={r['video_uid'] for r in relevant}
    return {'status':'passed', 'maximum_numeric_error':maximum_error,
        'sources':len(base), 'features':len(features), 'prediction_rows':len(scores), 'score_values':len(scores)*3,
        'cross_role_identical_video_count':0, 'config_sha256':receipt['config_sha256'],
        'summary_sha256':sha(root/'training/summary.json'), 'auditor_sha256':sha(__file__),
        'generator_confound':generator_confound(source,pairs),
        'per_prompt':dict(prompt_records),
        'validation_prompt_cluster_bootstrap':{'prompts':len(prompts),'replicates':2000,'seed':20260925,
            'percentile95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in distributions.items()},
            'valid_replicates':{k:len(v) for k,v in distributions.items()},
            'limitation':'4 exposed development prompts only; not formal test confidence'},
        'geometry_flag_counts':dict(Counter(f"{r['view']}:{r['geometry']['jacobian_below_legacy_half']}" for r in features if r['view']!='base'))}


def audit_validation(root,human_source):
    """Independent aggregation/Origin-formula audit of the frozen real-video run."""
    root=Path(root)
    config_path=root/'code/configs/dynamic-static-jitter/vjepa-validation-v1.json'
    config=json.loads(config_path.read_text())
    summary=json.loads((root/'analysis/summary.json').read_text())
    inputs=rows(root/'inputs/inputs.jsonl'); ledger={r['evaluation_id']:r for r in inputs}
    assert len(inputs)==len(ledger)==848
    assert sha(root/'inputs/inputs.jsonl')==summary['input_sha256']
    assert sha(config_path)==summary['config_sha256']
    outputs={}
    for backend in ('origin','vjepa'):
        outputs[backend]={}
        for directory in sorted((root/'scores'/backend).glob('shard-*')):
            completion=json.loads((directory/'completion.json').read_text())
            provenance=json.loads((directory/'provenance.json').read_text())
            assert completion['status']=='finished' and completion['failed']==0
            assert sha(directory/'scores.jsonl')==completion['scores_sha256']
            assert provenance['config_sha256']==sha(config_path)
            for name,digest in provenance['code'].items():
                assert sha(root/'code'/name)==digest
            if backend=='vjepa':
                assert provenance['model']['heads']==config['heads']
                assert provenance['model']['pretest_loading_check']['feature_exact']
            else:
                assert provenance['model']['upstream']['sha']==config['origin_upstream_commit']
                assert not provenance['model']['upstream']['dirty']
            for r in rows(directory/'scores.jsonl'):
                key=r['evaluation_id']
                assert key not in outputs[backend] and r['status']=='ok'
                assert r['input_sha256']==ledger[key]['sha256']
                for name in ('base_id','cohort','family','seed','prompt_id','generator'):
                    assert r[name]==ledger[key][name]
                outputs[backend][key]=r
        assert set(outputs[backend])==set(ledger)
    max_error=0.
    for key,r in outputs['origin'].items():
        d=r['diagnostics']; sample=d['sampling']; raw=d['official']
        scores=np.array(raw['raw_flow_top5_mean'])
        threshold=6*min(sample['frame_shape'])/256
        count=round(4*sample['sampled_frame_count']/16)
        assert len(scores)==15 and np.isfinite(scores).all()
        assert sample['sampled_source_frame_indices']==list(range(16)) and sample['sampling_interval']==1
        assert sample['source_fps']==8 and raw['official_threshold']==threshold and raw['official_count_num']==count
        assert float((scores>threshold).sum()>=count)==r['score']
        for arm in ('joint','natural_only'):
            v=outputs['vjepa'][key][arm]
            error=abs(1/(1+np.exp(-v['latent']))-v['score']); assert error<1e-12
            max_error=max(max_error,error)
    def score(row,backend):
        key=row['evaluation_id']
        if backend=='origin': return outputs['origin'][key]['score']
        if backend=='joint_latent': return outputs['vjepa'][key]['joint']['latent']
        return outputs['vjepa'][key][backend]['score']
    controls=0
    for cohort,n in [('dev32',32),('holdout',90)]:
        bases=sorted({r['base_id'] for r in inputs if r['cohort']==cohort and r['counterfactual_primary']})
        assert len(bases)==n
        selected=[]
        for uid in bases:
            source=[r for r in inputs if r['base_id']==uid]
            original=next(r for r in source if r['family']=='original')
            control=next(r for r in source if r['family']=='encoding_control')
            cf=[next(r for r in source if r['seed']==seed) for seed in (1701,2904)]
            assert outputs['origin'][original['evaluation_id']]['diagnostics']==outputs['origin'][control['evaluation_id']]['diagnostics']
            assert outputs['vjepa'][original['evaluation_id']]['feature_sha256']==outputs['vjepa'][control['evaluation_id']]['feature_sha256']
            for arm in ('joint','natural_only'):
                assert outputs['vjepa'][original['evaluation_id']][arm]==outputs['vjepa'][control['evaluation_id']][arm]
            controls+=1; selected.append((original,cf))
        for backend in ('origin','joint','natural_only','joint_latent'):
            base=np.array([score(a,backend) for a,_ in selected])
            cf=np.array([[score(r,backend) for r in b] for _,b in selected])
            change=cf-base[:,None]
            reference=summary['counterfactual'][cohort][backend]
            for name,value in [('base_mean',base.mean()),('cf_mean',cf.mean()),('delta',change.mean()),
                               ('mae',abs(change).mean()),('base_std',base.std()),('largest_drop',change.min()),
                               ('maximum_absolute_change',abs(change).max())]:
                error=abs(float(value)-reference[name]); assert error<1e-12
                max_error=max(max_error,error)
    assert controls==122
    natural={r['base_id']:r for r in inputs if r['cohort']=='holdout' and r['family']=='original'}
    exported=rows(root/'analysis/human_pairs.jsonl')
    assert sha(human_source)==config['human_pairs_sha256']
    actual=[]
    with Path(human_source).open() as handle:
        for r in csv.DictReader(handle):
            if r['dimension']!='dynamics_degree' or r['split']!='test': continue
            if r['video_a_uid'] not in natural or r['video_b_uid'] not in natural: continue
            actual.append(dict(a=r['video_a_uid'],b=r['video_b_uid'],prompt_id=r['prompt_id'],label=float(r['human_label'])))
    assert actual==exported and len(actual)==450
    ordered=[r for r in actual if r['label']!=.5]
    for backend in ('origin','joint','natural_only'):
        signed=np.array([(score(natural[r['a']],backend)-score(natural[r['b']],backend))*(2*r['label']-1) for r in ordered])
        reference=summary['natural_preferences'][backend]
        assert (int((signed>0).sum()),int((signed==0).sum()),int((signed<0).sum()))==(
            reference['strict_correct'],reference['predicted_ties'],reference['incorrect'])
        assert abs(((signed>0).sum()+.5*(signed==0).sum())/len(signed)-reference['concordance'])<1e-12
    legacy_root=root.parent/'dev32-cotracker3-boolean-v1'
    legacy={r['candidate_id']:r for p in legacy_root.glob('shard-*/scores.jsonl') for r in rows(p)}
    max_flow_error=0.; compared=0
    for r in inputs:
        if r['cohort']!='dev32': continue
        old=legacy[r['candidate_id']]; new=outputs['origin'][r['evaluation_id']]
        assert old['input_sha256']==new['input_sha256'] and old['origin']['score']==new['score']
        maximum=float(abs(np.array(old['origin']['diagnostics']['official']['raw_flow_top5_mean'])
                          -new['diagnostics']['official']['raw_flow_top5_mean']).max())
        max_flow_error=max(max_flow_error,maximum); compared+=1
    assert compared==128
    return {'status':'passed','origin_formula_verified':848,'sigmoid_scores_verified':1696,
        'encoding_controls_exact':controls,'maximum_statistic_error':max_error,
        'legacy_origin_scores_unchanged':compared,'legacy_raw_flow_max_abs_error':max_flow_error,
        'human_test_pairs_verified':450,'ordered_pairs':len(ordered),
        'ordered_prompts':len({r['prompt_id'] for r in ordered}),
        'counterfactual_statistics_recomputed':True,'snapshot_file_hashes_verified':True,
        'summary_sha256':sha(root/'analysis/summary.json'),'auditor_sha256':sha(__file__)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--verify-heads',action='store_true',help='H200: re-load saved small heads only, no encoder calls')
    parser.add_argument('--validation',action='store_true',help='audit the real VBench frozen validation outputs')
    parser.add_argument('--human-source',help='frozen pairwise CSV, required with --validation')
    args=parser.parse_args()
    output=Path(args.output).resolve()
    repo=Path(__file__).resolve().parents[2]
    if output==repo or any(output.is_relative_to(repo/p) for p in ('data','results','splits','runs')):
        raise ValueError('cannot write a frozen tree')
    if args.validation and (args.verify_heads or not args.human_source):
        parser.error('--validation requires --human-source and excludes --verify-heads')
    result=audit_validation(args.root,args.human_source) if args.validation else verify_heads(args.root) if args.verify_heads else audit(args.root)
    with output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
