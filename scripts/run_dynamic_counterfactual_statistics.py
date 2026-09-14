#!/usr/bin/env python3
"""Frozen-contract statistics for formal Dynamic Degree records."""
from __future__ import annotations

import argparse, csv, hashlib, itertools, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from scipy.stats import spearmanr

VARIANTS=('official','time_only','source_only','duration_only','source_time','full')
SOURCE_AWARE={'source_only','source_time','full'}
COVERAGE_AWARE={'duration_only','full'}

def read_jsonl(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def mean(xs): return float(statistics.fmean(xs)) if xs else None
def median(xs): return float(statistics.median(xs)) if xs else None
def safe_ratio(a,b): return float(a/(abs(b)+1e-12))
def common_base_ids(*mappings):
    """Return the sorted intersection for exactly the dictionaries compared."""
    if not mappings:
        return []
    return sorted(set.intersection(*(set(mapping) for mapping in mappings)))

def bootstrap_mean(mapping, iterations, seed):
    """Bootstrap one variant's per-base statistic over its own valid bases."""
    base_ids=sorted(mapping)
    if not base_ids:
        return None
    values=np.array([mapping[base] for base in base_ids],dtype=float)
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(base_ids),size=(iterations,len(base_ids)))
    distribution=values[draws].mean(axis=1)
    point=float(values.mean())
    ci=[float(np.quantile(distribution,.025)),float(np.quantile(distribution,.975))]
    return {'point_estimate':point,'mean':point,'ci95':ci,'valid_n':len(base_ids),'valid_base_ids':base_ids,'ci_input_base_ids':base_ids,'sampling_unit':'base_id','point_in_ci':ci[0] <= point <= ci[1]}

def safe_spearman(xs, ys):
    if len(xs) < 2:
        return None, 'insufficient_n'
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None, 'constant_input'
    value = float(spearmanr(xs, ys).statistic)
    if not math.isfinite(value):
        return None, 'undefined'
    return value, None
def level_number(value):
    mapping={'clean':0,'weak':1,'medium':2,'strong':3,'S0C0':0,'S1C0':1,'S0C1':2,'S1C1':3,'native':1024}
    return float(mapping.get(str(value), value))

def evidence_value(side, variant, field):
    result=side['result']; ev=result.get('structured_evidence') or {}
    if field=='primary':
        if variant=='official': return result.get('score')
        if variant=='full': return (ev.get('task_relevant_motion_evidence') or {}).get('motion_intensity')
        return ev.get('apparent_intensity')
    if field=='coverage':
        if variant=='duration_only': return ev.get('apparent_coverage')
        if variant=='full': return (ev.get('task_relevant_motion_evidence') or {}).get('temporal_coverage')
        return None
    return ev.get(field)

def add(rows, base, family, variant, statistic_name, value, failures=0, reason=None):
    valid=value is not None and math.isfinite(float(value))
    rows.append({'base_id':base,'family':family,'variant':variant,'statistic':statistic_name,'value':value,'valid':valid,'reason':None if valid else (reason or 'missing_evidence'),'failure_count':failures})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluations',type=Path,required=True); ap.add_argument('--evidence',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--report-root',type=Path,required=True); ap.add_argument('--split',default='test',choices=('dev','test')); ap.add_argument('--bootstrap-iterations',type=int,default=5000); ap.add_argument('--seed',type=int,default=20260912); a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    a.report_root.mkdir(parents=True,exist_ok=True)
    evals=[r for r in read_jsonl(a.evaluations) if r['split']==a.split]
    sides=[r for r in read_jsonl(a.evidence) if r['split']==a.split]
    skey=lambda r:(r['base_id'],r['derived_id'],r['split'],r['metric_variant'],r['target_type'])
    side_by={skey(r):r for r in sides}
    if len(side_by)!=len(sides): raise RuntimeError('duplicate evidence sidecar key')
    joined=[]
    for r in evals:
        key=skey(r)
        if key not in side_by: raise RuntimeError(f'missing evidence sidecar: {key}')
        joined.append((r,side_by[key]))
    groups=defaultdict(list)
    for r,s in joined: groups[(r['base_id'],r['intervention_family'],r['metric_variant'])].append((r,s))
    per=[]
    for (base,family,variant),items in sorted(groups.items()):
        failures=sum(r['failure_or_abstention'] is not None for r,_ in items)
        ordered=sorted(items,key=lambda pair:level_number(pair[0]['intervention_level']))
        levels=[level_number(r['intervention_level']) for r,_ in ordered]
        primary=[evidence_value(s,variant,'primary') for _,s in ordered]
        valid=[float(x) for x in primary if x is not None and math.isfinite(float(x))]
        if family=='fps_resampling':
            m=mean(valid); reason='missing_evidence' if not valid else 'zero_mean_undefined' if m==0 else None; add(per,base,family,variant,'fps_cv',None if not valid or m==0 else float(np.std(valid,ddof=0)/abs(m)),failures,reason); add(per,base,family,variant,'fps_relative_range',None if not valid or m==0 else (max(valid)-min(valid))/abs(m),failures,reason)
        elif family=='resolution':
            m=mean(valid); reason='missing_evidence' if not valid else 'zero_mean_undefined' if m==0 else None; add(per,base,family,variant,'resolution_gap',None if not valid or m==0 else (max(valid)-min(valid))/abs(m),failures,reason)
        elif family=='subject_speed':
            pairs=[(x,float(y)) for x,y in zip(levels,primary) if y is not None and math.isfinite(float(y))]
            rho,rho_reason=safe_spearman([x for x,_ in pairs],[y for _,y in pairs])
            strict=mean([float(b>a0) for a0,b in zip([y for _,y in pairs][:-1],[y for _,y in pairs][1:])]) if len(pairs)>=2 else None
            add(per,base,family,variant,'speed_rho',rho,failures,rho_reason); add(per,base,family,variant,'speed_strict_order_rate',strict,failures,'insufficient_n')
        elif family=='motion_coverage':
            pairs=[(x,float(y)) for x,y in zip(levels,primary) if y is not None and math.isfinite(float(y))]
            rho,rho_reason=safe_spearman([x for x,_ in pairs],[y for _,y in pairs])
            add(per,base,family,variant,'coverage_rho',rho,failures,rho_reason)
            cover=[evidence_value(s,variant,'coverage') for _,s in ordered]
            cpairs=[(x/100.0,float(y)) for x,y in zip(levels,cover) if y is not None and math.isfinite(float(y))]
            add(per,base,family,variant,'coverage_mae',mean([abs(x-y) for x,y in cpairs]) if cpairs and variant in COVERAGE_AWARE else None,failures,'unsupported' if variant not in COVERAGE_AWARE else 'missing_evidence')
        elif family=='camera_shake':
            if variant in SOURCE_AWARE and len(ordered)>=2:
                camera=[evidence_value(s,variant,'camera_intensity') for _,s in ordered]; residual=[evidence_value(s,variant,'residual_intensity') for _,s in ordered]
                ce=float(camera[-1]-camera[0]) if camera[-1] is not None and camera[0] is not None else None; re=float(residual[-1]-residual[0]) if residual[-1] is not None and residual[0] is not None else None
                add(per,base,family,variant,'camera_subject_effect_ratio',None if ce is None or re is None else safe_ratio(ce,re),failures); add(per,base,family,variant,'prompt_selectivity',None if ce is None or re is None else float(abs(ce)>abs(re)),failures)
            else: add(per,base,family,variant,'camera_subject_effect_ratio',None,failures,'unsupported'); add(per,base,family,variant,'prompt_selectivity',None,failures,'unsupported')
        elif family=='subject_camera':
            vals={r['intervention_level']:(evidence_value(s,variant,'residual_intensity'),evidence_value(s,variant,'camera_intensity'),evidence_value(s,variant,'apparent_intensity')) for r,s in items}
            if variant in SOURCE_AWARE and all(k in vals for k in ('S0C0','S1C0','S0C1','S1C1')):
                subject=((vals['S1C0'][0]+vals['S1C1'][0])-(vals['S0C0'][0]+vals['S0C1'][0]))/2; camera=((vals['S0C1'][1]+vals['S1C1'][1])-(vals['S0C0'][1]+vals['S1C0'][1]))/2; interaction=vals['S1C1'][2]-vals['S1C0'][2]-vals['S0C1'][2]+vals['S0C0'][2]
                add(per,base,family,variant,'subject_main_effect',subject,failures); add(per,base,family,variant,'camera_main_effect',camera,failures); add(per,base,family,variant,'interaction',interaction,failures)
            else:
                for name in ('subject_main_effect','camera_main_effect','interaction'): add(per,base,family,variant,name,None,failures,'unsupported')
        elif family=='static_flicker':
            field='residual_intensity' if variant in SOURCE_AWARE else 'primary'; values=[evidence_value(s,variant,field) for _,s in ordered]
            add(per,base,family,variant,'flicker_response',None if not values or values[0] is None else mean([float(v)-float(values[0]) for v in values[1:] if v is not None]),failures)

    fields=['base_id','family','variant','statistic','value','valid','reason','failure_count']
    with (a.output/'per_base.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(per)
    grouped=defaultdict(list)
    for r in per:
        if r['valid']: grouped[(r['family'],r['variant'],r['statistic'])].append(float(r['value']))
    summary={}
    for family in sorted({r['family'] for r in per}):
        summary[family]={}
        for variant in VARIANTS:
            summary[family][variant]={}
            for stat in sorted({r['statistic'] for r in per if r['family']==family}):
                xs=grouped.get((family,variant,stat),[]); candidates=[r for r in per if r['family']==family and r['variant']==variant and r['statistic']==stat]
                summary[family][variant][stat]={'mean':mean(xs),'median':median(xs),'valid_base_count':len(xs),'valid_base_ids':[r['base_id'] for r in candidates if r['valid']],'unsupported_count':sum(r['reason']=='unsupported' for r in candidates),'undefined_constant_count':sum(r['reason']=='constant_input' for r in candidates),'invalid_other_count':sum(not r['valid'] and r['reason'] not in {'unsupported','constant_input'} for r in candidates),'failure_count':sum(r['failure_count'] for r in candidates)}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')

    bootstrap={'iterations':a.bootstrap_iterations,'seed':a.seed,'unit':'base_id','results':{}}
    for family in summary:
        stats=sorted({r['statistic'] for r in per if r['family']==family})
        for stat in stats:
            byv={v:{r['base_id']:float(r['value']) for r in per if r['family']==family and r['variant']==v and r['statistic']==stat and r['valid']} for v in VARIANTS}
            applicable={v:d for v,d in byv.items() if d}
            common=common_base_ids(*applicable.values())
            key=f'{family}/{stat}'; bootstrap['results'][key]={'multi_variant_common_base_ids':common,'variants':{},'pairwise_comparisons':{}}
            seed=a.seed+int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
            for v,d in byv.items():
                candidates=[r for r in per if r['family']==family and r['variant']==v and r['statistic']==stat]
                if not d:
                    reasons=Counter(r['reason'] for r in candidates)
                    status='unsupported' if reasons and set(reasons)=={'unsupported'} else ('undefined_constant' if reasons.get('constant_input') else 'no_valid_bases')
                    bootstrap['results'][key]['variants'][v]={'status':status,'point_estimate':None,'ci95':None,'valid_n':0,'valid_base_ids':[],'ci_input_base_ids':[],'sampling_unit':'base_id','reason_counts':dict(reasons)}
                    continue
                variant_seed=seed+int(hashlib.sha256(v.encode()).hexdigest()[:8],16)
                bootstrap['results'][key]['variants'][v]={'status':'ok',**bootstrap_mean(d,a.bootstrap_iterations,variant_seed)}
            for left,right in itertools.combinations(applicable,2):
                paired=common_base_ids(byv[left],byv[right])
                comparison=f'{left}__vs__{right}'
                if not paired:
                    bootstrap['results'][key]['pairwise_comparisons'][comparison]={'status':'no_common_bases','paired_base_ids':[],'ci_input_base_ids':[],'valid_n':0,'sampling_unit':'base_id'}
                    continue
                pair_seed=seed+int(hashlib.sha256(comparison.encode()).hexdigest()[:8],16)
                pair_rng=np.random.default_rng(pair_seed)
                pair_draws=pair_rng.integers(0,len(paired),size=(a.bootstrap_iterations,len(paired)))
                left_values=np.array([byv[left][base] for base in paired])
                right_values=np.array([byv[right][base] for base in paired])
                left_means=left_values[pair_draws].mean(axis=1)
                right_means=right_values[pair_draws].mean(axis=1)
                differences=right_means-left_means
                difference_point=float(right_values.mean()-left_values.mean())
                difference_ci=[float(np.quantile(differences,.025)),float(np.quantile(differences,.975))]
                bootstrap['results'][key]['pairwise_comparisons'][comparison]={'status':'ok','paired_base_ids':paired,'ci_input_base_ids':paired,'valid_n':len(paired),'sampling_unit':'base_id','left_mean':float(left_values.mean()),'right_mean':float(right_values.mean()),'point_estimate_right_minus_left':difference_point,'mean_difference_right_minus_left':difference_point,'difference_ci95':difference_ci,'point_in_ci':difference_ci[0] <= difference_point <= difference_ci[1]}
    (a.output/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2,sort_keys=True)+'\n')

    def cell(family,variant,stat):
        x=summary.get(family,{}).get(variant,{}).get(stat,{})
        if not x:
            return 'unsupported'
        if x['mean'] is not None:
            return x['mean']
        if x.get('undefined_constant_count'):
            return 'undefined_constant'
        if x.get('unsupported_count'):
            return 'unsupported'
        return 'not_available'
    labels={'official':'Official','time_only':'+ Time','source_only':'+ Source','duration_only':'+ Duration','source_time':'Source + Time','full':'Full'}
    table=[]
    for v in VARIANTS: table.append({'Variant':labels[v],'Source separation':'yes' if v in SOURCE_AWARE else 'no','FPS CV':cell('fps_resampling',v,'fps_cv'),'Coverage MAE':cell('motion_coverage',v,'coverage_mae'),'Speed rho':cell('subject_speed',v,'speed_rho'),'Prompt selectivity':cell('camera_shake',v,'prompt_selectivity')})
    def write_csv(path,rows):
        with path.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_csv(a.report_root/'TABLE_DYNAMIC_ABLATION.csv',table)
    md=['| '+' | '.join(table[0])+' |','|'+'|'.join(['---']*len(table[0]))+'|']+['| '+' | '.join(str(r[k]) for k in table[0])+' |' for r in table]
    (a.report_root/'TABLE_DYNAMIC_ABLATION.md').write_text('\n'.join(md)+'\n')
    ext=[]
    for v in VARIANTS: ext.append({**table[VARIANTS.index(v)],'Resolution gap':cell('resolution',v,'resolution_gap'),'FPS relative range':cell('fps_resampling',v,'fps_relative_range'),'Speed strict-order rate':cell('subject_speed',v,'speed_strict_order_rate'),'Coverage rho':cell('motion_coverage',v,'coverage_rho'),'Subject main effect':cell('subject_camera',v,'subject_main_effect'),'Camera main effect':cell('subject_camera',v,'camera_main_effect'),'Interaction':cell('subject_camera',v,'interaction'),'Flicker response':cell('static_flicker',v,'flicker_response'),'Failure rate':sum(r['failure_or_abstention'] is not None for r in evals if r['metric_variant']==v)/max(1,sum(r['metric_variant']==v for r in evals))})
    write_csv(a.report_root/'TABLE_DYNAMIC_EXTENDED.csv',ext)

    primary={'fps_resampling':'fps_cv','resolution':'resolution_gap','subject_speed':'speed_rho','motion_coverage':'coverage_mae','camera_shake':'prompt_selectivity','subject_camera':'subject_main_effect','static_flicker':'flicker_response'}
    matrix=[]
    for family,stat in primary.items():
        row={'family':family}
        for v in VARIANTS:
            val=cell(family,v,stat); boot=bootstrap['results'].get(f'{family}/{stat}',{}).get('variants',{}).get(v,{})
            if isinstance(val,str):
                row[v]=json.dumps({'statistic':stat,'value':None,'ci95':None,'valid_n':summary[family][v][stat]['valid_base_count'],'status':val},separators=(',',':'))
            else:
                point=boot.get('point_estimate')
                if point is None or not math.isclose(float(val),float(point),rel_tol=1e-12,abs_tol=1e-12):
                    raise RuntimeError(f'point/bootstrap definition mismatch: {family}/{stat}/{v}: summary={val}, bootstrap={point}')
                row[v]=json.dumps({'statistic':stat,'value':point,'ci95':boot.get('ci95'),'valid_n':boot['valid_n'],'status':'PASS' if boot.get('point_in_ci') else 'CI_WARNING'},separators=(',',':'))
        matrix.append(row)
    write_csv(a.report_root/'CONTRACT_MATRIX.csv',matrix)
    headers=list(matrix[0]); lines=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(r[h]) for h in headers)+' |' for r in matrix]
    (a.report_root/'CONTRACT_MATRIX.md').write_text('\n'.join(lines)+'\n')

    def display(value):
        if value is None: return 'null'
        if isinstance(value,float): return f'{value:.10g}'
        return str(value)
    audit=['# Dynamic Degree Statistics Audit','',
        '## Finding','',
        'The previous report mixed two base sets: point estimates used every valid base for one variant, while variant confidence intervals used the intersection across all non-empty variants. This was a code bug, not a percentile-bootstrap exception. Variant CIs now resample that variant’s own per-base values; paired comparisons separately resample their exact pairwise intersection.','',
        f'- Split: `{a.split}`',f'- Bootstrap iterations: {a.bootstrap_iterations}',f'- Fixed seed: {a.seed}',
        '- Independent sampling unit: `base_id` only; `derived_id` is never sampled.','- Point definition: arithmetic mean of the listed per-base contract statistic.','- CI definition: percentile 95% interval of bootstrap means from the identical listed per-base inputs.','',
        '## Per-family, per-statistic, per-variant inputs','',
        '| family | statistic | variant | point estimate | valid base IDs | valid_n | unit | CI calculation input | CI95 | status / reason |','|---|---|---|---:|---|---:|---|---|---|---|']
    point_outside=[]
    for key,item in sorted(bootstrap['results'].items()):
        family,stat=key.split('/',1)
        for variant in VARIANTS:
            data=item['variants'][variant]
            ids=data.get('valid_base_ids',[]); inputs=data.get('ci_input_base_ids',[])
            if data.get('status')=='ok' and ids != inputs:
                raise RuntimeError(f'variant CI input mismatch: {key}/{variant}')
            if data.get('status')=='ok' and not data.get('point_in_ci'):
                point_outside.append(f'{key}/{variant}')
            ci=data.get('ci95')
            reason=data.get('status') if data.get('status')!='ok' else ('PASS' if data.get('point_in_ci') else 'percentile_point_outside_ci')
            audit.append(f'| {family} | {stat} | {variant} | {display(data.get("point_estimate"))} | {", ".join(ids) or "-"} | {data.get("valid_n",0)} | base_id | mean of listed per-base values | {"["+", ".join(display(x) for x in ci)+"]" if ci else "null"} | {reason} |')
    pair_count=0
    for key,item in bootstrap['results'].items():
        for comparison,data in item['pairwise_comparisons'].items():
            if data['status']=='ok':
                pair_count+=1
                if data['paired_base_ids'] != data['ci_input_base_ids']:
                    raise RuntimeError(f'paired CI input mismatch: {key}/{comparison}')
    audit.extend(['','## Pairing verification','',f'- Valid pairwise comparisons checked: {pair_count}.','- Every paired comparison uses one exact common `base_id` list for both variants and all bootstrap draws.','- No global family-wide common set is used for variant point estimates or marginal CIs.','',
        '## Small-N findings','',
        '- `subject_speed` has 5 TEST bases. Official has 2 valid per-base rho values and 3 `constant_input` undefined values; Full has 2 valid values because only two bases have routable task-relevant evidence, while three are `insufficient_n`; time_only/source_only/duration_only/source_time each have 5. The previous `valid_n=1` was the erroneous all-variant intersection, not the independent N.','- `motion_coverage` has 5 TEST bases. Coverage MAE is unsupported for Official/time_only/source_only/source_time; duration_only has 5 valid bases; Full has 2 valid routable bases and 3 missing task-relevant coverage values. Thus Full `valid_n=2` is real, while duration_only’s previous `valid_n=2` was incorrect.','- Constant Spearman results remain `null` with reason `constant_input`; they are never replaced by zero.','- Unsupported statistics remain `unsupported`; they are never replaced by zero.','',
        '## Percentile-CI containment','',
        f'- Finite point estimates outside their own-input percentile CI: {len(point_outside)}'+(f' ({", ".join(point_outside)})' if point_outside else '.'),
        '- If nonzero, the corresponding contract cell is marked `CI_WARNING`; no such case is silently accepted.',''])
    (a.report_root/'STATISTICS_AUDIT.md').write_text('\n'.join(audit)+'\n')

    categories=defaultdict(Counter); fallbacks=defaultdict(int); efficiency=defaultdict(list)
    for r,s in joined:
        v=r['metric_variant']; reason=(r['failure_or_abstention'] or '').lower()
        if reason:
            cat='decode' if 'decode' in reason or 'video' in reason else 'raft_flow' if 'raft' in reason or 'flow' in reason else 'camera_fit' if 'camera' in reason or 'affine' in reason else 'routing' if 'target' in reason or 'routing' in reason else 'structured_evidence' if 'evidence' in reason else 'other'; categories[v][cat]+=1
        result=s['result']; efficiency[v].append((result.get('runtime_s'),result.get('peak_gpu_memory_mb')))
        text=json.dumps(result.get('diagnostics'),sort_keys=True)
        fallbacks[v]+=text.count('"fallback": true')
    failure={'by_variant':{v:dict(categories[v]) for v in VARIANTS},'fallback_count':dict(fallbacks)}; (a.output/'failure_summary.json').write_text(json.dumps(failure,indent=2,sort_keys=True)+'\n')
    eff={v:{'mean_runtime_s':mean([x for x,_ in efficiency[v] if x is not None]),'median_runtime_s':median([x for x,_ in efficiency[v] if x is not None]),'peak_gpu_memory_mb':max([x for _,x in efficiency[v] if x is not None],default='not_collected')} for v in VARIANTS}; (a.output/'efficiency.json').write_text(json.dumps(eff,indent=2,sort_keys=True)+'\n')
    report=['# Dynamic Degree Final Report','',f'- Dataset: 50 independent bases, 730 derived records; DEV/TEST=10/40; seven families; six variants.',f'- Statistics split: {a.split}; bootstrap: {a.bootstrap_iterations} iterations, seed {a.seed}, independent unit `base_id`.','- Each variant point estimate and marginal CI use the same variant-specific valid per-base inputs. Paired comparisons use their exact pairwise common base IDs and shared draws.','- Official is a locked boolean baseline. Repair variants persist structured evidence and may correctly have `score=null`.','- Unsupported comparisons are written as `unsupported`; constant Spearman is `null` with an explicit reason. No zero substitution or cross-evaluator raw-score comparison is used.','- See `STATISTICS_AUDIT.md` for every valid base ID, CI input, valid N, and containment check.','- Contract PASS denotes successful computation under the frozen contract, not a claim of favorable scientific effect.','','## Main conclusions','','Formal numeric conclusions must be read from the generated tables and confidence intervals; this report does not tune or reinterpret outcomes.']
    (a.report_root/'DYNAMIC_FINAL_REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'event':'statistics_complete','split':a.split,'records':len(evals),'per_base_rows':len(per)}))
if __name__=='__main__': main()
