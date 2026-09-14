#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,os,sys
from collections import defaultdict,Counter
from pathlib import Path
R=Path('/root/vbench-audit'); D=Path('/root/autodl-tmp/vbench-audit-storage/datasets/e0_public'); O=Path('/root/autodl-tmp/vbench-audit-storage/scores/official/e0'); OUT=Path('/root/autodl-tmp/vbench-audit-storage/runs/official_dataset_compare_4dims'); U=Path('/root/vbench1')
DM={'dynamic_degree':'dynamics_degree','spatial_relationship':'spatial_relationship','human_action':'human_action','subject_consistency':'subject_consistency'}; K=('video_uid','dimension','prompt_id','generator','relative_video_path'); S={'pending','succeeded_scalar','succeeded_structured','unsupported','failed'}
def append(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def load(p):
 if not p.exists():return []
 try:return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
 except Exception as e:raise RuntimeError(f'manifest corruption: {e}')
def read(p):
 with p.open(newline='',encoding='utf8') as f:return [dict(x) for x in csv.DictReader(f)]
def join(m,o):
 q=defaultdict(list)
 for x in o:q[tuple(x[k] for k in K)].append(x)
 z={}
 for x in m:
  a=q[tuple(x[k] for k in K)]
  if len(a)!=1:raise RuntimeError(f'Official strict join {len(a)} matches: {tuple(x[k] for k in K)}')
  z[x['video_uid']]=a[0]
 return z
def paths():sys.path[:0]=[str(R/x) for x in ['公共/audit-core/src','指标/dynamic-degree/src','指标/spatial_relationship/src','指标/human_action/src','指标/subject_consistency/src']]+[str(U)]
def spatial():
 z={}
 for x in json.loads((U/'vbench/VBench_full_info.json').read_text(encoding='utf8')):
  a=x.get('auxiliary_info',{}).get('spatial_relationship',{});a=a.get('spatial_relationship',a) if isinstance(a,dict) else {}
  if all(isinstance(a.get(k),str) and a[k].strip() for k in ('object_a','object_b','relationship')):z[x['prompt_en']]=a
 return z
def meta(x,sp):
 m={'video':Path(x['relative_video_path']).name,'prompt':x['prompt_id']}
 if x['dimension']=='spatial_relationship':
  if x['prompt_id'] not in sp:return None
  m['dimension_metadata']={'spatial_relationship':sp[x['prompt_id']]}
 if x["dimension"]=="human_action":
  prefix="A person is "
  if not x["prompt_id"].startswith(prefix): raise RuntimeError("Human Action prompt is not an explicit K400 action prompt")
  m["dimension_metadata"]={"human_action":{"target_action":x["prompt_id"][len(prefix):]}}
 return m
def runner(dim,dev):
 print(json.dumps({'event':'model_load_start','dimension':dim}),flush=True)
 if dim=='dynamic_degree':
  from dynamic_degree.models import RaftFlowModel
  from dynamic_degree.metric import evaluate_audit_batch,weight_path,upstream_path
  from dynamic_degree.diagnostics import DiagnosticsLevel
  m=RaftFlowModel(dev,weight_path(),upstream_path()); f=lambda v,x:evaluate_audit_batch([v],x,dev,weight_path(),DiagnosticsLevel.FULL,flow_model=m)
 elif dim=='spatial_relationship':
  from spatial_relationship.backends.vbench import OfficialGritDetector
  from spatial_relationship.metric import evaluate_audit_batch,weight_path,upstream_path
  from spatial_relationship.diagnostics import DiagnosticsLevel
  m=OfficialGritDetector(dev,weight_path(),upstream_path());f=lambda v,x:evaluate_audit_batch([v],x,dev,weight_path(),DiagnosticsLevel.FULL,detector=m)
 elif dim=='human_action':
  from human_action.backends.audit import AuditHumanActionEvaluator
  from human_action.backends.vbench import import_official_module
  from human_action.models import LockedUmtClassifier
  from human_action.metric import evaluate_audit_batch,load_categories,weight_path,upstream_path
  from human_action.diagnostics import DiagnosticsLevel
  a,_=import_official_module(upstream_path());m=AuditHumanActionEvaluator(LockedUmtClassifier(dev,weight_path(),a));c=load_categories(upstream_path());f=lambda v,x:evaluate_audit_batch([v],x,dev,weight_path(),c,DiagnosticsLevel.FULL,evaluator=m)
 else:
  from subject_consistency.models import OfficialDinoFeatureExtractor
  from subject_consistency.metric import build_dino_config,evaluate_batch,upstream_path
  c=build_dino_config();m=OfficialDinoFeatureExtractor(dev,c,upstream_path());f=lambda v,x:evaluate_batch('audit',[v],x,dev,c,extractor=m)
 print(json.dumps({'event':'model_load_complete','dimension':dim,'model_load_count':1}),flush=True);return f
def base(x):
 return {k:x[k] for k in K}|{'video_path':str(D/x['relative_video_path']),'model':x['generator'],'group_id':x['group_id'],'split':x['split'],'synthetic_id':x['video_uid'],'repair_variant':'audit','repair_status':'pending','repair_score':None,'repair_prediction':None,'failure_or_abstention':None}
def write_predictions(out,m,sp):
 latest={(x['video_uid'],x.get('repair_variant','audit')):x for x in load(out/'repair_results.jsonl')}
 fields=['video_uid','video_path','model','dimension','split','prompt_id','group_id','relative_video_path','explicit_target_or_metadata','repair_score','repair_status','error']
 data=[]
 for x in m:
  r=latest.get((x['video_uid'],'audit'),base(x)); metadata=meta(x,sp)
  data.append({'video_uid':x['video_uid'],'video_path':str(D/x['relative_video_path']),'model':x['generator'],'dimension':x['dimension'],'split':x['split'],'prompt_id':x['prompt_id'],'group_id':x['group_id'],'relative_video_path':x['relative_video_path'],'explicit_target_or_metadata':json.dumps(metadata.get('dimension_metadata') if metadata else None,ensure_ascii=False,sort_keys=True),'repair_score':r.get('repair_score'),'repair_status':r.get('repair_status','pending'),'error':r.get('failure_or_abstention')})
 with (out/'predictions.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data);f.flush();os.fsync(f.fileno())
def report(out,m,off):
 q={(x['video_uid'],x.get('repair_variant','audit')):x for x in load(out/'repair_results.jsonl')};p=[]
 for x in m:
  r=q.get((x['video_uid'],'audit'))
  if not r:continue
  os_=off[x['video_uid']].get('status','');rs=r['repair_status'];st='official_unsupported' if os_ in ('unsupported','official_unsupported') else 'repair_unsupported' if rs=='unsupported' else 'repair_failure' if rs=='failed' else 'incompatible_scale' if r['repair_score'] is None else 'comparable';sc=float(off[x['video_uid']]['score']) if off[x['video_uid']].get('score') else None;delta=float(r['repair_score'])-sc if st=='comparable' and sc is not None else None
  p.append({**{k:x[k] for k in K},'official_status':os_,'official_score':sc,'official_prediction':off[x['video_uid']].get('official_video_boolean'),'repair_status':rs,'repair_score':r['repair_score'],'repair_prediction':r['repair_prediction'],'comparison_status':st,'agreement_if_defined':abs(delta)<1e-12 if delta is not None else None,'score_delta_if_defined':delta})
 F=list(K)+['official_status','official_score','official_prediction','repair_status','repair_score','repair_prediction','comparison_status','agreement_if_defined','score_delta_if_defined']
 for name,data in [('paired_results.csv',p),('disagreements.csv',[x for x in p if x['comparison_status']=='comparable' and x['agreement_if_defined'] is False])]:
  with (out/name).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=F);w.writeheader();w.writerows(data);f.flush();os.fsync(f.fileno())
 c=Counter(x['repair_status'] for x in q.values());comp=[x for x in p if x['comparison_status']=='comparable'];s={'total_manifest_n':len(m),'official_valid_n':len(m),'official_unsupported_n':0,'repair_valid_n':c['succeeded_scalar'],'repair_structured_valid_n':c['succeeded_structured'],'repair_unsupported_n':c['unsupported'],'repair_failure_n':c['failed'],'comparable_n':len(comp),'agreement_n':sum(x['agreement_if_defined'] is True for x in comp),'disagreement_n':sum(x['agreement_if_defined'] is False for x in comp),'agreement_rate':sum(x['agreement_if_defined'] is True for x in comp)/len(comp) if comp else None,'incompatible_scale_n':sum(x['comparison_status']=='incompatible_scale' for x in p),'by_generator':{}}
 (out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');return s
def run(a):
 full=[x for x in read(R/'data/processed/e0_scoring_manifest.csv') if x['dimension']==DM[a.dimension]]
 if len({x['video_uid'] for x in full})!=len(full):raise RuntimeError('duplicate video_uid')
 for x in full:
  if not (D/x['relative_video_path']).is_file():raise RuntimeError('manifest video missing '+x['relative_video_path'])
 off=join(full,read(O/DM[a.dimension]/'results.csv'));m=full[:a.limit] if a.limit else full;out=Path(a.output_root)/('smoke' if a.smoke else '')/a.dimension;out.mkdir(parents=True,exist_ok=True)
 old={(x.get('dimension'),x.get('video_uid'),x.get('repair_variant','audit')):x for x in load(out/'repair_results.jsonl')};todo=[];skipped=0
 for x in m:
  s=old.get((x['dimension'],x['video_uid'],'audit'),{}).get('repair_status')
  if s in {'succeeded_scalar','succeeded_structured','unsupported'} or s=='failed' and not a.retry_failed:skipped+=1
  else:todo.append(x)
 sp=spatial() if a.dimension=='spatial_relationship' else {};go=[];n=0
 for x in todo:
  if meta(x,sp) is None:
   z=base(x);z.update(repair_status='unsupported',failure_or_abstention='MissingOfficialSpatialMetadata: unsupported_metadata');append(out/'repair_results.jsonl',z);n+=1
  else:go.append(x)
 if go:
  import torch
  paths();ev=runner(a.dimension,torch.device('cuda:0'))
  for x in go:
   v=D/x['relative_video_path']
   try:
    print(json.dumps({'event':'video_inference_start','dimension':a.dimension,'video_uid':x['video_uid'],'video_path':str(v)}),flush=True)
    z=ev(v,{v.name:meta(x,sp)})
    if not isinstance(z,list) or len(z)!=1:raise RuntimeError('single-video API returned non-single result')
    y=z[0];r=base(x);r.update(repair_score=y.get('score'),repair_prediction=y.get('prediction',y.get('official_video_boolean')),failure_or_abstention=y.get('failure_reason') or y.get('error'))
    if y.get('status')=='failed':r['repair_status']='failed';e=None
    elif y.get('score') is None and not y.get('diagnostics'):r.update(repair_status='failed',failure_or_abstention='API returned neither scalar score nor structured evidence');e=None
    else:r['repair_status']='succeeded_scalar' if y.get('score') is not None else 'succeeded_structured';e={'video_uid':x['video_uid'],'dimension':x['dimension'],'routing':y.get('routing'),'motion_intensity':y.get('motion_intensity'),'temporal_coverage':y.get('temporal_coverage'),'camera_evidence':y.get('camera_evidence'),'subject_residual_evidence':y.get('subject_residual_evidence'),'diagnostics':y.get('diagnostics')}
    if e is not None:append(out/'structured_evidence.jsonl',e)
    append(out/'repair_results.jsonl',r);n+=1
    print(json.dumps({'event':'video_inference_complete','dimension':a.dimension,'video_uid':x['video_uid'],'repair_status':r['repair_status'],'repair_score':r['repair_score']}),flush=True)
   except Exception as e:
    r=base(x);r.update(repair_status='failed',failure_or_abstention=f'{type(e).__name__}: {e}');append(out/'repair_results.jsonl',r);n+=1
 s=report(out,m,off);write_predictions(out,m,sp);print(json.dumps({'dimension':a.dimension,'evaluated':n,'skipped':skipped,'model_load_count':1 if go else 0,'summary':s}),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dimension',required=True,choices=DM);p.add_argument('--limit',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--retry-failed',action='store_true');p.add_argument('--output-root',default=str(OUT));a=p.parse_args();run(a)
if __name__=='__main__':main()
