#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,os,sys
from collections import defaultdict,Counter
from pathlib import Path
R=Path(__file__).resolve().parents[1]
D=Path(os.environ.get('VBENCH_AUDIT_DATA_ROOT', '/root/autodl-tmp/vbench-audit-storage/datasets/e0_public'))
O=Path(os.environ.get('VBENCH_AUDIT_OFFICIAL_ROOT', '/root/autodl-tmp/vbench-audit-storage/scores/official/e0'))
OUT=Path(os.environ.get('VBENCH_AUDIT_OUTPUT_ROOT', '/root/autodl-tmp/vbench-audit-storage/runs/official_dataset_compare_4dims'))
U=Path(os.environ.get('VBENCH_AUDIT_UPSTREAM', str(R.parent / 'VBench')))
DM={'dynamic_degree':'dynamics_degree','spatial_relationship':'spatial_relationship','human_action':'human_action','subject_consistency':'subject_consistency','motion_smoothness':'motion_smoothness'}; K=('video_uid','dimension','prompt_id','generator','relative_video_path'); S={'pending','succeeded_scalar','succeeded_structured','unsupported','failed'}
def append(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def load(p):
 if not p.exists():return []
 try:return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
 except Exception as e:raise RuntimeError(f'manifest corruption: {e}')
def read(p):
 with p.open(newline='',encoding='utf8') as f:return [dict(x) for x in csv.DictReader(f)]
def video_path(relative):
 p=D/relative
 if p.is_file(): return p
 # The mirror preserves the generator/dimension directory but some packs
 # transcode the same frozen sample from GIF to MP4. Resolve only the same
 # relative path stem; never search another dimension or generator.
 for suffix in ('.mp4','.gif'):
  q=p.with_suffix(suffix)
  if q.is_file(): return q
 return p
def motion_input(path):
 if path.suffix.lower() != '.gif': return path
 import hashlib
 cache=OUT/'_motion_mp4';cache.mkdir(parents=True,exist_ok=True)
 target=cache/(hashlib.sha256(str(path).encode()).hexdigest()[:24]+'.mp4')
 if target.is_file(): return target
 from PIL import Image,ImageSequence
 import cv2,numpy as np
 with Image.open(path) as image:
  frames=[np.asarray(frame.convert('RGB')) for frame in ImageSequence.Iterator(image)]
 if not frames: raise RuntimeError(f'GIF contains no frames: {path}')
 height,width=frames[0].shape[:2]
 writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),16.0,(width,height))
 try:
  if not writer.isOpened(): raise RuntimeError(f'cannot create motion cache: {target}')
  for frame in frames: writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
 finally: writer.release()
 return target
def join(m,o):
 q=defaultdict(list)
 for x in o:q[tuple(x[k] for k in K)].append(x)
 z={}
 for x in m:
  a=q[tuple(x[k] for k in K)]
  if len(a)!=1:raise RuntimeError(f'Official strict join {len(a)} matches: {tuple(x[k] for k in K)}')
  z[x['video_uid']]=a[0]
 return z
# packages/audit-models/src is required: dynamic_degree.models imports
# vbench_audit_models, and without it every dynamic_degree shard dies on import.
def paths():sys.path[:0]=[str(R/x) for x in ['packages/audit-core/src','packages/audit-models/src','metrics/dynamic-degree/src','metrics/spatial-relationship/src','metrics/human-action/src','metrics/subject-consistency/src','metrics/motion-smoothness/src','metrics/scene/src','metrics/multiple-objects/src']]+[str(U)]
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
def runner(dim,dev,backend='repair'):
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
 elif dim=='motion_smoothness':
  from motion_smoothness.backends.vbench import OfficialMotionSmoothnessEvaluator,DEFAULT_WEIGHT as AMT_WEIGHT
  from motion_smoothness.backends.audit import evaluate_timed_frames
  from motion_smoothness.models import RaftFlowEstimator,decode_timed_frames
  from motion_smoothness.schemas import MotionSmoothnessConfig
  if backend == 'official':
   checkpoint=Path(os.environ.get('VBENCH_AUDIT_AMT_WEIGHT',str(AMT_WEIGHT))).expanduser()
   model=OfficialMotionSmoothnessEvaluator(dev,checkpoint=checkpoint,upstream_path=U)
   f=lambda v,x:[{'score':model.evaluate_video(v),'status':'succeeded'}]
  else:
   checkpoint=Path(os.environ.get('VBENCH_AUDIT_RAFT_WEIGHT',str(Path.home()/'.cache/vbench/raft_model/models/raft-things.pth'))).expanduser()
   estimator=RaftFlowEstimator(dev,checkpoint,U);config=MotionSmoothnessConfig()
   def f(v,x):
    frames,_=decode_timed_frames(v)
    item=evaluate_timed_frames(v,frames,estimator,config)
    return [{'score':item.score,'status':'succeeded','diagnostics':item.diagnostics}]
 elif dim=='scene' or dim=='multiple_objects':
  raise RuntimeError(f'{dim} repair runner is not wired into the comparator')
 else:
  from subject_consistency.models import OfficialDinoFeatureExtractor
  from subject_consistency.metric import build_dino_config,evaluate_batch,upstream_path
  c=build_dino_config();m=OfficialDinoFeatureExtractor(dev,c,upstream_path());f=lambda v,x:evaluate_batch('audit',[v],x,dev,c,extractor=m)
 print(json.dumps({'event':'model_load_complete','dimension':dim,'model_load_count':1}),flush=True);return f
def base(x,variant='audit'):
 return {k:x[k] for k in K}|{'video_path':str(video_path(x['relative_video_path'])),'model':x['generator'],'group_id':x['group_id'],'split':x['split'],'synthetic_id':x['video_uid'],'repair_variant':variant,'repair_status':'pending','repair_score':None,'repair_prediction':None,'failure_or_abstention':None}
def write_predictions(out,m,sp,variant='audit'):
 latest={(x['video_uid'],x.get('repair_variant','audit')):x for x in load(out/'repair_results.jsonl')}
 fields=['video_uid','video_path','model','dimension','split','prompt_id','group_id','relative_video_path','explicit_target_or_metadata','repair_score','repair_status','error']
 data=[]
 for x in m:
  r=latest.get((x['video_uid'],variant),base(x,variant)); metadata=meta(x,sp)
  data.append({'video_uid':x['video_uid'],'video_path':str(video_path(x['relative_video_path'])),'model':x['generator'],'dimension':x['dimension'],'split':x['split'],'prompt_id':x['prompt_id'],'group_id':x['group_id'],'relative_video_path':x['relative_video_path'],'explicit_target_or_metadata':json.dumps(metadata.get('dimension_metadata') if metadata else None,ensure_ascii=False,sort_keys=True),'repair_score':r.get('repair_score'),'repair_status':r.get('repair_status','pending'),'error':r.get('failure_or_abstention')})
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
 full=[x for x in read(Path(a.manifest)) if x['dimension']==DM[a.dimension]]
 if a.video_uid:
  full=[x for x in full if x['video_uid']==a.video_uid]
  if len(full)!=1:raise RuntimeError(f'expected exactly one manifest row for video_uid={a.video_uid!r}, got {len(full)}')
 elif a.num_shards != 1:
  if not 0 <= a.shard_index < a.num_shards:raise ValueError('shard_index must be in [0, num_shards)')
  full=full[a.shard_index::a.num_shards]
 if len({x['video_uid'] for x in full})!=len(full):raise RuntimeError('duplicate video_uid')
 m=full[:a.limit] if a.limit else full
 for x in m:
  if not video_path(x['relative_video_path']).is_file():raise RuntimeError('manifest video missing '+x['relative_video_path'])
 variant='audit' if a.backend == 'repair' else 'official'
 off=join(m,read(O/DM[a.dimension]/'results.csv')) if a.backend == 'repair' else {};out=Path(a.output_root)/('smoke' if a.smoke else '')/a.dimension;out.mkdir(parents=True,exist_ok=True)
 old={(x.get('dimension'),x.get('video_uid'),x.get('repair_variant','audit')):x for x in load(out/'repair_results.jsonl')};todo=[];skipped=0
 for x in m:
  s=old.get((x['dimension'],x['video_uid'],variant),{}).get('repair_status')
  if s in {'succeeded_scalar','succeeded_structured','unsupported'} or s=='failed' and not a.retry_failed:skipped+=1
  else:todo.append(x)
 sp=spatial() if a.dimension=='spatial_relationship' else {};go=[];n=0
 for x in todo:
  if meta(x,sp) is None:
   z=base(x,variant);z.update(repair_status='unsupported',failure_or_abstention='MissingOfficialSpatialMetadata: unsupported_metadata');append(out/'repair_results.jsonl',z);n+=1
  else:go.append(x)
 if go:
  import torch
  paths();ev=runner(a.dimension,torch.device('cuda:0'),a.backend)
  for x in go:
   v=video_path(x['relative_video_path'])
   if a.dimension == 'motion_smoothness': v=motion_input(v)
   try:
    print(json.dumps({'event':'video_inference_start','dimension':a.dimension,'video_uid':x['video_uid'],'video_path':str(v)}),flush=True)
    z=ev(v,{v.name:meta(x,sp)})
    if not isinstance(z,list) or len(z)!=1:raise RuntimeError('single-video API returned non-single result')
    y=z[0];r=base(x,variant);r.update(repair_score=y.get('score'),repair_prediction=y.get('prediction',y.get('official_video_boolean')),failure_or_abstention=y.get('failure_reason') or y.get('error'))
    if y.get('status')=='failed':r['repair_status']='failed';e=None
    elif y.get('score') is None and not y.get('diagnostics'):r.update(repair_status='failed',failure_or_abstention='API returned neither scalar score nor structured evidence');e=None
    else:r['repair_status']='succeeded_scalar' if y.get('score') is not None else 'succeeded_structured';e={'video_uid':x['video_uid'],'dimension':x['dimension'],'routing':y.get('routing'),'motion_intensity':y.get('motion_intensity'),'temporal_coverage':y.get('temporal_coverage'),'camera_evidence':y.get('camera_evidence'),'subject_residual_evidence':y.get('subject_residual_evidence'),'diagnostics':y.get('diagnostics')}
    if e is not None:append(out/'structured_evidence.jsonl',e)
    append(out/'repair_results.jsonl',r);n+=1
    print(json.dumps({'event':'video_inference_complete','dimension':a.dimension,'video_uid':x['video_uid'],'repair_status':r['repair_status'],'repair_score':r['repair_score']}),flush=True)
   except Exception as e:
    r=base(x,variant);r.update(repair_status='failed',failure_or_abstention=f'{type(e).__name__}: {e}');append(out/'repair_results.jsonl',r);n+=1
 if a.backend == 'repair':
  s=report(out,m,off)
 else:
  records=[x for x in load(out/'repair_results.jsonl') if x.get('repair_variant') == variant]
  s={'total_manifest_n':len(m),'official_valid_n':sum(x.get('repair_status') == 'succeeded_scalar' for x in records),'official_failure_n':sum(x.get('repair_status') == 'failed' for x in records)}
 write_predictions(out,m,sp,variant);print(json.dumps({'dimension':a.dimension,'backend':a.backend,'evaluated':n,'skipped':skipped,'model_load_count':1 if go else 0,'summary':s}),flush=True)
def main():
 global D,O,OUT,U
 p=argparse.ArgumentParser()
 p.add_argument('--dimension',required=True,choices=DM)
 p.add_argument('--backend',choices=('repair','official'),default='repair')
 p.add_argument('--limit',type=int)
 p.add_argument('--video-uid',help='evaluate one frozen-manifest record for a targeted smoke')
 p.add_argument('--smoke',action='store_true')
 p.add_argument('--retry-failed',action='store_true')
 p.add_argument('--shard-index',type=int,default=0)
 p.add_argument('--num-shards',type=int,default=1)
 p.add_argument('--data-root',default=str(D))
 p.add_argument('--official-root',default=str(O))
 p.add_argument('--output-root',default=str(OUT))
 p.add_argument('--upstream',default=str(U))
 p.add_argument('--manifest',default=str(R/'data/processed/e0_scoring_manifest.csv'))
 a=p.parse_args()
 D=Path(a.data_root); O=Path(a.official_root); OUT=Path(a.output_root); U=Path(a.upstream)
 os.environ['VBENCH_AUDIT_UPSTREAM']=str(U)
 run(a)
if __name__=='__main__':main()
