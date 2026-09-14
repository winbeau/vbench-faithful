#!/usr/bin/env python3
"""Manifest-only, resumable official VBench 1.0 E0 per-dimension runner."""
from __future__ import annotations

import argparse, csv, datetime as dt, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get('VBENCH_AUDIT_UPSTREAM', str(ROOT.parent / 'VBench')))
MANIFEST = ROOT / 'data/processed/e0_scoring_manifest.csv'
DATA = Path('/root/autodl-tmp/vbench-audit-storage/datasets/e0_public')
OUT = Path('/root/autodl-tmp/vbench-audit-storage/scores/official/e0')
LOG = Path('/root/autodl-tmp/vbench-audit-storage/logs/e0_official')
EXPECTED = {'human_action': 2000, 'subject_consistency': 1440, 'dynamics_degree': 1440, 'spatial_relationship': 2160}
OFFICIAL = {'human_action': 'human_action', 'subject_consistency': 'subject_consistency', 'dynamics_degree': 'dynamic_degree', 'spatial_relationship': 'spatial_relationship'}
FIELDS = ['video_uid','dimension','split','prompt_id','group_id','generator','relative_video_path','score','status','elapsed_s','error','official_video_boolean']

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def command(*args):
    try: return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
    except Exception as e: return f'ERROR: {e}'
def rows(dim):
    with MANIFEST.open(newline='',encoding='utf-8') as f: r=[dict(x) for x in csv.DictReader(f) if x['dimension']==dim]
    if len(r)!=EXPECTED[dim] or len({x['video_uid'] for x in r})!=len(r): raise RuntimeError(f'manifest validation failed for {dim}: {len(r)}')
    for x in r:
        p=DATA/x['relative_video_path']
        if not p.is_file(): raise RuntimeError(f'manifest video missing: {p}')
    return r
def atomic_text(path, text):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f: f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def status_lines(update=None):
    p=OUT/'status.txt'; values={d:'PENDING' for d in EXPECTED}
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line:
                key,val=line.split(' ',1)
                if key in values: values[key]=val
    if update: values[update[0]]=update[1]
    atomic_text(p,''.join(f'{d} {values[d]}\n' for d in EXPECTED))
def read_results(dim):
    p=OUT/dim/'results.csv'; got={}
    if p.exists():
        with p.open(newline='',encoding='utf-8') as f:
            for x in csv.DictReader(f): got[x['video_uid']]=x
    return got
def persist(dim, ordered, result):
    current=read_results(dim); current[result['video_uid']]=result
    p=OUT/dim/'results.csv'; tmp=p.with_suffix('.csv.tmp'); p.parent.mkdir(parents=True,exist_ok=True)
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        for item in ordered:
            if item['video_uid'] in current: w.writerow({k:current[item['video_uid']].get(k,'') for k in FIELDS})
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)
def base_result(x): return {**x,'score':'','status':'error','elapsed_s':'0','error':'','official_video_boolean':''}
def import_paths(dim):
    src={'human_action':'human_action','subject_consistency':'subject_consistency','dynamics_degree':'dynamic-degree','spatial_relationship':'spatial_relationship'}[dim]
    sys.path[:0]=[str(ROOT/'metrics'/src/'src'),str(UPSTREAM)]
def patch_remote(module):
    module.UPSTREAM_REMOTE=command('git','-C',str(UPSTREAM),'remote','get-url','origin')
def spatial_metadata():
    full=json.load(open(UPSTREAM/'vbench/VBench_full_info.json',encoding='utf-8')); out={}
    for x in full:
        aux=x.get('auxiliary_info',{}).get('spatial_relationship',{}).get('spatial_relationship')
        if aux: out[x['prompt_en']]=aux
    return out
def evaluator(dim):
    import torch
    import_paths(dim); device=torch.device('cuda:0')
    if dim=='human_action':
        import human_action.backends.vbench as vb; patch_remote(vb)
        weight=Path('/root/autodl-tmp/vbench-audit-storage/models/umt/l16_ptk710_ftk710_ftk400_f16_res224.pth')
        return vb.OfficialHumanActionEvaluator(device,weight,UPSTREAM), lambda ev,x: (float((z:=ev.evaluate_video(DATA/x['relative_video_path'])).matched), bool(z.matched))
    if dim=='subject_consistency':
        import subject_consistency.backends.vbench as vb; patch_remote(vb)
        from subject_consistency.backends.vbench import import_official_module, official_diagnostics
        module,_=import_official_module(UPSTREAM)
        model=torch.hub.load(repo_or_dir='/root/autodl-tmp/vbench-audit-storage/models/dino/facebookresearch_dino_main',model='dino_vitb16',source='local',pretrained=False)
        model.load_state_dict(torch.load('/root/autodl-tmp/vbench-audit-storage/models/dino/dino_vitbase16_pretrain.pth',map_location='cpu'),strict=True)
        model=model.to(device).eval()
        class OfflineDino:
            def extract(self, video):
                images=module.dino_transform(224)(module.load_video(str(video)))
                with module.torch.no_grad():
                    return module.torch.cat([module.F.normalize(model(image.unsqueeze(0).to(device)),dim=-1,p=2) for image in images],dim=0)
        ev=OfflineDino()
        return ev, lambda ev,x: (float(official_diagnostics(ev.extract(DATA/x['relative_video_path'])).final_score), None)
    if dim=='dynamics_degree':
        import dynamic_degree.backends.vbench as vb; patch_remote(vb)
        weight=Path('/root/autodl-tmp/vbench-audit-storage/models/raft/raft-things.pth')
        return vb.OfficialDynamicEvaluator(device,weight,UPSTREAM), lambda ev,x: (float((z:=ev.evaluate_video(DATA/x['relative_video_path'])).official_video_boolean), bool(z.official_video_boolean))
    import spatial_relationship.backends.vbench as vb; patch_remote(vb)
    from spatial_relationship.metric import parse_query
    weight=Path('/root/autodl-tmp/vbench-audit-storage/models/grit/grit_b_densecap_objectdet.pth')
    ev=vb.OfficialVBenchEvaluator(device,weight,UPSTREAM); metadata=spatial_metadata()
    def score(ev,x):
        if x['prompt_id'] not in metadata: raise ValueError('MissingOfficialSpatialMetadata: prompt absent from locked VBench 1.0 VBench_full_info.json')
        meta={'prompt':x['prompt_id'],'dimension_metadata':{'spatial_relationship':metadata[x['prompt_id']]}}
        q=parse_query(meta); raw=ev.evaluate_video(DATA/x['relative_video_path'],meta,q)
        return float(raw[1][0]['video_results']),None
    return ev,score
def systemic(exc):
    s=f'{type(exc).__name__}: {exc}'.lower()
    return any(x in s for x in ('out of memory','cuda error','cudaerr','device-side assert','cudnn_status','driver version is insufficient'))
def do_preflight(dim):
    r=rows(dim); status_lines((dim,'PREFLIGHT RUNNING 0/'+str(len(r))))
    started=time.monotonic(); ev,score=evaluator(dim); x=r[0]; result=base_result(x)
    try:
        value,boolean=score(ev,x); result.update(score=value,status='success',elapsed_s=f'{time.monotonic()-started:.6f}',official_video_boolean='' if boolean is None else str(boolean).lower())
    except Exception as e:
        result.update(error=f'{type(e).__name__}: {e}',elapsed_s=f'{time.monotonic()-started:.6f}')
    p=OUT/'_preflight'/f'{dim}.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerow({k:result.get(k,'') for k in FIELDS});f.flush();os.fsync(f.fileno())
    ok=result['status']=='success'; status_lines((dim,('PREFLIGHT PASS' if ok else 'PREFLIGHT FAIL')+f' 1/{len(r)}'))
    print(f'{dim} PREFLIGHT {"PASS" if ok else "FAIL"}: {result["error"] or result["score"]}',flush=True); return 0 if ok else 1
def run(dim):
    r=rows(dim); old=read_results(dim); status_lines((dim,f'RUNNING {sum(1 for x in old.values() if x.get("status")=="success")}/{len(r)}'))
    ev,score=evaluator(dim); started=time.monotonic()
    for index,x in enumerate(r,1):
        if old.get(x['video_uid'],{}).get('status')=='success': continue
        t=time.monotonic(); result=base_result(x)
        try:
            value,boolean=score(ev,x); result.update(score=value,status='success',elapsed_s=f'{time.monotonic()-t:.6f}',official_video_boolean='' if boolean is None else str(boolean).lower())
        except Exception as e:
            if systemic(e): raise RuntimeError(f'SYSTEMIC {type(e).__name__}: {e}') from e
            result.update(error=f'{type(e).__name__}: {e}',elapsed_s=f'{time.monotonic()-t:.6f}')
        persist(dim,r,result); old[x['video_uid']]=result
        done=sum(1 for q in old.values() if q.get('status') in ('success','error')); good=sum(1 for q in old.values() if q.get('status')=='success')
        status_lines((dim,f'RUNNING {done}/{len(r)} success={good}'))
        print(f'{dim} {index}/{len(r)} {result["status"]} {x["video_uid"]} {result["elapsed_s"]}s',flush=True)
    good=sum(1 for q in old.values() if q.get('status')=='success'); err=sum(1 for q in old.values() if q.get('status')=='error')
    status_lines((dim,(f'PASS {good}/{len(r)}' if not err else f'COMPLETE {good}/{len(r)} error={err}')))
    print(f'{dim} COMPLETE runtime_s={time.monotonic()-started:.2f} success={good} error={err}',flush=True)
def metadata():
    import torch
    cps={'UMT':'/root/autodl-tmp/vbench-audit-storage/models/umt/l16_ptk710_ftk710_ftk400_f16_res224.pth','DINO':'/root/autodl-tmp/vbench-audit-storage/models/dino/dino_vitbase16_pretrain.pth','RAFT':'/root/autodl-tmp/vbench-audit-storage/models/raft/raft-things.pth','GRiT':'/root/autodl-tmp/vbench-audit-storage/models/grit/grit_b_densecap_objectdet.pth'}
    text={'started_utc':now(),'vbench_audit_sha':command('git','-C',str(ROOT),'rev-parse','HEAD'),'vbench1_sha':command('git','-C',str(UPSTREAM),'rev-parse','HEAD'),'manifest':str(MANIFEST),'manifest_sha256':sha(MANIFEST),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':command('nvidia-smi','--query-gpu=name','--format=csv,noheader'),'checkpoints':{k:{'path':v,'sha256':sha(v)} for k,v in cps.items()},'official_dimension_mapping':OFFICIAL,'spatial_overlay':{'Pillow':'9.5.0','detectron2':'cc87e7ec','CUDA_HOME':'/root/autodl-tmp/vbench-audit-storage/toolchains/cuda-12.8'}}
    atomic_text(OUT/'run_metadata.txt',json.dumps(text,ensure_ascii=False,indent=2)+'\n')
def summary():
    lines=[]; total=[0,0,0,0]
    for d,n in EXPECTED.items():
        got=read_results(d); suc=sum(x.get('status')=='success' for x in got.values()); err=sum(x.get('status')=='error' for x in got.values()); missing=n-len(got); runtime=sum(float(x.get('elapsed_s') or 0) for x in got.values()); mean=runtime/len(got) if got else 0
        lines.append(f'{d} expected={n} success={suc} error={err} missing={missing} runtime_s={runtime:.3f} mean_time_per_video_s={mean:.3f}'); total[0]+=n;total[1]+=suc;total[2]+=err;total[3]+=missing
    lines.append(f'total expected={total[0]} success={total[1]} error={total[2]} missing={total[3]}'); atomic_text(OUT/'summary.txt','\n'.join(lines)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['metadata','preflight','run','summary']);p.add_argument('--dimension',choices=EXPECTED);a=p.parse_args()
    if a.action=='metadata': metadata(); status_lines(); return
    if a.action=='summary': summary(); return
    if a.action=='preflight': raise SystemExit(do_preflight(a.dimension))
    run(a.dimension)
if __name__=='__main__': main()
