"""Bound single-frame-pair re-localization diagnostic, not motion acceptance.

Prepare proposals from an already measured DEV video; infer on that same video's
target frame. Neither a paired clean video nor manually chosen target pixels are
model inputs. All SAM alternatives, including empty predictions, are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from dynamic_degree.prompted_regions import relocation_prompts
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from .probe_native_region_motion import load_masks, resolve_media
from .review_selection import select_reviewed_candidates
from .score_static_jitter import command_output, source_hashes
from .static_jitter import digest


def prepare(args):
    root = Path(__file__).resolve().parents[2]
    manifest, review, native, regions = map(Path, (args.manifest,args.review,args.native_run,args.region_run))
    rows = select_reviewed_candidates([json.loads(s) for s in manifest.read_text().splitlines()],review,manifest,root)
    row = next(r for r in rows if r['candidate_id']==args.candidate)
    identity = json.loads((native/'provenance.json').read_text())
    region_id = json.loads((regions/'provenance.json').read_text())
    if (identity['status']!='diagnostic_only' or identity['completed']!=len(rows)
            or digest(native/'diagnostics.jsonl')!=identity['diagnostics_sha256']
            or identity['common_identity']['region_provenance_sha256']!=digest(regions/'provenance.json')
            or region_id['manifest_sha256']!=digest(manifest) or region_id['review_sha256']!=digest(review)
            or identity['common_identity']['input_sha256']!={r['candidate_id']:r['sha256'] for r in rows}):
        raise ValueError('bound complete native/SAM evidence and reviewed development cohort required')
    with (native/'diagnostics.jsonl').open() as f:
        record=next(r for r in map(json.loads,f) if r['candidate_id']==args.candidate)
    pair=next(p for p in record['pairs'] if p['start']==args.start and p['lag']==args.lag)
    measured=pair['regions'][args.region]
    if measured['whole_frame_control']:
        raise ValueError('this recovery diagnostic requires an actual source SAM region')
    hypotheses=[{'origin':f'native_integer_peak_{i}','displacement_pixels':h['integer_peak']['displacement_pixels']}
                for i,h in enumerate(measured['hypotheses'])]
    keys=set(measured['sift_match_source_keys'])
    hypotheses.extend({'origin':f'sift_key_{m["source_key"]}',
                       'displacement_pixels':np.subtract(m['target_xy'],m['source_xy']).tolist()}
                      for m in pair['sift_matches'] if m['source_key'] in keys)
    hypotheses.append({'origin':'unshifted_identity_proposal','displacement_pixels':[0.,0.]})
    frames,times,sampling=decode_video(resolve_media(row,args.video_root),TrajectoryConfig(sample_fps=8,max_side=512))
    cache=regions/'evidence'/f'{args.candidate}.npz'
    masks=load_masks(cache,region_id['cache_sha256'][args.candidate],frames,times)
    prompts=relocation_prompts(masks[args.start][args.region],hypotheses)
    request={'scope':'posthoc DEV mechanism probe; not full-video score or acceptance',
             'candidate_id':args.candidate,'video':row['video'],'video_sha256':row['sha256'],
             'cache_sha256':digest(cache),'start':args.start,'target':args.start+args.lag,'source_region':args.region,
             'input_shape':list(frames.shape),'timestamps':times.tolist(),'sampling':sampling,
             'source_frame_sha256':hashlib.sha256(frames[args.start].tobytes()).hexdigest(),
             'target_frame_sha256':hashlib.sha256(frames[args.start+args.lag].tobytes()).hexdigest(),
             'native_provenance_sha256':digest(native/'provenance.json'),
             'native_diagnostics_sha256':identity['diagnostics_sha256'],
             'region_provenance_sha256':digest(regions/'provenance.json'),
             'sam_source_files':region_id['model_identity']['source_files'],
             'sam_checkpoint_sha256':region_id['model_identity']['checkpoint_sha256'],
             'sam_revision':region_id['sam_revision'],'prompts':prompts,
             'preparation_script_sha256':digest(Path(__file__)),
             'prompt_algorithm_sha256':digest(root/'metrics/dynamic-degree/src/dynamic_degree/prompted_regions.py')}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(request,f,indent=2,allow_nan=False)
    print(json.dumps({'output':str(output),'prompts':len(prompts),'scope':request['scope']}))
    return 0


def infer(args):
    import torch
    from vbench_audit_models.sam_regions import SamPromptRegionModel

    request_path,output=Path(args.request),Path(args.output)
    if output.exists():raise FileExistsError('fresh output required; never overwrite a probe')
    request=json.loads(request_path.read_text())
    video,cache=Path(args.video or request['video']),Path(args.region_cache)
    if digest(video)!=request['video_sha256'] or digest(cache)!=request['cache_sha256']:
        raise ValueError('same-video byte-bound media and source SAM cache required')
    frames,times,sampling=decode_video(video,TrajectoryConfig(sample_fps=8,max_side=512))
    if list(frames.shape)!=request['input_shape'] or not np.array_equal(times,request['timestamps']):
        raise ValueError('native shape or timeline changed')
    for kind,index in (('source',request['start']),('target',request['target'])):
        if hashlib.sha256(frames[index].tobytes()).hexdigest()!=request[f'{kind}_frame_sha256']:
            raise ValueError('native decoded pixels differ across environments')
    masks=load_masks(cache,request['cache_sha256'],frames,times)
    prepared=relocation_prompts(masks[request['start']][request['source_region']],request['prompts'])
    if prepared!=request['prompts']:raise ValueError('request prompts differ from bound source geometry')
    torch.set_num_threads(4);torch.manual_seed(42);cv2.setNumThreads(1)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
        free,_=torch.cuda.mem_get_info()
        if free<24*1024**3:raise RuntimeError('at least 24 GiB free required; model not started')
        torch.cuda.set_per_process_memory_fraction(.15)
    if digest(Path(args.sam_weight))!=request['sam_checkpoint_sha256']:
        raise ValueError('reuse the exact existing local SAM weight')
    if command_output(['git','-C',args.sam_root,'rev-parse','HEAD'])!=request['sam_revision']:
        raise ValueError('SAM revision changed')
    output.mkdir(parents=True)
    runtime={'status':'running','pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    started=time.monotonic()
    (output/'runtime.json').write_text(json.dumps(runtime,indent=2))
    try:
        model=SamPromptRegionModel(args.sam_root,args.sam_weight,args.device)
        if model.identity['source_files']!=request['sam_source_files']:
            raise ValueError('SAM execution source differs from earlier automatic regions')
        used=[(i,p) for i,p in enumerate(request['prompts']) if p['status']=='proposal_only']
        predicted=model.propose(frames[request['target']],[p for _,p in used])
        arrays=[];records=[]
        for r in predicted:
            mask=r.pop('segmentation');arrays.append(mask)
            records.append({**r,'request_prompt_index':used[r['prompt_index']][0],
                            'area_pixels':int(mask.sum()),'output_region':len(records)})
        stacked=np.stack(arrays) if arrays else np.zeros((0,*frames.shape[1:3]),bool)
        np.savez_compressed(output/'masks.npz',masks_packed=np.packbits(stacked,axis=-1),
                            image_shape=np.array(frames.shape[1:3]))
        provenance={'status':'diagnostic_only','score':None,'formal_acceptance':'NOT EVALUATED',
                    'request_sha256':digest(request_path),'request':request,'sampling':sampling,
                    'model_identity':model.identity,'sam_dirty':command_output(['git','-C',args.sam_root,'status','--porcelain']),
                    'script_sha256':digest(Path(__file__)),
                    'code_files':source_hashes(Path(__file__).resolve().parents[2]),
                    'masks_sha256':digest(output/'masks.npz'),'regions':records,
                    'environment':{'python':platform.python_version(),'torch':torch.__version__,'opencv':cv2.__version__,
                                   'device':args.device,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}}
        (output/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False))
        runtime.update(status='finished',predictions=len(records))
    except Exception as exc:
        runtime.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        runtime.update(elapsed_seconds=time.monotonic()-started,
                       finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        if args.device.startswith('cuda'):
            runtime['max_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        (output/'runtime.json').write_text(json.dumps(runtime,indent=2))
    print(json.dumps(runtime))
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for name in ('manifest','review','native-run','region-run','candidate','output'):prep.add_argument('--'+name,required=True)
    for name in ('start','lag','region'):prep.add_argument('--'+name,type=int,required=True)
    prep.add_argument('--video-root',action='append',default=[])
    run=sub.add_parser('infer')
    for name in ('request','region-cache','sam-root','sam-weight','output'):run.add_argument('--'+name,required=True)
    run.add_argument('--video');run.add_argument('--device',default='cuda:0')
    args=p.parse_args(argv)
    if args.action=='prepare':
        if min(args.start,args.region)<0 or args.lag<1:p.error('nonnegative indices and positive lag required')
        return prepare(args)
    return infer(args)


if __name__=='__main__':raise SystemExit(main())
