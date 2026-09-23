"""Inspect all recovered SAM alternatives and replay one native alignment.

The selected source pair is a post-hoc DEV mechanism case, not a score cohort.
All generated masks are shown; no visual selection feeds the alignment routine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dynamic_degree.native_region_motion import (NativeMotionConfig, fit_native_regions, rank_with_common_sift)
from dynamic_degree.trajectory import decode_video, TrajectoryConfig
from dynamic_degree.sparse_structure import METHODS, extract_features, reciprocal_matches
from vbench_audit_models.sam_regions import unpack_masks
from .probe_native_region_motion import load_masks, resolve_media
from .score_static_jitter import source_hashes
from .static_jitter import digest


def selected_prediction_indices(identity, mode):
    if mode not in {'all','rgb-only'}:raise ValueError('unknown proposal-source ablation')
    selected=[]
    for i,r in enumerate(identity['regions']):
        origin=identity['request']['prompts'][r['request_prompt_index']]['origin']
        if mode=='all' or origin.startswith('native_integer_peak_') or origin=='unshifted_identity_proposal':
            selected.append(i)
    return selected


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('probe-run','region-run','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--video-root',action='append',default=[])
    p.add_argument('--proposal-source',choices=['all','rgb-only'],default='all',
                   help='RGB-only keeps SIFT witnesses out of the SAM prompt proposal stage')
    p.add_argument('--keypoint-method',choices=METHODS,default='sift')
    args=p.parse_args(argv)
    probe,regions,output=map(Path,(args.probe_run,args.region_run,args.output))
    if output.exists():raise FileExistsError('fresh inspection output required')
    identity=json.loads((probe/'provenance.json').read_text());request=identity['request']
    runtime=json.loads((probe/'runtime.json').read_text())
    if (runtime['status']!='finished' or digest(probe/'masks.npz')!=identity['masks_sha256']
            or digest(regions/'provenance.json')!=request['region_provenance_sha256']):
        raise ValueError('complete bound prompted/SAM run required')
    video=resolve_media({'video':request['video'],'sha256':request['video_sha256']},args.video_root)
    frames,times,_=decode_video(video,TrajectoryConfig(sample_fps=8,max_side=512))
    source,target,index=request['start'],request['target'],request['source_region']
    for kind,i in (('source',source),('target',target)):
        if hashlib.sha256(frames[i].tobytes()).hexdigest()!=request[f'{kind}_frame_sha256']:
            raise ValueError('native frame pixels changed')
    masks=load_masks(regions/'evidence'/f'{request["candidate_id"]}.npz',request['cache_sha256'],frames,times)
    with np.load(probe/'masks.npz',allow_pickle=False) as z:recovered=unpack_masks(z['masks_packed'],z['image_shape'])
    if len(recovered)!=len(identity['regions']) or len(recovered)!=runtime['predictions']:
        raise ValueError('mask/record count mismatch')
    source_mask=masks[source][index]
    selected=selected_prediction_indices(identity,args.proposal_source)
    targets=np.concatenate((masks[target],recovered[selected]))
    config=NativeMotionConfig(target_visibility='associated_sam')
    cv2.setNumThreads(1)
    matches=reciprocal_matches(extract_features(frames[source],args.keypoint_method),
                               extract_features(frames[target],args.keypoint_method),config.sift_ratio)
    old=fit_native_regions(frames[source],frames[target],source_mask[None],masks[target],matches,config)[0]
    new=fit_native_regions(frames[source],frames[target],source_mask[None],targets,matches,config)[0]
    check=rank_with_common_sift(new,targets,matches)
    output.mkdir(parents=True)
    views=[(frames[source],f'source frame {source}'),
           ((frames[source]*np.where(source_mask[...,None],1.,.15)).astype(np.uint8),f'source region {index}'),
           (frames[target],f'target frame {target}')]
    for mask,r in zip(recovered,identity['regions']):
        title=f"id {r['output_region']} p{r['request_prompt_index']} {'box+pt' if r['uses_positive_point'] else 'box'} {'multi' if r['multimask_output'] else 'single'}"
        views.append(((frames[target]*np.where(mask[...,None],1.,.15)).astype(np.uint8),title))
    w,h,cols=176,204,6
    sheet=np.full((44+((len(views)+cols-1)//cols)*h,cols*w,3),255,np.uint8)
    cv2.putText(sheet,'Every prompted SAM output; none selected by visual review',(6,25),cv2.FONT_HERSHEY_SIMPLEX,.52,(20,20,20),1,cv2.LINE_AA)
    for i,(frame,title) in enumerate(views):
        x,y=(i%cols)*w,44+(i//cols)*h
        cv2.putText(sheet,title,(x+2,y+18),cv2.FONT_HERSHEY_SIMPLEX,.32,(20,20,20),1,cv2.LINE_AA)
        sheet[y+28:y+h,x:x+w]=cv2.resize(frame,(w,w),interpolation=cv2.INTER_AREA)
    figure=output/'all_prompted_regions.png'
    if not cv2.imwrite(str(figure),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR)):raise RuntimeError('image write failed')
    result={'status':'diagnostic_only','score':None,'formal_acceptance':'NOT EVALUATED',
            'scope':'single selected source region/frame pair; no full-video or invariance acceptance',
            'probe_provenance_sha256':digest(probe/'provenance.json'),'script_sha256':digest(Path(__file__)),
            'code_files':source_hashes(Path(__file__).resolve().parents[2]),
            'original_target_regions':len(masks[target]),'added_hypotheses':len(selected),
            'proposal_source':args.proposal_source,'selected_prediction_indices':selected,
            'keypoint_method':args.keypoint_method,
            'baseline':old,'augmented':new,'spatial_matches':matches,'geometric_compatibility':check,
            'legacy_match_field_note':'regional sift_* keys store the declared keypoint_method, not necessarily SIFT',
            'all_selected_targets_given_to_alignment':True,'figure_sha256':digest(figure)}
    (output/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'original_target_regions':len(masks[target]),'added':len(selected),
                     'baseline_displacement':old['displacement_pixels'],
                     'augmented_displacement':new['displacement_pixels'],
                     'common_witnesses':check['unique_witness_locations'],
                     'compatible_hypothesis':check['best_compatible_hypothesis']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
