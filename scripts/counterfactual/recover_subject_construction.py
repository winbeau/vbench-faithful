"""Apply frozen wider GrabCut recovery to prior independent construction masks.

Reuses the original SegFormer pass and raw video bytes, not scoring masks or
scores. Rejected source inputs remain in the new candidate population.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .common import sha256_file
from .region_discrimination import RejectedBase, recover_grabcut_extent
from .subject_artifacts import (artifact_path,new_output,object_sha256,read_jsonl,read_png_sequence,
                               upstream_frames,write_json,write_jsonl,write_npz,write_png_sequence)


def recover_entry(arguments):
    row, source, output, video_root, mapping = arguments
    cv2.setNumThreads(1)
    original_path=artifact_path(source,row['manifest'])
    original=json.loads(original_path.read_text());entry=deepcopy(original)
    if (entry.get('role')!='construction' or entry.get('localizer',{}).get('family')!='segformer'
            or entry['base']['video_uid']!=row['video_uid'] or entry['base']['base_id']!=row['base_id']):
        raise ValueError('recovery requires independent SegFormer construction input')
    entry['recovery_source_manifest_sha256']=sha256_file(original_path)
    entry['construction_recovery']={'method':'second_grabcut_with_padded_component_boxes',
                                  'padding_fraction':mapping['recovery_padding_fraction'],'scores_used':False}
    entry['class_map_sha256']=object_sha256(mapping)
    if 'mask_file' in original:
        path=artifact_path(source,original['mask_file']['path'])
        if sha256_file(path)!=original['mask_file']['sha256']:raise ValueError('source construction mask changed')
        with np.load(path,allow_pickle=False) as data:masks,others=data['masks'],data['other_instances']
        if original.get('frames'):
            frames=read_png_sequence(source,original['frames'])
        else:
            video=artifact_path(video_root,entry['base']['relative_video_path'])
            if sha256_file(video)!=entry['source_video_sha256']:raise ValueError('raw source video changed')
            frames=upstream_frames(video)
        if frames.shape[:3]!=masks.shape:raise ValueError('source frame/mask shapes differ')
        recovered,reasons,fallbacks=[],[],[]
        for frame,mask in zip(frames,masks):
            why=[];fallback=None
            try:current=recover_grabcut_extent(frame,mask,person=entry['base']['subject_en']=='person',
                                              padding_fraction=mapping['recovery_padding_fraction'])
            except RejectedBase as exc:
                if mapping.get('recovery_keep_valid_initial') and .01 <= float(mask.mean()) <= .50:
                    current=mask.copy();fallback=exc.reason
                else:current=np.zeros_like(mask);why.append(exc.reason)
            if (mapping.get('recovery_keep_valid_initial') and .01 <= float(mask.mean()) <= .50
                    and not .01 <= float(current.mean()) <= .50):
                current=mask.copy();fallback='recovered_area_outside_frozen_gates'
            area=float(current.mean())
            if area<.01:why.append('mask_area_below_1_percent')
            if area>.50:why.append('mask_area_above_50_percent')
            recovered.append(current);reasons.append(why);fallbacks.append(fallback)
        masks=np.stack(recovered);others=others*(1-masks)
        entry.update(area_pixels=masks.sum(axis=(1,2)).tolist(),area_ratio=masks.mean(axis=(1,2)).tolist(),
                     frame_rejection_reasons=reasons,rejection_reasons=sorted({r for rs in reasons for r in rs}))
        if mapping.get('recovery_keep_valid_initial'):
            entry['frame_recovery_fallbacks']=fallbacks
            entry['construction_recovery']['keep_valid_initial_on_failed_recovery']=True
        entry.pop('frames',None);entry.pop('mask_file',None)
        entry['status']='rejected' if entry['rejection_reasons'] else 'accepted'
        path=output/'masks'/(row['video_uid']+'.npz')
        write_npz(path,masks=masks,other_instances=others,area=np.asarray(entry['area_pixels']),
                  metadata_json=json.dumps(entry,sort_keys=True))
        entry['mask_file']={'path':str(path.relative_to(output)),'sha256':sha256_file(path)}
        if entry['status']=='accepted':entry['frames']=write_png_sequence(output,'frames/'+row['video_uid'],frames)
    relative='manifests/'+entry['base']['base_id']+'.json';write_json(output/relative,entry)
    return {**row,'manifest':relative,'status':entry['status'],'rejection_reasons':entry['rejection_reasons']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('source','output','protocol','class-map','video-root'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());mapping=json.loads(a.class_map.read_text())
    if sha256_file(a.source/'run.json')!=protocol['source_construction_run_sha256'] or sha256_file(a.class_map)!=protocol['class_map_sha256']:
        raise ValueError('frozen construction inputs changed')
    source_run=json.loads((a.source/'run.json').read_text())
    if not source_run.get('completed') or sha256_file(a.source/'index.jsonl')!=source_run['index_sha256']:
        raise ValueError('source construction incomplete or changed')
    rows=read_jsonl(a.source/'index.jsonl')
    if len(rows)!=protocol['videos'] or len({r['video_uid'] for r in rows})!=len(rows) or a.workers<1:
        raise ValueError('invalid candidate population or workers')
    out=new_output(a.output);started=time.time()
    run={'started_unix':started,'protocol_sha256':sha256_file(a.protocol),'workers':a.workers,
         'source_construction_run_sha256':sha256_file(a.source/'run.json'),
         'source_sha256':{name:sha256_file(Path(__file__).with_name(name)) for name in
                         ('recover_subject_construction.py','region_discrimination.py','subject_artifacts.py')}}
    write_json(out/'run.json',run);write_json(out/'protocol.json',protocol)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(recover_entry,[(r,a.source,out,a.video_root,mapping) for r in rows]))
    write_jsonl(out/'index.jsonl',results)
    summary={'total_bases':len(results),'accepted':sum(r['status']=='accepted' for r in results),
             'rejected':sum(r['status']!='accepted' for r in results),
             'base_rejection_counts':dict(Counter(x for r in results for x in r['rejection_reasons']))}
    write_json(out/'summary.json',summary)
    run.update(completed=True,finished_unix=time.time(),wall_seconds=time.time()-started,index_sha256=sha256_file(out/'index.jsonl'))
    write_json(out/'run.json',run);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
