"""Expanded background intervention experiment, with automatic scoring masks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from .build_region_discrimination import verify
from .common import ROOT, sha256_file
from .generate_subject_masks import deterministic_setup
from .score_region_discrimination import ClipExtractor
from .subject_artifacts import artifact_path, new_output, read_jsonl, read_png_sequence, write_json, write_npz


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','output','detector-checkpoint','mobilesam-checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--dino-repo',required=True);parser.add_argument('--dino-weight',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--num-shards',type=int,default=1);parser.add_argument('--shard-index',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.shard_index<args.num_shards:parser.error('invalid shard')
    config_root=ROOT/'configs/subject-repair'
    protocol=json.loads((config_root/'background288_protocol.json').read_text())
    method=json.loads((config_root/'natural1440_protocol_v2.json').read_text())
    if sha256_file(config_root/'natural1440_protocol_v2.json')!=protocol['method_config_sha256']:
        raise ValueError('method configuration changed')
    manifest=config_root/'background288_manifest.jsonl'
    if sha256_file(manifest)!=protocol['manifest_sha256']:raise ValueError('candidate manifest changed')
    bases=read_jsonl(manifest); index=read_jsonl(args.dataset/'index.jsonl')
    if len(index)!=288 or {r['video_uid'] for r in index}!={r['video_uid'] for r in bases}:
        raise ValueError('dataset is not the full preregistered cohort')
    integrity=verify(args.dataset)
    import numpy as np
    import torch
    from subject_consistency.automatic_localizer import CocoSubjectBoxDetector, AutomaticMobileSamSubjectMaskProvider
    from subject_consistency.localizer import load_mobile_sam_predictor
    from subject_consistency.metric import build_dino_config, evaluate_masked_batch, subject_consistency_diagnostics, upstream_path
    from subject_consistency.backends.vbench import official_diagnostics
    from subject_consistency.models import OfficialDinoPatchExtractor

    deterministic_setup();torch.set_num_threads(2)
    output=new_output(args.output);started=time.time()
    assets=json.loads((config_root/'assets.lock.json').read_text())
    if sha256_file(args.mobilesam_checkpoint)!=assets['mobilesam']['checkpoint_sha256']:
        raise ValueError('SAM checkpoint changed')
    if sha256_file(args.dino_weight)!=method['dino_weight_sha256']:
        raise ValueError('DINO checkpoint changed')
    detector=CocoSubjectBoxDetector(args.detector_checkpoint,expected_sha256=method['box_detector']['weight_sha256'],
        device=args.device,threshold=method['box_detector']['threshold'],size=method['box_detector']['size'])
    provider=AutomaticMobileSamSubjectMaskProvider(detector,load_mobile_sam_predictor(args.mobilesam_checkpoint,args.device),
        weights_sha256=assets['mobilesam']['checkpoint_sha256'])
    extractor=OfficialDinoPatchExtractor(args.device,build_dino_config(args.dino_repo,args.dino_weight),upstream_path())
    entries=index[args.shard_index::args.num_shards]
    run={'stage':protocol['stage'],'protocol_sha256':sha256_file(config_root/'background288_protocol.json'),
         'method_config_sha256':protocol['method_config_sha256'],'dataset_index_sha256':sha256_file(args.dataset/'index.jsonl'),
         'shard_index':args.shard_index,'num_shards':args.num_shards,'base_count':len(entries),
         'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'device':args.device,'started_unix':started,
         'localizer':provider.provenance,'upstream':vars(extractor.upstream_state),'integrity':integrity,
         'source_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for directory in
             (ROOT/'scripts/counterfactual',ROOT/'metrics/subject-consistency/src') for p in directory.rglob('*.py')}}
    write_json(output/'run.json',run);write_json(output/'protocol.json',protocol)
    count=failures=accepted=0
    for entry in entries:
        source=json.loads(artifact_path(args.dataset,entry['manifest']).read_text())
        base=source['base'];bid=base['base_id']
        row={'base':base,'construction_status':source['status'],'rejection_reasons':source['rejection_reasons'],
             'variants':{},'status':'construction_rejected'}
        if source['status']=='accepted':
            accepted+=1
            try:
                for variant,files in source['variants'].items():
                    frames=torch.from_numpy(read_png_sequence(args.dataset,files)).permute(0,3,1,2)
                    value={'scores':{}}
                    features=extractor.features_from_frames(frames)
                    for name,fn in [('official',official_diagnostics),('aggregation',subject_consistency_diagnostics)]:
                        value['scores'][name]={'status':'succeeded','score':fn(features).final_score}
                    masks=provider.masks_for(Path(bid),frames,base['subject_en'])
                    value['localizer_diagnostics']=provider.last_diagnostics
                    path=output/'scoring_masks'/f"{bid}__{variant.replace('/','__')}.npz"
                    write_npz(path,masks=masks.instance_masks.numpy().astype('uint8'),present=masks.instance_present.numpy(),
                        metadata_json=json.dumps({**provider.provenance,'source_frame_sha256':files[0]['sha256']},sort_keys=True))
                    value['scoring_mask_file']={'path':str(path.relative_to(output)),'sha256':sha256_file(path)}
                    class FixedMasks:
                        def masks_for(self,*unused):return masks
                    for mode in method['representations']:
                        adapter=ClipExtractor(extractor,frames)
                        for policy in ('zero','exclude'):
                            result=evaluate_masked_batch([Path(bid)],{bid:{'subject_en':base['subject_en']}},args.device,{},
                                FixedMasks(),extractor=adapter,encoding_mode=mode,missing_policy=policy)[0]
                            value['scores'][mode+'_'+policy]={k:result[k] for k in ('status','score','failure_reason','diagnostics')}
                    row['variants'][variant]=value
                row['status']='completed'
            except Exception as exc:
                row.update(status='failed',failure_reason=f'{type(exc).__name__}: {exc}');failures+=1
        with (output/'scores.jsonl').open('a') as handle:
            handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
        count+=1
        progress={'processed':count,'total':len(entries),'construction_accepted':accepted,'runtime_failures':failures,
                  'elapsed_s':time.time()-started}
        write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    run.update(completed=True,finished_unix=time.time(),wall_seconds=time.time()-started,
               runtime_failures=failures,scores_sha256=sha256_file(output/'scores.jsonl'))
    write_json(output/'run.json',run)
    return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
