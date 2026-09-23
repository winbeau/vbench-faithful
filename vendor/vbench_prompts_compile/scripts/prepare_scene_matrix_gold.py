#!/usr/bin/env python3
"""Unlabelled census of the already-frozen Scene synonym domain, all 16 frames.

This supplementary annotation plan is prepared after the initial test2 scorecard.
Selection uses only the previously frozen synonym eligibility, never predictions,
labels or scores. It changes neither test2 nor any training/decoding configuration.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile import records as R


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True)
    p.add_argument('--video-root',default='/root/wenbiao_zhao/datasets/vbench-1.0-human-preference')
    p.add_argument('--out',required=True)
    args=p.parse_args(argv)
    source=Path(args.inputs);out=Path(args.out)
    selected=[r for r in J.read_jsonl(source) if r.get('transform')=='synonym' and r.get('eligible')]
    rows=[]
    for row in selected:
        captions=row.get('frame_captions')
        if not captions or len(captions)!=16:
            raise ValueError('supplement requires a complete original caption cache')
        for frame,caption in enumerate(captions):
            identity=[row['original_prompt'],row['relative_path'],frame]
            rows.append({'sample_id':'scene-matrix-'+R.sha256_text(R.canonical_json(identity))[:24],
                'task':'scene','input':{'prompt':row['original_prompt'],'caption':caption},'target':None,
                'source':'vbench_matrix_scene_caption','source_id':row['relative_path']+':'+str(frame),
                'group_id':row['family'],'quality':'teacher_candidate_unreviewed',
                'meta':{'visual_truth':None,'split':'test','pair_type':'original_synonym_domain',
                    'frame_video':str(Path(args.video_root)/row['relative_path']),'frame_index':frame,
                    'frame_source_prompt':row['original_prompt'],'image':None,
                    'matrix_sample_id':row['sample_id'],'video_sha256':row['video_sha256'],
                    'frame_sha256':row['frame_sha256'][frame],
                    'licence':'VBench 1.0 public human-preference videos; official Tag2Text captions'}})
    if len({J.observation_id(r) for r in rows})!=len(rows):
        raise ValueError('duplicate planned observation')
    manifest={'inputs_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'videos':len(selected),'observations':len(rows),'families':len({r['group_id'] for r in rows}),
        'unique_texts':len({(r['input']['prompt'],r['input']['caption']) for r in rows}),
        'by_prompt':dict(Counter(r['input']['prompt'] for r in rows)),
        'scope':'supplementary census of the preexisting frozen strict-synonym domain; planned after initial test2 scorecard, no prediction-driven selection'}
    J.bind_job(out.with_suffix('.manifest.json'),manifest)
    R.write_jsonl(out,rows)
    print(json.dumps(manifest,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
