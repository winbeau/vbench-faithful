#!/usr/bin/env python3
"""Compare cached UMT decisions with the locked upstream's actual GPU entry point.

Use the first sorted positive and negative cached example per generator to cover
both decision branches. This is a stratified parity check, not a quality estimate.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile.official_replay import UPSTREAM_SHA,action_score,action_filename_label
from cache_backend_outputs import sha


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--vbench-root',default='/root/wenbiao_zhao/VBench')
    p.add_argument('--weights',default='/root/.cache/vbench/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth')
    p.add_argument('--device',default='cuda:0');args=p.parse_args(argv)
    root=Path(args.vbench_root)
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if commit!=UPSTREAM_SHA:raise ValueError('upstream revision changed')
    for name in ['vbench/human_action.py','vbench/utils.py']:
        if (root/name).read_bytes()!=subprocess.check_output(['git','-C',str(root),'show','HEAD:'+name]):
            raise ValueError('upstream working file changed: '+name)
    selected={}
    for row in sorted(J.read_jsonl(Path(args.cache)),key=lambda r:r['relative_path']):
        if row['status']!='ok':continue
        cached=action_score(action_filename_label(row['video_path']),row['top5'])
        selected.setdefault((row['relative_path'].split('/')[1],cached),row)
    if len(selected)!=8:raise ValueError('expected two cached decision branches for each of four generators')
    for row in selected.values():
        if sha(row['video_path'])!=row['video_sha256']:raise ValueError('source video changed')
    sys.path.insert(0,str(root))
    from vbench.human_action import human_action
    score,actual=human_action(args.weights,[r['video_path'] for r in selected.values()],args.device)
    cached_by_path={r['video_path']:action_score(action_filename_label(r['video_path']),r['top5']) for r in selected.values()}
    comparisons=[{**r,'cache_replay':cached_by_path[r['video_path']],
                  'equal':int(r['video_results'])==cached_by_path[r['video_path']]} for r in actual]
    report={'upstream_commit':commit,'cache_sha256':sha(args.cache),'weights_sha256':sha(args.weights),
        'script_sha256':sha(__file__),'comparisons':comparisons,'passed':all(r['equal'] for r in comparisons),
        'official_mean':score,'selection':'first sorted cached positive/negative per generator; parity only',
        'upstream_source_sha256':{name:sha(root/name) for name in ['vbench/human_action.py','vbench/utils.py']}}
    J.atomic_json(Path(args.out),report);print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
