#!/usr/bin/env python3
"""Partition an unchanged visibility census, then merge without losing failures.

The initial prefix remains owned by the pilot. Other workers have disjoint IDs,
use the same annotation script and share one process-locked request budget.
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
from vbench_prompts_compile.records import write_jsonl
from cache_matrix_transforms import evidence_id
from cache_backend_outputs import sha


def endpoints(rows):
    result=[r for r in rows if r['task']=='objects' and r.get('occlusion_control')=='target' and r.get('level')==1]
    if len({evidence_id(r) for r in result})!=len(result):
        raise ValueError('duplicate endpoint IDs')
    return result


def merge(rows, sources):
    planned={evidence_id(r):r for r in endpoints(rows)}
    done={}
    for source in sources:
        for row in source:
            key=row['evidence_id']
            if key not in planned or key in done:
                raise ValueError('unplanned or duplicate visibility ID: '+key)
            done[key]=row
    output=[]
    for key,row in planned.items():
        output.append(done.get(key,{'evidence_id':key,'relative_path':row['relative_path'],
            'status':'annotation_not_run','endpoint_invisible_verified':False,'removal_verified':False,
            'human_reviewed':False,'method':None}))
    incomplete={'annotation_not_run','transform_unavailable'}
    report={'planned':len(planned),'completed':sum(r['status'] not in incomplete for r in output),
        'complete':all(r['status'] not in incomplete for r in output),
        'status':dict(Counter(r['status'] for r in output)),
        'endpoint_invisible_verified':sum(bool(r.get('endpoint_invisible_verified')) for r in output),
        'removal_verified':sum(bool(r.get('removal_verified')) for r in output),
        'note':'All frozen videos retained. Multiple passes use the same vision model; this is model consensus, not human gold.'}
    return output,report


def request_usage(entries):
    responses=[r['response'] for r in entries if 'response' in r]
    if any(not isinstance(r,dict) for r in responses):
        raise ValueError('malformed response journal entry')
    return {'journal_entries':len(entries),'successful_responses':len(responses),
        'failed_requests':sum('error' in r and 'response' not in r for r in entries),
        'entries_without_response':len(entries)-len(responses),
        'prompt_tokens':sum(r.get('prompt_tokens',0) for r in responses),
        'completion_tokens':sum(r.get('completion_tokens',0) for r in responses),'currency_cost':None}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('plan');a.add_argument('--plan',required=True);a.add_argument('--pilot',required=True)
    a.add_argument('--reserved',type=int,default=10);a.add_argument('--workers',type=int,default=8);a.add_argument('--out',required=True)
    b=sub.add_parser('merge');b.add_argument('--plan',required=True);b.add_argument('--input',action='append',required=True)
    b.add_argument('--out',required=True);b.add_argument('--budget-state')
    args=p.parse_args(argv);rows=J.read_jsonl(Path(args.plan))
    if args.command=='plan':
        selected=endpoints(rows)
        if not 0<=args.reserved<=len(selected) or args.workers<1:raise ValueError('invalid partition')
        pilot_ids={evidence_id(r) for r in selected[:args.reserved]}
        if not {r['evidence_id'] for r in J.read_jsonl(Path(args.pilot))}<=pilot_ids:
            raise ValueError('pilot went outside reserved prefix')
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        assignments={'pilot':sorted(pilot_ids)}
        for i in range(args.workers):
            values=selected[args.reserved+i::args.workers]
            path=out/f'worker-{i}.jsonl'
            if path.exists():raise ValueError('partition already exists')
            write_jsonl(path,values);assignments[str(path)]=[evidence_id(r) for r in values]
        J.atomic_json(out/'manifest.json',{'source_plan_sha256':sha(args.plan),'planned':len(selected),
            'workers':args.workers,'reserved_for_pilot':args.reserved,'assignments':assignments,
            'annotation_script_sha256':sha(ROOT/'scripts/annotate_object_visibility.py'),
            'note':'Operational partition only; all workers use unchanged labels, images, prompts, model and shared budget.'})
        print(json.dumps({'planned':len(selected),'pilot':args.reserved,'workers':args.workers}))
        return 0
    output,report=merge(rows,[J.read_jsonl(Path(v)) for v in args.input])
    report['input_sha256']={v:sha(v) for v in [args.plan,*args.input]}
    if args.budget_state:report['budget']=json.loads(Path(args.budget_state).read_text())
    journals=[Path(v).with_suffix(f'.requests-pass-{n}.jsonl') for v in args.input for n in [1,2,3]]
    report['usage']=request_usage([r for v in journals for r in J.read_jsonl(v)])
    report['journal_sha256']={str(v):sha(v) for v in journals if v.is_file()}
    write_jsonl(Path(args.out),output);J.atomic_json(Path(args.out).with_suffix('.report.json'),report)
    print(json.dumps(report,indent=2))
    return 0 if report['complete'] else 2


if __name__=='__main__':raise SystemExit(main())
