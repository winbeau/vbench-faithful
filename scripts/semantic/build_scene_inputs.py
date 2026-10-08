#!/usr/bin/env python3
"""Attach the exact same cached 16 captions to every eligible Scene text variant.

Original and synonym annotation-plan inputs can be appended without consuming any
labels. Missing evidence remains a 16-frame placeholder for downstream coverage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/prompt-compiler/src'))
sys.path.insert(0, str(ROOT))
from scripts.semantic.paths import DATA_ROOT, RAW_ROOT, FIXTURES_ROOT
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile.records import write_jsonl
from scripts.semantic.construction import build_conditions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--annotation-plan')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    matrix = J.read_jsonl(Path(args.matrix))
    cache = {r['relative_path']: r for r in J.read_jsonl(Path(args.cache))}
    rows = []
    for row in matrix:
        if row['task'] != 'scene':
            continue
        evidence = cache.get(row['relative_path'], {})
        captions = evidence.get('frame_captions') if evidence.get('status') == 'ok' else None
        if not row['eligible']:
            captions = None
        rows.append({**row, 'frame_captions': captions, 'cache_status': evidence.get('status', 'missing'),
                     'video_sha256': evidence.get('video_sha256'), 'frame_sha256': evidence.get('frame_sha256')})
    if args.annotation_plan:
        conditions, _ = build_conditions([], plan_rows=J.read_jsonl(Path(args.annotation_plan)))
        rows.extend({k: r[k] for k in ['prompt', 'caption', 'source_id']} for r in conditions)
    paths = {'matrix': args.matrix, 'cache': args.cache, 'annotation_plan': args.annotation_plan}
    identity = {'inputs_sha256': {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in paths.items() if v},
        'rows': len(rows), 'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    J.bind_job(out.with_suffix('.manifest.json'), identity)
    write_jsonl(out, rows)
    print(json.dumps(identity, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
