"""Validate and merge completed independent Subject stability scoring shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json, write_jsonl


def merge(source: Path, output: Path):
    records, runs, protocols = [], [], []
    keys = ('protocol_sha256','dataset_index_sha256','source_sha256','num_shards','origin_reference')
    for path in sorted(source.glob('shard*')):
        if not path.is_dir(): continue
        run = json.loads((path/'run.json').read_text())
        if not run.get('completed') or sha256_file(path/'scores.jsonl') != run['scores_sha256']:
            raise ValueError('incomplete or changed shard')
        protocol = json.loads((path/'protocol.json').read_text())
        if runs and (any(run[k] != runs[0][k] for k in keys) or protocol != protocols[0]):
            raise ValueError('incompatible shards')
        rows = read_jsonl(path/'scores.jsonl')
        if len(rows) != run['base_count']:
            raise ValueError('incomplete shard population')
        records.extend({**row,'source_shard':path.name} for row in rows)
        runs.append(run);protocols.append(protocol)
    if not runs or {r['shard_index'] for r in runs} != set(range(runs[0]['num_shards'])) or len(runs) != runs[0]['num_shards']:
        raise ValueError('missing or duplicate shard')
    if len(records) != protocols[0]['cohort_candidates'] or len({r['base']['video_uid'] for r in records}) != len(records):
        raise ValueError('incomplete or duplicate video cohort')
    records.sort(key=lambda r:r['base']['video_uid'])
    out = new_output(output)
    write_jsonl(out/'scores.jsonl',records);write_json(out/'protocol.json',protocols[0])
    run = {**runs[0], 'completed':True, 'shard_index':None, 'base_count':len(records),
           'completed_bases':sum(r['completed_bases'] for r in runs),
           'runtime_failures':sum(r['runtime_failures'] for r in runs),
           'maximum_origin_reference_error':max(r['maximum_origin_reference_error'] for r in runs),
           'physical_gpu':[r['physical_gpu'] for r in runs],
           'started_unix':min(r['started_unix'] for r in runs),
           'finished_unix':max(r['finished_unix'] for r in runs),
           'wall_seconds':max(r['finished_unix'] for r in runs)-min(r['started_unix'] for r in runs),
           'shards':runs, 'source_root':str(source.resolve()),
           'artifact_paths':'scoring mask paths are relative to their original shard; masks are not copied',
           'merge_source_sha256':sha256_file(Path(__file__)),
           'scores_sha256':sha256_file(out/'scores.jsonl')}
    write_json(out/'run.json',run)
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=merge(a.source,a.output)
    print(json.dumps({k:r[k] for k in ('completed_bases','runtime_failures','wall_seconds','maximum_origin_reference_error')}))


if __name__=='__main__':main()
