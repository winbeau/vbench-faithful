"""Merge disjoint construction shards in official manifest order."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from .common import ROOT, sha256_file
from .subject_artifacts import new_output, read_jsonl, write_json


def validate_protocol(source_path, expected_sha, shards):
    # Each shard stores a canonical JSON copy; the run hash pins the original
    # preregistration bytes. Compare values for the copies, bytes for the source.
    if sha256_file(source_path) != expected_sha:
        raise ValueError('source construction protocol changed')
    protocol = json.loads(source_path.read_text())
    if any(json.loads((root/'protocol.json').read_text()) != protocol for root in shards):
        raise ValueError('shard protocol changed')
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/background-repair/construction_refined_dev_v3.json')
    args = parser.parse_args()
    runs = [json.loads((root/'run.json').read_text()) for root in args.shards]
    reference = runs[0]
    count = reference['shard']['count']
    if len(runs) != count or {run['shard']['index'] for run in runs} != set(range(count)):
        raise ValueError('incomplete or duplicate shard set')
    shared = ['protocol_sha256', 'input_manifest_sha256', 'source_sha256', 'localizer', 'semantic_cache', 'upstream']
    combined = {}
    for root, run in zip(args.shards, runs):
        if not run.get('completed') or run['index_sha256'] != sha256_file(root/'index.jsonl'):
            raise ValueError('incomplete or modified shard')
        if any(run.get(key) != reference.get(key) for key in shared):
            raise ValueError('shards do not share construction provenance')
        for entry in read_jsonl(root/'index.jsonl'):
            uid = entry['video_uid']
            if uid in combined:
                raise ValueError('duplicate video across shards')
            if sha256_file(root/entry['manifest']) != entry['manifest_sha256']:
                raise ValueError('shard manifest changed')
            combined[uid] = entry
    protocol = validate_protocol(args.protocol, reference['protocol_sha256'], args.shards)
    manifest = ROOT/'configs/background-repair/natural1720_manifest_v2.jsonl'
    if sha256_file(manifest) != reference['input_manifest_sha256']:
        raise ValueError('input manifest changed')
    expected = [row for row in read_jsonl(manifest) if row['split'] == protocol['split']]
    if set(combined) != {row['video_uid'] for row in expected} or len(combined) != protocol['candidate_count']:
        raise ValueError('merged input coverage mismatch')
    output = new_output(args.output)
    for root in args.shards:
        for directory in ('manifests', 'construction_masks', 'clips'):
            if (root/directory).exists():
                # Artifacts are immutable after each shard completes; links are
                # validated by the independent pixel replay after this merge.
                shutil.copytree(root/directory, output/directory, dirs_exist_ok=True,
                                copy_function=lambda source, target: Path(target).hardlink_to(source))
    shutil.copyfile(args.protocol, output/'protocol.json')
    ordered = [combined[row['video_uid']] for row in expected]
    (output/'index.jsonl').write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in ordered))
    counts = dict(Counter(row['status'] for row in ordered))
    run = {key: value for key, value in reference.items() if key != 'shard'}
    run.update(total=len(ordered), started_unix=min(r['started_unix'] for r in runs),
               finished_unix=max(r['finished_unix'] for r in runs), completed=True, counts=counts,
               index_sha256=sha256_file(output/'index.jsonl'), merger_sha256=sha256_file(Path(__file__)),
               shards=[{'path': str(root), 'run_sha256': sha256_file(root/'run.json')} for root in args.shards])
    write_json(output/'run.json', run)
    write_json(output/'progress.json', {'processed': len(ordered), 'total': len(ordered), 'counts': counts,
                                      'elapsed_seconds': run['finished_unix']-run['started_unix']})
    print(json.dumps({'total': len(ordered), 'counts': counts, 'output': str(output)}))


if __name__ == '__main__':
    main()
