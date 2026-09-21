"""Independent CPU construction shards with an explicit merged artifact index."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

from .common import sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, write_json, write_jsonl


def worker(arguments):
    rows, video_root, output, model_dir, mapping = arguments
    import torch
    from .generate_subject_masks import deterministic_setup, SegFormerConstructionLocalizer, generate
    torch.set_num_threads(1); deterministic_setup()
    localizer = SegFormerConstructionLocalizer(model_dir, device='cpu')
    return generate(rows, video_root, new_output(output), mapping, localizer)


def merge_shards(output, expected):
    rows = []
    seen = set()
    for shard in sorted(output.glob('shard*')):
        for row in read_jsonl(shard/'index.jsonl'):
            uid = row['video_uid']
            if uid in seen or uid not in expected:
                raise ValueError('duplicate/unexpected construction input')
            seen.add(uid)
            source = artifact_path(shard, row['manifest']); entry = json.loads(source.read_text())
            if entry['base'] != expected[uid]:
                raise ValueError('construction input identity changed')
            entry['parallel_source_manifest_sha256'] = sha256_file(source)
            if 'mask_file' in entry:
                artifact_path(shard, entry['mask_file']['path'])
                entry['mask_file']['path'] = str(Path(shard.name)/entry['mask_file']['path'])
            for frame in entry.get('frames', []):
                artifact_path(shard, frame['path'])
                frame['path'] = str(Path(shard.name)/frame['path'])
            relative = 'manifests/'+entry['base']['base_id']+'.json'
            write_json(output/relative, entry)
            rows.append({**row, 'manifest': relative})
    if seen != set(expected):
        raise ValueError('incomplete construction shards')
    rows.sort(key=lambda r:r['video_uid'])
    write_jsonl(output/'index.jsonl', rows)
    summary = {'total_bases': len(rows), 'accepted': sum(r['status']=='accepted' for r in rows),
               'rejected': sum(r['status']!='accepted' for r in rows),
               'base_rejection_counts': dict(Counter(x for r in rows for x in r['rejection_reasons']))}
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bases','protocol','class-map','model-dir','video-root','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if sha256_file(args.bases) != protocol['manifest_sha256'] or sha256_file(args.class_map) != protocol['class_map_sha256']:
        raise ValueError('frozen construction inputs changed')
    rows = read_jsonl(args.bases)
    if args.workers < 1 or len(rows) != protocol['videos'] or len({r['video_uid'] for r in rows}) != len(rows):
        raise ValueError('invalid worker count or cohort')
    output = new_output(args.output); started = time.time()
    run = {'protocol_sha256':sha256_file(args.protocol),'manifest_sha256':sha256_file(args.bases),
           'class_map_sha256':sha256_file(args.class_map),'workers':args.workers,'torch_threads_per_worker':1,
           'device':'cpu','started_unix':started,'source_sha256':{
               name:sha256_file(Path(__file__).with_name(name)) for name in
               ('generate_subject_masks_parallel.py','generate_subject_masks.py','region_discrimination.py','subject_artifacts.py')}}
    write_json(output/'run.json',run);write_json(output/'protocol.json',protocol)
    mapping=json.loads(args.class_map.read_text())
    arguments=[(rows[i::args.workers],args.video_root,output/f'shard{i}',args.model_dir,mapping) for i in range(args.workers)]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, result in enumerate(pool.map(worker,arguments)):
            print(json.dumps({'shard':i,**result}),flush=True)
    summary=merge_shards(output,{r['video_uid']:r for r in rows})
    run.update(completed=True,wall_seconds=time.time()-started,finished_unix=time.time(),index_sha256=sha256_file(output/'index.jsonl'))
    write_json(output/'run.json',run);print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
