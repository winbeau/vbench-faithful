#!/usr/bin/env python3
"""Cache locked VBench visual evidence once, with durable per-video recovery.

Scene: all 16 Tag2Text captions; Objects: exact GRiT det_obj label sets;
Spatial: GRiT descriptions/boxes and actual resized frame dimensions;
Action: UMT sigmoid top-5, IDs, labels, raw/rounded confidence.

The official scoring environment is separate from the project training environment.
No scoring decisions or Repair predictions influence video selection.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile.official_replay import UPSTREAM_SHA, action_filename_label

DIMENSIONS = {'spatial': 'spatial_relationship', 'objects': 'multiplt_object', 'action': 'human_action', 'scene': 'scene'}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dimension', required=True, choices=DIMENSIONS)
    p.add_argument('--manifest', required=True)
    p.add_argument('--video-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--vbench-root', default='/root/wenbiao_zhao/VBench')
    p.add_argument('--grit-weights', default='/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth')
    p.add_argument('--tag2text-weights', default='/root/.cache/vbench/caption_model/tag2text_swin_14m.pth')
    p.add_argument('--umt-weights', default='/root/.cache/vbench/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth')
    p.add_argument('--num-frames', type=int, default=16)
    p.add_argument('--limit', type=int)
    p.add_argument('--per-prompt', type=int)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--frozen-split')
    p.add_argument('--split', choices=['dev', 'test', 'all'], default='all')
    p.add_argument('--max-videos', type=int, help='new videos this invocation; preserve full manifest')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args(argv)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows_from_manifest(path, limit=None, per_prompt=None):
    by_prompt = {}
    with Path(path).open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            prompt = (row.get('prompt_en') or '').strip()
            if not prompt:
                continue
            videos = by_prompt.setdefault(prompt, [])
            for key in ['video_a_path', 'video_b_path']:
                relative = row.get(key)
                if relative and relative not in videos:
                    videos.append(relative)
    cap = max((len(v) for v in by_prompt.values()), default=0)
    if per_prompt is not None:
        cap = min(cap, per_prompt)
    selected = []
    for i in range(cap):
        for prompt, videos in by_prompt.items():
            if i < len(videos):
                selected.append({'prompt': prompt, 'relative_path': videos[i]})
    return selected[:limit] if limit is not None else selected


def selected_rows(args):
    rows = rows_from_manifest(args.manifest, None, args.per_prompt)
    if args.split != 'all':
        if not args.frozen_split:
            raise ValueError('--split requires --frozen-split')
        with Path(args.frozen_split).open() as f:
            allowed = {r['prompt_id'].strip() for r in csv.DictReader(f)
                       if r['dimension'] == DIMENSIONS[args.dimension] and r['split'] == args.split}
        rows = [r for r in rows if r['prompt'] in allowed]
    return rows[:args.limit] if args.limit is not None else rows


def initialize_umt(args, device):
    # Import the pinned official module: it registers the exact timm architecture.
    import torch
    from vbench import human_action as upstream
    model = upstream.create_model('vit_large_patch16_224', pretrained=False, num_classes=400,
        all_frames=16, tubelet_size=1, use_learnable_pos_emb=False, fc_drop_rate=0., drop_rate=0.,
        drop_path_rate=0.2, attn_drop_rate=0., drop_block_rate=None, use_checkpoint=False,
        checkpoint_num=16, use_mean_pooling=True, init_scale=0.001)
    state = torch.load(args.umt_weights, map_location='cpu')
    loading = model.load_state_dict(state, strict=False)
    transform = upstream.Compose([upstream.Resize(256, interpolation='bilinear'),
        upstream.CenterCrop(size=(224, 224)), upstream.ClipToTensor(),
        upstream.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    print(json.dumps({'umt_missing_keys': loading.missing_keys, 'umt_unexpected_keys': loading.unexpected_keys}), flush=True)
    return model.to(device).eval(), transform, upstream.build_dict()


def run(args, out):
    rows = selected_rows(args)
    if not rows:
        raise ValueError('empty video selection')
    if args.num_frames != 16:
        raise ValueError('official matrix freezes 16 frames; alternate frame counts need a separate protocol')
    upstream = subprocess.check_output(['git', '-C', args.vbench_root, 'rev-parse', 'HEAD'], text=True).strip()
    if upstream != UPSTREAM_SHA:
        raise ValueError('upstream commit differs from frozen protocol')
    weight = {'scene': args.tag2text_weights, 'spatial': args.grit_weights, 'objects': args.grit_weights, 'action': args.umt_weights}[args.dimension]
    identity = {'schema': 2, 'dimension': args.dimension, 'videos': rows, 'num_frames': args.num_frames,
                'upstream_sha': upstream, 'weights_sha256': sha(weight), 'manifest_sha256': sha(args.manifest),
                'split_sha256': sha(args.frozen_split) if args.frozen_split else None, 'split': args.split}
    J.bind_job(out.with_suffix('.manifest.json'), identity)
    print(json.dumps({'planned_videos': len(rows), 'families': len({r['prompt'] for r in rows}), 'dimension': args.dimension}), flush=True)
    if args.dry_run:
        return 0
    journal = out.with_suffix('.attempts.jsonl')
    latest = {}
    for entry in J.read_jsonl(out) + J.read_jsonl(journal):
        if entry.get('status') != 'not_run':
            latest[entry['relative_path']] = entry
    pending = [r for r in rows if r['relative_path'] not in latest or (args.retry_failed and latest[r['relative_path']]['status'] != 'ok')]
    if args.max_videos is not None:
        pending = pending[:args.max_videos]
    if pending:
        sys.path.insert(0, str(Path(args.vbench_root).resolve()))
        import torch
        import torchvision.transforms as tv
        from vbench.utils import load_video
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('requested CUDA backend unavailable')
        device = torch.device(args.device)
        if args.dimension == 'scene':
            from vbench.third_party.tag2Text.tag2text import tag2text_caption
            from vbench.utils import tag2text_transform
            model = tag2text_caption(pretrained=args.tag2text_weights, image_size=384, vit='swin_b').to(device).eval()
            transform = tag2text_transform(384)
        elif args.dimension == 'action':
            model, transform, classes = initialize_umt(args, device)
        else:
            from vbench.third_party.grit_model import DenseCaptioning
            model = DenseCaptioning(device)
            model.initialize_model_det(model_weight=args.grit_weights)
        for index, row in enumerate(pending):
            path = Path(args.video_root) / row['relative_path']
            entry = {**row, 'video_path': str(path), 'dimension': args.dimension,
                     'video_uid': hashlib.sha256(row['relative_path'].encode()).hexdigest()[:24]}
            try:
                entry['video_sha256'] = sha(path)
                if args.dimension == 'scene':
                    frames = load_video(str(path), num_frames=16, return_tensor=False, width=384, height=384)
                    tensor = torch.cat([transform(frame).unsqueeze(0) for frame in frames]).to(device)
                    with torch.no_grad():
                        captions, _ = model.generate(tensor, tag_input=None, return_tag_predict=True)
                    entry['frame_captions'] = list(captions)
                    entry['frame_sha256'] = [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]
                elif args.dimension == 'action':
                    inputs = load_video(str(path), transform, num_frames=16).unsqueeze(0)
                    entry['model_input_sha256'] = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
                    with torch.no_grad():
                        scores, ids = torch.topk(torch.sigmoid(model(inputs.to(device))), 5, dim=1)
                    entry['top5'] = [{'class_id': int(i), 'label': classes[str(i)], 'score': float(s), 'rounded_score': round(float(s), 4)}
                                     for i, s in zip(ids.squeeze().tolist(), scores.squeeze().tolist())]
                    entry['official_filename_label'] = action_filename_label(str(path))
                else:
                    video = load_video(str(path), num_frames=16)
                    _, _, h, w = video.size()
                    entry['original_frame_size'] = [int(w), int(h)]
                    if min(h, w) > 768:
                        scale = 720. / min(h, w)
                        video = tv.Resize(size=(int(scale * h), int(scale * w)))(video)
                    frames = video.permute(0, 2, 3, 1).numpy()
                    entry['frame_size'] = [int(frames.shape[2]), int(frames.shape[1])]
                    entry['frame_sha256'] = [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]
                    detections, labels = [], []
                    with torch.no_grad():
                        for frame in frames:
                            ret = model.run_caption_tensor(frame)
                            detections.append([{'label': item[0], 'box': [float(v) for v in item[1][:4]]} for item in ret[0]])
                            labels.append(sorted(set(str(v) for v in ret[0][0][2])) if ret[0] else [])
                    entry['frame_detections'], entry['frame_labels'] = detections, labels
                entry['status'] = 'ok'
            except Exception as error:
                entry.update(status=f'error:{type(error).__name__}', error=str(error)[:300])
            J.append_jsonl(journal, entry)
            latest[row['relative_path']] = entry
            print(json.dumps({'processed_this_run': index + 1, 'cached': len(latest), 'planned': len(rows), 'last_status': entry['status']}), flush=True)
    # Export one row per frozen video, including missing rows. The append-only
    # attempt journal remains authoritative if interrupted before this export.
    records = [latest.get(r['relative_path'], {**r, 'dimension': args.dimension, 'status': 'not_run'}) for r in rows]
    temp = out.with_suffix('.jsonl.tmp')
    with temp.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    temp.replace(out)
    status = dict(Counter(r['status'] for r in records))
    report = {'planned': len(rows), 'status': status, 'complete': status.get('ok', 0) == len(rows),
              'dimension': args.dimension, 'upstream_sha': upstream, 'weights_sha256': identity['weights_sha256']}
    J.atomic_json(out.with_suffix('.report.json'), report)
    print(json.dumps(report), flush=True)
    return 0 if report['complete'] else 2


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    with J.job_lock(out.with_suffix('.lock')):
        return run(args, out)


if __name__ == '__main__':
    raise SystemExit(main())
