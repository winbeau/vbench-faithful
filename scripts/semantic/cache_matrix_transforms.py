#!/usr/bin/env python3
"""Cache physical video mirrors and controlled occlusions on the full frozen plan.

Every source/variant retains all 16 frame slots. Per-frame missing localization,
impossible controls, failed pixel parity and missing backend results remain in the
denominator. Geometry coverage never substitutes for independent visibility labels.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/prompt-compiler/src'))
sys.path.insert(0, str(ROOT))
from scripts.semantic.paths import DATA_ROOT, RAW_ROOT, FIXTURES_ROOT
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile.official_replay import UPSTREAM_SHA
from vbench_prompts_compile.visual_transforms import mirror_detections, patch_frame
from scripts.semantic.construction import sha


class BackendUnavailable(RuntimeError):
    """Stop a batch on model initialization failure instead of repeating it."""


def evidence_id(row):
    fields = [row['task'], row['relative_path'], row['evidence_kind'], row.get('axis'),
              row.get('occlusion_control'), row.get('level')]
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest()[:24]


def unique_plan(rows):
    result = {}
    for row in rows:
        if row['evidence_kind'] == 'original':
            continue
        key = evidence_id(row)
        result.setdefault(key, {**row, 'evidence_id': key, 'matrix_ids': []})['matrix_ids'].append(row['sample_id'])
    return list(result.values())


def frame_hash(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()


def frames_for_backend(path, load_video):
    import torchvision.transforms as tv
    video = load_video(str(path), num_frames=16)
    _, _, height, width = video.size()
    if min(height, width) > 768:
        scale = 720. / min(height, width)
        video = tv.Resize(size=(int(scale * height), int(scale * width)))(video)
    return video.permute(0, 2, 3, 1).numpy()


def physical_mirror(source, target, axis, load_video):
    """Lossless RGB encode; GIFs use the official PIL decoder and frame indices."""
    import numpy as np
    target.parent.mkdir(parents=True, exist_ok=True)
    input_bytes = None
    timing = 'source timestamps passed through'
    input_args = ['-i', str(source)]
    if source.suffix.lower() == '.gif':
        # FFmpeg's GIF decoder can discard zero-duration frames that the official
        # PIL loader includes. Decode with that same loader before lossless encode.
        from fractions import Fraction
        from PIL import Image, ImageSequence
        with Image.open(source) as image:
            durations = [f.info.get('duration') for f in ImageSequence.Iterator(image)]
        known = set(durations) - {None, 0}
        if len(known) > 1 or (known and any(d is None or d == 0 for d in durations)):
            raise ValueError('variable GIF timing requires an explicit timestamp-preserving encoder')
        rate = str(Fraction(1000, next(iter(known)))) if known else '10'
        timing = 'uniform GIF duration preserved' if known else 'GIF has no duration metadata; 10 fps representation, all PIL frame indices preserved'
        full = load_video(str(source), return_tensor=False)
        _, height, width, _ = full.shape
        input_bytes = full.tobytes()
        input_args = ['-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                      '-framerate', rate, '-i', 'pipe:0']
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-y', *input_args,
        '-map', '0:v:0', '-vf', axis, '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'veryfast',
        '-pix_fmt', 'rgb24', '-fps_mode', 'passthrough', '-an', '-f', 'mp4']
    if not target.exists():
        temporary = target.with_suffix('.partial.mp4')
        run = subprocess.run(command + [str(temporary)], input=input_bytes, capture_output=True)
        if run.returncode:
            raise RuntimeError('ffmpeg: ' + run.stderr[-800:].decode(errors='replace'))
        temporary.replace(target)
    before = load_video(str(source), num_frames=16, return_tensor=False)
    after = load_video(str(target), num_frames=16, return_tensor=False)
    expected = np.flip(before, axis=2 if axis == 'hflip' else 1)
    parity = before.shape == after.shape and np.array_equal(expected, after)
    if not parity:
        difference = float(np.max(np.abs(expected.astype('float32') - after.astype('float32')))) if expected.shape == after.shape else None
        raise ValueError(f'lossless mirror sample mismatch: {before.shape} vs {after.shape}, max error {difference}')
    return {'transformed_video': str(target), 'transformed_video_sha256': sha(target),
            'timing_policy': timing,
            'sampled_pixels_equal_exact_mirror': True, 'source_sample_sha256': [frame_hash(f) for f in before],
            'transformed_sample_sha256': [frame_hash(f) for f in after], 'ffmpeg_command': command + [str(target)]}


def detect(model, frame):
    import torch
    with torch.no_grad():
        result = model.run_caption_tensor(frame)
    detections = [{'label': r[0], 'box': [float(v) for v in r[1][:4]]} for r in result[0]]
    labels = sorted(set(str(v) for v in result[0][0][2])) if result[0] else []
    return detections, labels


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['spatial','objects'], required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--cache', required=True, help='complete original backend export for this task')
    p.add_argument('--video-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--artifact-root', required=True)
    p.add_argument('--vbench-root', default='/root/wenbiao_zhao/VBench')
    p.add_argument('--grit-weights', default='/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-videos', type=int, help='pilot; keeps complete plan')
    p.add_argument('--reserve-disk-gb', type=float, default=20)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args(argv)


def run(args, out):
    plan = unique_plan([r for r in J.read_jsonl(Path(args.plan)) if r['task'] == args.task])
    if not plan:
        raise ValueError('empty transform plan')
    cache = {r['relative_path']: r for r in J.read_jsonl(Path(args.cache))}
    upstream = subprocess.check_output(['git','-C',args.vbench_root,'rev-parse','HEAD'], text=True).strip()
    if upstream != UPSTREAM_SHA:
        raise ValueError('upstream commit mismatch')
    identity = {'schema': 1, 'task': args.task, 'plan_sha256': sha(args.plan), 'cache_sha256': sha(args.cache),
        'code_sha256': {n: sha(ROOT / n) for n in ['scripts/semantic/cache_matrix_transforms.py', 'packages/prompt-compiler/src/vbench_prompts_compile/visual_transforms.py']},
        'upstream_sha': upstream, 'weights_sha256': sha(args.grit_weights), 'frames': 16,
        'variants': len(plan), 'artifact_root': args.artifact_root,
        'note': 'occlusion geometry verified, semantic visibility unverified'}
    J.bind_job(out.with_suffix('.manifest.json'), identity)
    if args.dry_run:
        print(json.dumps(identity, indent=2)); return 0
    journal = out.with_suffix('.attempts.jsonl')
    latest = {r['evidence_id']: r for r in J.read_jsonl(journal)}
    grouped = defaultdict(list)
    for row in plan:
        if row['evidence_id'] not in latest:
            grouped[row['relative_path']].append(row)
    # GPU imports are only needed for pending physical variants.
    sys.path.insert(0, str(Path(args.vbench_root).resolve()))
    import numpy as np
    import torch
    from PIL import Image
    from vbench.utils import load_video
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA backend unavailable')
    model = None
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    videos_attempted = 0
    stopped = None
    for relative, variants in grouped.items():
        if args.max_videos is not None and videos_attempted >= args.max_videos:
            break
        if shutil.disk_usage(artifact_root).free < args.reserve_disk_gb * 1024 ** 3:
            stopped = 'disk_reserve_reached'; break
        base = cache.get(relative, {})
        source = Path(args.video_root) / relative
        video_uid = hashlib.sha256(relative.encode()).hexdigest()[:24]
        frames = None
        frames_error = None
        memo = {}
        if base.get('status') == 'ok' and any(v['eligible'] and v['evidence_kind'] != 'evidence_mirror' for v in variants):
            try:
                if sha(source) != base.get('video_sha256'):
                    raise ValueError('source video changed since original cache')
                frames = frames_for_backend(source, load_video)
                if len(frames) != 16 or [frame_hash(f) for f in frames] != base['frame_sha256']:
                    raise ValueError('original frame sampling/preprocessing changed')
                for frame, detections, labels in zip(frames, base['frame_detections'], base['frame_labels']):
                    memo[frame_hash(frame)] = (detections, labels)
            except Exception as error:
                frames_error = f'{type(error).__name__}: {error}'
        for row in variants:
            record = {**row, 'source_video_sha256': base.get('video_sha256'), 'frame_size': base.get('frame_size')}
            try:
                if not row['eligible']:
                    record['status'] = 'ineligible'
                elif base.get('status') != 'ok':
                    record.update(status='source_cache_missing', source_status=base.get('status'))
                elif row['evidence_kind'] == 'evidence_mirror':
                    record.update(status='ok', frame_detections=mirror_detections(base['frame_detections'], base['frame_size'], row['axis']),
                                  frame_labels=base['frame_labels'], frame_status=['ok'] * 16,
                                  source_frame_sha256=base['frame_sha256'])
                else:
                    if frames_error or frames is None:
                        raise ValueError(frames_error or 'no source frames')
                    if row['evidence_kind'] == 'video_mirror':
                        target = artifact_root / 'spatial' / (video_uid + '-' + row['axis'] + '.mp4')
                        record.update(physical_mirror(source, target, row['axis'], load_video))
                        transformed = frames_for_backend(target, load_video)
                        patch_details = [{'status': 'ok', 'kind': row['axis']} for _ in transformed]
                    else:
                        required = row['base_official_target'].split(' and ')
                        patched = [patch_frame(frame, detections, required, row['occlusion_control'], row['level'])
                                   for frame, detections in zip(frames, base['frame_detections'])]
                        transformed = [pair[0] for pair in patched]
                        patch_details = [pair[1] for pair in patched]
                    detections, labels, hashes, errors = [], [], [], []
                    for i, (frame, details) in enumerate(zip(transformed, patch_details)):
                        if frame is None:
                            detections.append(None); labels.append(None); hashes.append(None); errors.append(details['status'])
                            continue
                        digest = frame_hash(frame)
                        hashes.append(digest)
                        try:
                            if digest not in memo:
                                if model is None:
                                    try:
                                        from vbench.third_party.grit_model import DenseCaptioning
                                        candidate = DenseCaptioning(torch.device(args.device))
                                        candidate.initialize_model_det(model_weight=args.grit_weights)
                                        model = candidate
                                    except Exception as error:
                                        raise BackendUnavailable(str(error)) from error
                                memo[digest] = detect(model, frame)
                            d, lab = memo[digest]
                            detections.append(d); labels.append(lab); errors.append('ok')
                        except BackendUnavailable:
                            raise
                        except Exception as error:
                            detections.append(None); labels.append(None); errors.append('backend:' + type(error).__name__)
                            details['backend_error'] = str(error)[:300]
                        if args.task == 'objects' and row.get('occlusion_control') == 'target' and row.get('level') == 1:
                            png = artifact_root / 'objects' / video_uid / f'target-full-frame-{i:02d}.png'
                            png.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(frame.astype('uint8')).save(png)
                            details['endpoint_image'] = str(png)
                            details['endpoint_image_sha256'] = sha(png)
                    record.update(status='ok' if all(e == 'ok' for e in errors) else 'partial',
                        frame_detections=detections, frame_labels=labels, frame_sha256=hashes, frame_status=errors,
                        frame_transform=patch_details, source_frame_sha256=base['frame_sha256'])
            except BackendUnavailable:
                raise
            except Exception as error:
                record.update(status='error:' + type(error).__name__, error=str(error)[:800])
            J.append_jsonl(journal, record)
            latest[row['evidence_id']] = record
            print(json.dumps({'cached_variants':len(latest), 'planned_variants':len(plan), 'last_status':record['status'],
                              'task':args.task, 'relative_path':relative, 'transform':row['transform']}), flush=True)
        videos_attempted += 1
    records = [latest.get(row['evidence_id'], {**row, 'status':'not_run'}) for row in plan]
    temporary = out.with_suffix('.jsonl.tmp')
    with temporary.open('w') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(out)
    statuses = dict(Counter(r['status'] for r in records))
    report = {'planned':len(plan), 'attempted':len(latest), 'status':statuses,
        'all_attempted':len(latest) == len(plan), 'all_eligible_complete':all(r['status']=='ok' for r in records if r['eligible']),
        'frame_status':dict(Counter(s for r in records for s in r.get('frame_status', []))),
        'stopped':stopped, 'visibility_verified':0, 'note':'full box coverage is not a visibility annotation'}
    J.atomic_json(out.with_suffix('.report.json'), report)
    print(json.dumps(report), flush=True)
    return 0 if report['all_attempted'] else 2


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    with J.job_lock(out.with_suffix('.lock')):
        return run(args, out)


if __name__ == '__main__':
    raise SystemExit(main())
