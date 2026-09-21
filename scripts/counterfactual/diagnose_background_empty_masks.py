"""Audit empty construction masks without treating them as absent subjects."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .common import ROOT, sha256_file
from .subject_artifacts import artifact_path, new_output, write_json


def empty_cause(fractions, selected):
    """Algorithmic cause only; no inference about what the video depicts."""
    if np.any(fractions[selected]):
        return 'selected_semantic_nonempty_refinement_empty'
    if any(np.any(values) for values in fractions.values()):
        return 'selected_absent_despite_other_foreground_evidence'
    return 'no_detection_in_ten_class_vocabulary'


def render_one(arguments):
    dataset, video_root, output, entry, number, label_ids = arguments
    from vbench_audit_core.upstream import import_official_module
    import torch
    torch.set_num_threads(1)
    path = artifact_path(dataset, entry['manifest'])
    if sha256_file(path) != entry['manifest_sha256']:
        raise ValueError('manifest changed')
    source = json.loads(path.read_text())
    mask_path = artifact_path(dataset, source['construction_mask']['path'])
    if sha256_file(mask_path) != source['construction_mask']['sha256']:
        raise ValueError('mask changed')
    with np.load(mask_path, allow_pickle=False) as data:
        masks, labels = data['masks'], data['semantic_labels']
    if masks.any():
        raise ValueError('reported empty mask is not empty')
    video = video_root/source['base']['relative_video_path']
    if sha256_file(video) != source['video_sha256']:
        raise ValueError('source video changed')
    module, _ = import_official_module('background_consistency')
    frames = module.load_video(str(video)).permute(0, 2, 3, 1).numpy().astype(np.uint8)
    if frames.shape[:3] != masks.shape:
        raise ValueError('decoded frames and masks differ')
    fractions = source['semantic_class_fractions']
    union = np.isin(labels, label_ids)
    union_areas = union.mean((1, 2))
    peak = int(np.argmax(union_areas))
    positions = [0, len(frames)//2, len(frames)-1]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    panel = Image.new('RGB', (848, 256), 'white')
    draw = ImageDraw.Draw(panel)
    cause = empty_cause(fractions, source['construction_target'])
    short = {'selected_semantic_nonempty_refinement_empty': 'REFINEMENT EMPTY',
             'selected_absent_despite_other_foreground_evidence': 'ABSENT CLASS SELECTED',
             'no_detection_in_ten_class_vocabulary': 'NO TEN-CLASS DETECTION'}[cause]
    draw.text((6, 3), f"{number:03d} | {source['base']['prompt_id']} | {source['base']['generator']} | {short}", font=font, fill='black')
    draw.text((6, 22), f"{entry['video_uid']} | selected {source['construction_target']} | union peak {union_areas[peak]:.2%}", font=small, fill='black')
    pictures = [frames[t] for t in positions]
    overlay = frames[peak].copy()
    overlay[union[peak]] = np.rint(.45*overlay[union[peak]]+.55*np.array([255, 20, 40])).astype(np.uint8)
    pictures.append(overlay)
    for k, picture in enumerate(pictures):
        x = 6+210*k
        title = f'Original frame {positions[k]}' if k < 3 else f'All 10 classes, peak frame {peak}'
        draw.text((x, 39), title, font=small, fill='black')
        fitted = ImageOps.contain(Image.fromarray(picture), (204, 196))
        panel.paste(fitted, (x+(204-fitted.width)//2, 57+(196-fitted.height)//2))
    name = f'panels/{number:03d}_{entry["video_uid"]}.png'
    panel.save(output/name)
    failures = source['refinement_failures']
    return {'number': number, 'video_uid': entry['video_uid'], 'base': source['base'], 'cause': cause,
            'construction_target': source['construction_target'], 'num_frames': len(frames),
            'selected_semantic_nonempty_frames': int(np.count_nonzero(fractions[source['construction_target']])),
            'any_semantic_nonempty_frames': int(np.count_nonzero(union_areas)),
            'union_mean_area': float(union_areas.mean()), 'union_peak_area': float(union_areas[peak]),
            'refinement_failure_frames': len(failures), 'refinement_failure_reasons': dict(Counter(x['reason'] for x in failures)),
            'displayed_original_frames': positions, 'displayed_union_frame': peak,
            'manifest_sha256': entry['manifest_sha256'], 'video_sha256': source['video_sha256'],
            'mask_sha256': source['construction_mask']['sha256'], 'panel': name, 'panel_sha256': sha256_file(output/name)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-n', type=int, default=182)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--video-root', type=Path, default=Path('/root/wenbiao_zhao/datasets/vbench-1.0-human-preference/videos'))
    args = parser.parse_args()
    index_bytes = (args.dataset/'index.jsonl').read_bytes()
    entries = [json.loads(line) for line in index_bytes.decode().splitlines()][:args.first_n]
    if len(entries) != args.first_n:
        raise ValueError('requested prefix is not yet complete')
    output = new_output(args.output); (output/'panels').mkdir()
    protocol = json.loads((args.dataset/'protocol.json').read_text())
    id2label = json.loads((ROOT/'configs/subject-repair/ade20k_labels.json').read_text())['id2label']
    label_ids = [int(k) for k, name in id2label.items() if name in protocol['foreground_labels']]
    selected = []
    for entry in entries:
        source = json.loads(artifact_path(args.dataset, entry['manifest']).read_text())
        if source.get('foreground_fraction') and not np.any(source['foreground_fraction']):
            selected.append(entry)
    arguments = [(args.dataset, args.video_root, output, entry, i+1, label_ids) for i, entry in enumerate(selected)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(render_one, arguments))
    pages = []
    for start in range(0, len(rows), 6):
        group = rows[start:start+6]
        page = Image.new('RGB', (848, 256*len(group)), '#e8e8e8')
        for i, row in enumerate(group):
            with Image.open(output/row['panel']) as panel:
                page.paste(panel, (0, 256*i))
        name = f'page_{start//6+1:02d}.png'; page.save(output/name); pages.append(name)
    result = {'dataset': str(args.dataset), 'index_snapshot_sha256': hashlib.sha256(index_bytes).hexdigest(),
              'prefix_count': len(entries), 'prefix_index': entries, 'empty_count': len(rows),
              'counts': dict(Counter(x['cause'] for x in rows)), 'metric_scores_used': False,
              'interpretation': 'Counts are computational failure categories, not visual subject-absence judgments.',
              'renderer_sha256': sha256_file(Path(__file__)), 'pages': pages, 'examples': rows}
    write_json(output/'diagnosis.json', result)
    with (output/'empty_cases.csv').open('w', newline='') as handle:
        fields = ['number', 'video_uid', 'cause', 'construction_target', 'num_frames', 'selected_semantic_nonempty_frames',
                  'any_semantic_nonempty_frames', 'union_mean_area', 'union_peak_area', 'refinement_failure_frames', 'panel']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    sections = []
    for row in rows:
        title = html.escape(f"{row['number']:03d} · {row['base']['prompt_id']} · {row['cause']}")
        sections.append(f'<section><h2>{title}</h2><img loading="lazy" src="{row["panel"]}"></section>')
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>空掩码逐例核查</title><style>body{font:16px system-ui;margin:24px}h2{font-size:17px}img{max-width:100%;width:848px}section{border-top:1px solid #ccc;padding:12px 0}</style><h1>前 182 条中的全部空掩码核查</h1><p>前三列是原片首、中、末帧；第四列是所有前景类别检出面积最大的帧及红色语义覆盖。全空指构造输出，不代表原片空白或真的没有主体。未读取 origin/repair 分数。</p>'+''.join(sections))
    print(json.dumps({'empty_count': len(rows), 'counts': result['counts'], 'output': str(output)}))


if __name__ == '__main__':
    main()
