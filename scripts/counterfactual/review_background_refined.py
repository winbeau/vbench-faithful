"""Display native background data, semantic seed, GrabCut mask and real blur."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .common import ROOT, sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true', help='Preview a live index snapshot; never a final review for scoring.')
    args = p.parse_args()
    run = json.loads((args.dataset/'run.json').read_text())
    index_bytes = (args.dataset/'index.jsonl').read_bytes()
    index_sha = hashlib.sha256(index_bytes).hexdigest()
    if not args.allow_partial and (not run.get('completed') or run['index_sha256'] != index_sha):
        raise ValueError('complete construction required')
    output = new_output(args.output)
    (output/'panels').mkdir()
    entries = [json.loads(line) for line in index_bytes.decode().splitlines() if line.strip()]
    label_map = json.loads((ROOT/'configs/subject-repair/ade20k_labels.json').read_text())['id2label']
    label_ids = {v: int(k) for k, v in label_map.items()}
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    cell, gap = 256, 8
    width = cell*4+gap*5
    rows, sections, overview = [], [], []
    for entry in entries:
        path = artifact_path(args.dataset, entry['manifest'])
        if sha256_file(path) != entry['manifest_sha256']:
            raise ValueError('manifest changed')
        source = json.loads(path.read_text())
        if source['status'] != 'accepted':
            continue
        uid = entry['video_uid']; number = len(rows)+1
        mask_path = artifact_path(args.dataset, source['construction_mask']['path'])
        if sha256_file(mask_path) != source['construction_mask']['sha256']:
            raise ValueError('construction mask changed')
        with np.load(mask_path, allow_pickle=False) as stored:
            masks = stored['masks']; labels = stored['semantic_labels']
        positions = sorted({0, len(masks)//2, len(masks)-1})
        panels, files = [], []
        for t in positions:
            images = []
            for variant in ['clean', 'full/subject_blur']:
                ref = source['variants'][variant][t]
                path = artifact_path(args.dataset, ref['path'])
                if sha256_file(path) != ref['sha256']:
                    raise ValueError('display image changed')
                images.append(np.array(Image.open(path).convert('RGB')))
                files.append(ref)
            original, blur = images
            semantic = labels[t] == label_ids[source['construction_target']]
            refined = masks[t].astype(bool)
            if not np.array_equal(original[~refined], blur[~refined]):
                raise ValueError('actual blur changed pixels outside the construction mask')
            overlays = []
            for mask in [semantic, refined]:
                tinted = original.copy()
                tinted[mask] = np.rint(.45*original[mask]+.55*np.array([255,20,40])).astype(np.uint8)
                overlays.append(tinted)
            panel = Image.new('RGB', (width, 316), '#f2f4f7'); draw = ImageDraw.Draw(panel)
            draw.text((8, 3), f"Frame {t} | semantic {semantic.mean():.2%} -> GrabCut {refined.mean():.2%}", font=font, fill='black')
            for j, (picture, title) in enumerate(zip([original, *overlays, blur], ['Original', 'SegFormer target (red)', 'GrabCut target (red)', 'Actual subject blur'])):
                x = gap+j*(cell+gap)
                draw.text((x, 28), title, font=small, fill='black')
                fitted = ImageOps.contain(Image.fromarray(picture), (cell,cell))
                panel.paste(fitted, (x+(cell-fitted.width)//2, 52+(cell-fitted.height)//2))
            panels.append(panel)
        title = f"{number:02d} | {source['base']['prompt_id']} | {source['base']['generator']} | {source['construction_target']} | mean {masks.mean():.2%}"
        canvas = Image.new('RGB', (width, 58+sum(x.height for x in panels)), 'white')
        draw = ImageDraw.Draw(canvas); draw.text((8,5), title, font=font, fill='black')
        draw.text((8,30), uid, font=small, fill='black'); y=58
        for panel in panels:
            canvas.paste(panel, (0,y)); y+=panel.height
        name = f'panels/{number:02d}_{uid}.png'; canvas.save(output/name)
        overview.append((title, panels[positions.index(len(masks)//2)]))
        rows.append({'number': number, 'video_uid': uid, 'manifest_sha256': entry['manifest_sha256'],
            'base': source['base'], 'target': source['construction_target'], 'mean_mask_area': float(masks.mean()),
            'inspected_frame_candidates': positions, 'panel': name, 'panel_sha256': sha256_file(output/name),
            'construction_mask_sha256': source['construction_mask']['sha256'], 'displayed_images': files})
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{uid}</p><a href="{name}"><img loading="lazy" src="{name}"></a></section>')
    for start in range(0,len(overview),4):
        selected = overview[start:start+4]
        canvas = Image.new('RGB',(width,len(selected)*350),'white'); draw=ImageDraw.Draw(canvas);y=0
        for title, panel in selected:
            draw.text((8,y+7),title,font=font,fill='black');canvas.paste(panel,(0,y+34));y+=350
        canvas.save(output/f'overview_{start//4+1:02d}.png')
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Background 原数据：SegFormer + GrabCut</title><style>body{font:16px system-ui;margin:24px;background:#f6f7fa}section{background:white;padding:12px;margin:20px 0}img{width:100%;max-width:1064px}h2{font-size:18px}</style><h1>Background 官方原片：主体糊化检查</h1><p>首、中、末帧：原图 → SegFormer 主体类别 → GrabCut 细化 → 实际糊化。百分比是画面面积，不是主体覆盖率。这是评分前的构造质量检查，尚未认定为有效干预。</p>'+''.join(sections))
    write_json(output/'provenance.json', {'dataset': str(args.dataset), 'dataset_index_sha256': index_sha,
        'construction_complete': bool(run.get('completed')), 'partial_preview': args.allow_partial,
        'candidate_count': len(entries), 'numerically_accepted': len(rows), 'examples': rows,
        'renderer_sha256': sha256_file(Path(__file__)), 'metric_scores_used': False})
    print(json.dumps({'review_clips': len(rows), 'output': str(output)}))


if __name__ == '__main__':
    main()
