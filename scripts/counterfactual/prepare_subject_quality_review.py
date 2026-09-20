"""Export raw first/middle/last frames for subject-image suitability review.

No metric, construction masks, detector, or score files enter this exporter.
Candidate ordering is supplied by a frozen metadata-only list. This gallery
does not approve a construction mask or freeze a scoring cohort.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import cv2

from .common import sha256_file
from .subject_artifacts import new_output, read_jsonl, upstream_frames, write_json, write_jsonl, write_png_sequence


def export(candidates: Path, video_root: Path, output: Path) -> None:
    packets, sections = [], []
    for ordinal, row in enumerate(read_jsonl(candidates), 1):
        source = (video_root / row["relative_video_path"]).resolve()
        if video_root.resolve() not in source.parents:
            raise ValueError("video path escapes input root")
        packet = {**row, "ordinal": ordinal, "review_status": "unreviewed"}
        try:
            frames = upstream_frames(source)
            indices = sorted({0, len(frames) // 2, len(frames) - 1})
            files = write_png_sequence(output, f'frames/{row["video_uid"]}', frames[indices])
            packet.update(source_video_sha256=sha256_file(source), num_frames=len(frames),
                          frame_indices=indices, frame_files=files, image_size=list(frames.shape[1:3]),
                          export_status="succeeded")
            images = []
            for index in indices:
                image = frames[index]
                height, width = image.shape[:2]
                image = cv2.resize(image, (round(320 * width / height), 320), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise OSError("preview JPEG encoding failed")
                uri = base64.b64encode(encoded).decode()
                images.append(f'<figure><img src="data:image/jpeg;base64,{uri}"><figcaption>Frame {index}</figcaption></figure>')
            content = '<div class="frames">' + ''.join(images) + '</div>'
        except Exception as exc:
            packet.update(export_status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
            content = '<p>' + html.escape(packet['failure_reason']) + '</p>'
        uid = html.escape(row['video_uid'])
        sections.append(f'<section id="c{ordinal}"><h2>#{ordinal} · {html.escape(row["prompt_en"])} · '
                        f'{html.escape(row["generator"])}</h2><p>{uid}</p>{content}'
                        f'<label>图像是否适合：<select data-uid="{uid}"><option value="pending">未审核</option>'
                        '<option value="candidate">保留候选</option><option value="exclude">不合适</option></select></label> '
                        f'<input data-reason="{uid}" placeholder="理由或主体说明（可选）"></section>')
        packets.append(packet)
        write_jsonl(output / 'candidates.jsonl', packets)
        print(f'{ordinal}: {row["video_uid"]} {packet["export_status"]}', flush=True)
    inputs = json.dumps(packets, ensure_ascii=False).replace('</', '<\\/')
    document = '''<!doctype html><meta charset="utf-8"><title>Subject image quality review</title>
<style>body{font:16px system-ui;margin:2rem}.frames{display:flex;gap:1rem;overflow:auto}figure{margin:0}
img{height:260px}section{border-top:1px solid #ccc;margin:2rem 0;padding:1rem 0}input{min-width:25rem}
header{position:sticky;top:0;background:white;padding:1rem;border:1px solid #ddd}button,select{padding:.5rem}</style>
<header><h1>先审核原始图像，再审核分割</h1><p>28 条人物候选：7 个官方 prompt × 4 个生成器，固定 seed 0。
首、中、末帧仅供初筛。主体应清楚可辨，背景应有可糊化的结构；不按一致性分数筛选。
保留候选不表示分割合格，也不会自动开始评分。</p><button onclick="download()">导出已审核条目</button>
<span id="count"></span></header>''' + ''.join(sections) + '''<script>
const packets=/*PACKETS*/[];
function download(){const decisions=[];for(const s of document.querySelectorAll('select[data-uid]')){
 if(s.value==='pending')continue;const p=packets.find(p=>p.video_uid===s.dataset.uid);
 decisions.push({video_uid:p.video_uid,prompt_en:p.prompt_en,source_video_sha256:p.source_video_sha256,
 image_suitability:s.value,reason:document.querySelector(`[data-reason="${p.video_uid}"]`).value,
 review_source:'human_raw_frame_gallery',reviewed_at:new Date().toISOString(),final_dataset_acceptance:false});}
 if(!decisions.length){document.querySelector('#count').textContent='请先审核至少一条。';return;}
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([decisions.map(d=>JSON.stringify(d)).join('\\n')+'\\n'],{type:'application/jsonl'}));
 a.download='subject_quality.reviewed.jsonl';a.click();URL.revokeObjectURL(a.href);}
</script>'''
    (output / 'quality_review.html').write_text(document.replace('/*PACKETS*/[]', inputs))
    write_json(output / 'manifest.json', {'candidate_source_sha256': sha256_file(candidates),
               'total_candidates': len(packets), 'decoded': sum(p['export_status'] == 'succeeded' for p in packets),
               'score_inputs_used': False, 'construction_masks_used': False, 'decode': 'VBench@fd18b3d.load_video; all frames',
               'purpose': 'visual screening only; no cohort is approved by this export'})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.candidates, args.video_root, new_output(args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
