"""Review actual lossless clean/subject-blurred PNGs; no rendered substitutes."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil

from .common import sha256_file
from .subject_artifacts import artifact_path, new_output, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--variant", default="full/subject_blur")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = read_jsonl(args.dataset/"index.jsonl")
    output = new_output(args.output)
    (output/"images").mkdir()
    sections, evidence = [], []
    for entry in index:
        path = artifact_path(args.dataset, entry["manifest"])
        if sha256_file(path) != entry["manifest_sha256"]:
            raise ValueError("manifest hash mismatch")
        row = json.loads(path.read_text())
        if row["status"] != "accepted":
            continue
        base = row["base"]
        uid = base["video_uid"]
        clean, edited = row["variants"]["clean"], row["variants"][args.variant]
        positions = sorted({0, len(clean)//2, len(clean)-1})
        cells = []
        for position in positions:
            for label, files in (("Original", clean), ("Subject blurred; background preserved", edited)):
                source = artifact_path(args.dataset, files[position]["path"])
                if sha256_file(source) != files[position]["sha256"]:
                    raise ValueError("PNG content changed")
                name = f"{uid}_{position}_{'clean' if label == 'Original' else 'blur'}.png"
                shutil.copyfile(source, output/"images"/name)
                cells.append(f'<figure><figcaption>Frame {position} · {label}</figcaption><img loading="lazy" src="images/{name}"></figure>')
                evidence.append({"file": "images/"+name, "source": str(source), "sha256": files[position]["sha256"]})
        sections.append(f'<section><h2>{html.escape(base["prompt_id"])}</h2><p>{html.escape(base["generator"])} · {uid}</p><div class="grid">'+''.join(cells)+'</div></section>')
    (output/"subject_blur_review.html").write_text('''<!doctype html><meta charset="utf-8"><title>Background consistency — subject blur</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f5f5;color:#222}section{background:white;padding:16px;margin:20px 0}h2{font-size:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%;max-height:360px;object-fit:contain}figcaption{margin:8px}</style>
<h1>糊化主体，保留背景</h1><p>Actual lossless PNGs: first, middle and last frames. Independent mask quality is approximate. This page is visual evidence, not a repair effectiveness claim.</p>'''+''.join(sections))
    write_json(output/"provenance.json", {"dataset": str(args.dataset), "variant": args.variant,
        "dataset_index_sha256": sha256_file(args.dataset/"index.jsonl"), "reviewed_clips": len(sections),
        "candidate_count": len(index), "files": evidence})
    print(json.dumps({"clips": len(sections), "images": len(evidence), "output": str(output)}))


if __name__ == "__main__":
    main()
