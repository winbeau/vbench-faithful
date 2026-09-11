#!/usr/bin/env python3
"""Flatten official nested human_anno into one canonical pairwise table."""
import argparse, csv, json
from pathlib import Path

FIELDS = ["dimension","instance_id","prompt_id","prompt_en","group_id","model_a","model_b",
          "video_a_id","video_b_id","video_a_path","video_b_path","human_label","label_source"]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--dimension",required=True); ap.add_argument("--out",type=Path,required=True); ns=ap.parse_args()
    rows=[]; data=json.loads(ns.input.read_text())
    for i, obj in enumerate(data):
        videos=obj.get("videos",{}); anno=obj.get("human_anno",{})
        prompt=obj.get("prompt_en",""); group=obj.get("group_id",obj.get("style_en",obj.get("category_en","")))
        for a, inner in anno.items():
            for b, label in inner.items():
                if a >= b or label not in (0, .5, 1): continue
                rows.append({"dimension":ns.dimension,"instance_id":i,"prompt_id":prompt,
                  "prompt_en":prompt,"group_id":group,"model_a":a,"model_b":b,
                  "video_a_id":a,"video_b_id":b,"video_a_path":videos.get(a,""),
                  "video_b_path":videos.get(b,""),"human_label":label,"label_source":"official_human_anno"})
    ns.out.parent.mkdir(parents=True,exist_ok=True)
    with ns.out.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} canonical pairs to {ns.out}")
if __name__ == "__main__": main()
