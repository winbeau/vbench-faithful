#!/usr/bin/env python3
"""Write/validate explicit Action Execution manifest; performs no inference."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path("/root/autodl-tmp/vbench-audit-storage/datasets/counterfactual/human_action/action_execution")
TARGETS={"guitar":"strumming guitar","teeth":"brushing teeth","watermelon":"cutting watermelon","dishes":"washing dishes","window":"cleaning windows"}
STATES=("positive","partial","negative")
def main():
    src=ROOT/"source"; meta=ROOT/"metadata"; meta.mkdir(parents=True,exist_ok=True)
    records=[]; missing=[]
    for group,target in TARGETS.items():
      for state in STATES:
        path=src/f"{group}_{state}.mp4"; exists=path.is_file(); missing.extend([] if exists else [path.name])
        records.append({"action_group":group,"video_state":state,"video_path":str(path),"target_action":target,"prompt":f"A person is {target}","dimension_metadata":{"target_action":target},"status":"ready" if exists else "missing"})
    (meta/"expected_manifest.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records))
    (ROOT/"README.md").write_text("Upload exactly 15 MP4s under source/: guitar, teeth, watermelon, dishes, window crossed with positive, partial, negative. Targets are explicit in metadata/expected_manifest.jsonl.\n")
    print(json.dumps({"expected":15,"present":15-len(missing),"missing":missing,"upload_target":str(src)}))
if __name__=="__main__": main()
