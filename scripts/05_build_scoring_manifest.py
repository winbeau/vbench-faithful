#!/usr/bin/env python3
"""Build stable per-video scoring manifest without downloading or decoding videos."""
import argparse, csv, hashlib
from collections import defaultdict
from pathlib import Path

MAIN = {"dynamics_degree": 1440, "subject_consistency": 1440,
        "human_action": 2000, "spatial_relationship": 2160}

def uid(dimension, path):
    # The official relative path is reused across dimensions; scope the UID by dimension.
    return "v_" + hashlib.sha256((dimension + "\0" + path).encode("utf-8")).hexdigest()[:20]

def read(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--master",type=Path,default=Path("data/processed/pairwise_master_split.csv")); ap.add_argument("--manifest",type=Path,default=Path("data/processed/e0_scoring_manifest.csv")); ap.add_argument("--out-master",type=Path,default=Path("data/processed/pairwise_master_split.csv")); ns=ap.parse_args()
    rows=read(ns.master); videos={}; prompt_splits={}; bare_paths=defaultdict(set)
    for r in rows:
        key=(r["dimension"],r["instance_id"],r["prompt_id"]); prompt_splits[(r["dimension"],r["prompt_id"])]=r["split"]
        for side in ("a","b"):
            path=r[f"video_{side}_path"]; gen=r[f"video_{side}_id"]
            k=(r["dimension"],path); bare_paths[path].add(r["dimension"])
            record={"video_uid":uid(r["dimension"],path),"dimension":r["dimension"],"split":r["split"],"prompt_id":r["prompt_id"],"group_id":r["instance_id"],"generator":gen,"relative_video_path":path}
            if k in videos: assert videos[k] == record, f"conflicting mapping: {k}"
            else: videos[k]=record
    manifest=sorted(videos.values(),key=lambda x:(x["dimension"],x["relative_video_path"]))
    assert len({x["video_uid"] for x in manifest}) == len(manifest)
    conflicts={p:sorted(ds) for p,ds in bare_paths.items() if len(ds)>1}
    for x in manifest: assert x["split"] == prompt_splits[(x["dimension"],x["prompt_id"])]
    for dim,n in MAIN.items():
        got=sum(x["dimension"]==dim for x in manifest); assert got==n,(dim,got,n)
    assert sum(x["dimension"] in MAIN for x in manifest) == 7040
    fields=["video_uid","dimension","split","prompt_id","group_id","generator","relative_video_path"]
    ns.manifest.parent.mkdir(parents=True,exist_ok=True)
    with ns.manifest.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(manifest)
    mapping={(x["dimension"],x["relative_video_path"]):x for x in manifest}
    out=[]
    for r in rows:
        q=dict(r)
        for side in ("a","b"):
            x=mapping[(r["dimension"],r[f"video_{side}_path"])]
            q[f"video_{side}_uid"]=x["video_uid"]
        out.append(q)
    fields2=list(rows[0]) + [x for x in ("video_a_uid","video_b_uid") if x not in rows[0]]
    with ns.out_master.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields2); w.writeheader(); w.writerows(out)
    counts=defaultdict(int)
    for x in manifest: counts[(x["dimension"],x["split"],x["generator"])] += 1
    print(f"PASS unique_videos={len(manifest)} main_videos=7040 pairs={len(rows)}")
    print(f"WARNING bare_relative_path_cross_dimension_conflicts={len(conflicts)} (UID is dimension-scoped)")
    for k in sorted(counts): print(*k,counts[k],sep="\t")

if __name__ == "__main__": main()
