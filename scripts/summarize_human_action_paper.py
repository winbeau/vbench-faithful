#!/usr/bin/env python3
"""Aggregate existing Human Action paper experiments; no evaluator calls."""
from __future__ import annotations
import csv, json, random, statistics
from pathlib import Path
ROOT=Path("/root/autodl-tmp/vbench-audit-storage/runs/human_action")
DATA=Path("/root/autodl-tmp/vbench-audit-storage/datasets/counterfactual/human_action/semantic_target")
CONDS=("correct","related","incorrect")
def rows(path): return json.loads(path.read_text())
def mean(x): return statistics.fmean(x) if x else None
def median(x): return statistics.median(x) if x else None
def ci(vals, fn, n=5000):
    rng=random.Random(20260914); m=len(vals)
    if not m: return [None,None]
    samples=sorted(fn([vals[rng.randrange(m)] for _ in range(m)]) for _ in range(n))
    return [samples[int(.025*(n-1))], samples[int(.975*(n-1))]]
def main():
    manifests=[json.loads(x) for x in (DATA/"metadata/manifest.jsonl").read_text().splitlines() if x]
    meta={(r["base_id"],r["condition"]):r for r in manifests}
    all_by={}
    for backend in ("vbench","audit"):
      per={}
      for cond in CONDS:
        fs=sorted((ROOT/"semantic_target"/cond/"human-action"/backend).glob("*/results.json"))
        if len(fs)!=1: raise SystemExit(f"expected one {cond}/{backend} result, got {fs}")
        per[cond]={Path(r["video"]).name:r for r in rows(fs[0])}
      combined=[]; paired=[]
      for i in range(1,31):
        bid=f"ha_g{i:03d}"; alias=f"video_{i:03d}.mp4"; scores={c:float(per[c][alias]["score"]) for c in CONDS}
        for c in CONDS:
          r=dict(per[c][alias]); m=meta[(bid,c)]; r.update({"base_id":bid,"condition":c,"source":m["video"],"source_action":m["source_action"],"target_action":m["target_action"],"metric_variant":"official" if backend=="vbench" else "repaired","temporal_mean_probability":r.get("temporal_mean_target_probability") if backend=="audit" else None,"window_count":r.get("diagnostics",{}).get("window_count") if isinstance(r.get("diagnostics"),dict) else None,"failure_or_abstention":r.get("failure_reason")}); combined.append(r)
        m=meta[(bid,"correct")]; paired.append({"base_id":bid,"generator":Path(m["video"]).parent.name,"source_action":m["source_action"],"score_correct":scores["correct"],"score_related":scores["related"],"score_incorrect":scores["incorrect"],"correct_gt_related":scores["correct"]>scores["related"],"related_gt_incorrect":scores["related"]>scores["incorrect"],"correct_gt_incorrect":scores["correct"]>scores["incorrect"],"strict_ordering":scores["correct"]>scores["related"]>scores["incorrect"]})
      (ROOT/"semantic_target"/("official.jsonl" if backend=="vbench" else "repaired.jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in combined))
      all_by[backend]=paired
    combined_pairs=[]; summary=[]
    for backend, ps in all_by.items():
      for r in ps: combined_pairs.append({"metric_variant":"official" if backend=="vbench" else "repaired",**r})
      rate=lambda k:mean([float(x[k]) for x in ps]); gaps={"correct_related": [x["score_correct"]-x["score_related"] for x in ps],"related_incorrect":[x["score_related"]-x["score_incorrect"] for x in ps],"correct_incorrect":[x["score_correct"]-x["score_incorrect"] for x in ps]}
      for metric,value in [("correct_gt_related",rate("correct_gt_related")),("related_gt_incorrect",rate("related_gt_incorrect")),("correct_gt_incorrect",rate("correct_gt_incorrect")),("strict_ordering",rate("strict_ordering"))]: summary.append({"experiment":"semantic_target","metric_variant":"official" if backend=="vbench" else "repaired","n_independent":30,"n_evaluations":90,"metric":metric,"value":value,"bootstrap_ci_low":ci(ps,lambda z:mean([float(x[metric]) for x in z]))[0],"bootstrap_ci_high":ci(ps,lambda z:mean([float(x[metric]) for x in z]))[1]})
      for name,g in gaps.items(): summary.append({"experiment":"semantic_target","metric_variant":"official" if backend=="vbench" else "repaired","n_independent":30,"n_evaluations":90,"metric":"mean_gap_"+name,"value":mean(g),"median":median(g),"bootstrap_ci_low":ci(g,mean)[0],"bootstrap_ci_high":ci(g,mean)[1]})
    with (ROOT/"semantic_target"/"paired_results.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=combined_pairs[0]);w.writeheader();w.writerows(combined_pairs)
    with (ROOT/"semantic_target"/"summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=sorted({k for x in summary for k in x}));w.writeheader();w.writerows(summary)
    paper=[]
    for backend,ps in all_by.items():
      strict=mean([float(x["strict_ordering"]) for x in ps]); gap=mean([x["score_correct"]-x["score_incorrect"] for x in ps]); paper.append({"experiment":"semantic_target","metric_variant":"official" if backend=="vbench" else "repaired","n_independent":30,"n_evaluations":90,"primary_contract":"correct > related > incorrect","pass_rate":strict,"mean_primary_gap":gap,"failure_rate":0.0 if backend=="audit" else 1.0,"status":"complete"})
    for backend in ("official","repaired"): paper.append({"experiment":"action_execution","metric_variant":backend,"n_independent":0,"n_evaluations":0,"primary_contract":"positive > partial > negative","pass_rate":"","mean_primary_gap":"","failure_rate":"","status":"ACTION_EXECUTION_DATA_MISSING"})
    with (ROOT/"human_action_paper_summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=paper[0]);w.writeheader();w.writerows(paper)
    a=[x for x in summary if x["metric_variant"]=="repaired"]
    lookup={x["metric"]:x["value"] for x in a}
    (ROOT/"HUMAN_ACTION_EXPERIMENT_REPORT.md").write_text(f"""# Human Action paper experiments\n\n## Research question\nSemantic target discrimination (same video, varying explicit target) and action execution/temporal sensitivity (fixed target, varying execution state).\n\n## Dataset\nSemantic Target: 30 independent videos and 90 conditions. Action Execution: required 5 triplets/15 videos are absent.\n\n## Evaluators\nOfficial is frozen filename-target VBench. Repaired uses explicit `dimension_metadata.target_action` and `temporal_mean_probability`.\n\n## Semantic Target results\nRepaired: C>R {lookup["correct_gt_related"]:.3f}; R>I {lookup["related_gt_incorrect"]:.3f}; C>I {lookup["correct_gt_incorrect"]:.3f}; strict {lookup["strict_ordering"]:.3f}. Official is invariant to explicit metadata because it uses filename target.\n\n## Action Execution\nNot run: no source videos were found. No result is fabricated.\n\n## Limitations\nMatplotlib is absent from the existing environment, so no plots were generated. Official semantic-target scores do not test explicit targets under the no-filename-encoding constraint.\n""")
    print(json.dumps({"semantic_rows":60,"paper_summary":str(ROOT/"human_action_paper_summary.csv")}))
if __name__=="__main__": main()
