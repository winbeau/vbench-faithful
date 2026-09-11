#!/usr/bin/env python3
"""Toy-only smoke tests for pair deduplication and prompt-disjoint split."""
import csv, random, tempfile
from pathlib import Path

def split(rows, seed=0, test_fraction=.4):
    prompts=sorted({r["prompt_id"] for r in rows}); random.Random(seed).shuffle(prompts)
    cut=max(1, round(len(prompts)*test_fraction)); test=set(prompts[:cut])
    return ([r for r in rows if r["prompt_id"] not in test], [r for r in rows if r["prompt_id"] in test])
def main():
    rows=[]
    for p in range(5):
        for a,b,label in [("A","B",1),("A","C",.5),("B","C",0)]:
            rows.append({"prompt_id":f"p{p}","model_a":a,"model_b":b,"human_label":str(label)})
    dev,test=split(rows); assert {x["prompt_id"] for x in dev}.isdisjoint({x["prompt_id"] for x in test})
    keys={(x["prompt_id"],min(x["model_a"],x["model_b"]),max(x["model_a"],x["model_b"])) for x in rows}; assert len(keys)==len(rows)
    assert sum(float(x["human_label"])==.5 for x in rows)==5
    print(f"PASS toy_records={len(rows)} dev={len(dev)} test={len(test)} ties=5 disjoint=True unique_pairs=True")
if __name__ == "__main__": main()
