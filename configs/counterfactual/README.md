# Counterfactual base selection of record

`bases_published.jsonl` is the base selection the published
[`xjuIcthub/counterfactual-vbench`](https://huggingface.co/datasets/xjuIcthub/counterfactual-vbench)
dataset was built from: 205 bases over 7 dimensions.

It is committed because the working copy of the selection,
`output/counterfactual/bases.jsonl`, lives under `output/`, which is gitignored,
and because `select_bases.py` on its own does **not** reproduce the published
detector-dependent subsets.

Per dimension, the published clip inventory and the plain `select_bases.py`
output agree on only 5 of 7 dimensions:

| dimension | published | `select_bases.py` alone | overlap |
|---|---:|---:|---:|
| dynamics_degree | 40 | 40 | 40 |
| human_action | 25 | 25 | 25 |
| motion_smoothness | 25 | 25 | 25 |
| scene | 25 | 25 | 25 |
| spatial_relationship | 40 | 40 | 40 |
| multiplt_object | 25 | 25 | **6** |
| subject_consistency | 25 | 25 | **9** |

The two mismatching dimensions are exactly the detector-dependent ones.
`66c4a99` added `pick_detectable.py` alongside the ranked pool, and it
re-selects precisely those dimensions by walking a ranked candidate pool and
keeping the first `budget` candidates that GRiT can actually ground. The
published subsets are the oversampled ones.

## Reproducing the published selection

```bash
python -m scripts.counterfactual.select_bases \
  --output output/counterfactual/bases.jsonl \
  --summary output/counterfactual/bases_summary.json

python -m scripts.counterfactual.pick_detectable \
  --bases output/counterfactual/bases.jsonl \
  --dataset-root /root/wenbiao_zhao/datasets/vbench-1.0-human-preference \
  --output output/counterfactual/bases.jsonl \
  --summary output/counterfactual/detector_eligibility.json
```

With no `--dimension`, `pick_detectable.py` re-selects all three
detector-dependent dimensions (`subject_consistency`, `multiplt_object`,
`spatial_relationship`) and copies the rest through unchanged.

Verified on 2026-09-15: re-selecting the two dimensions whose plain selection
differed and copying the others through reproduces the published clip inventory
**on all 7 dimensions, 205/205 bases**.

## Detector eligibility counts

Recorded because the construction rejection count was previously only printed:

| dimension | ranked pool scanned | kept (dev/test) | rejected |
|---|---:|---|---:|
| subject_consistency | 25 | 5 / 20 | 0 |
| multiplt_object | 49 | 5 / 20 | **24** |

For Multiple Objects more than half of the ranked candidates had to be skipped
because GRiT could not ground the named targets, which is the fixture finding
recorded in `docs/counterfactual-reports/multiplt_object.md`.
