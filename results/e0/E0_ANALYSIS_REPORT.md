# E0 Analysis Report

## Reproducibility

- Local repository Git SHA: `f3f8190ef554ca58403182419415d0bcdf0b1f86`
- Server evaluator Git SHA: `f3f8190ef554ca58403182419415d0bcdf0b1f86`
- Locked VBench 1.0 Git SHA: `13dee903cc97e2633ed6e8f50dea61bc90717935`
- Inputs: `data/processed/pairwise_master_split.csv`, `data/processed/e0_scoring_manifest.csv`, `splits/e0_prompt_split.csv`, `results/e0/raw_official_scores/*/results.csv`
- Input SHA256: master `788cd605d90a91c242da7a0642a6836e26ed06960798230ae0af7035cbd6d2ae`, manifest `1e08807b42045afc43a71397b115fe8045c6c88251762fea1aafa1fc8b203f47`, split `831bed0aee880cb0c501ac89fe4cbe4e1e97422427619e24983a9e86a5d3c328`
- Split: frozen prompt-disjoint `(dimension, prompt_id)`, seed=20260911, nominal dev/test=40%/60%; no re-splitting.
- Human label: 1=model_a wins, 0=model_b wins, 0.5=tie.
- Win ratio: win=1, loss=0, tie=0.5, divided by comparisons participated; Spatial human and metric rates use the same supported subset.
- Tie-margin calibration: dev only; candidates are all unique absolute dev score differences plus zero; maximize dev tie-aware accuracy; smallest delta wins ties.
- Test protocol: freeze each dimension's selected delta without test tuning.
- Kendall tau-b: metric `(score_a-score_b)` versus human ordering `{-1,0,1}`; ties retained. CSV reports dev, test, and all-supported; summary uses test.
- Bootstrap: 2000 repetitions, clustered by test `prompt_id`, seed base=20260911; percentile 95% intervals.

## Data counts

- dynamics_degree: prompts=72, groups=360, pairs=2160, labels={'0': 701, '0.5': 1268, '1': 191}
- subject_consistency: prompts=72, groups=360, pairs=2160, labels={'0.5': 498, '1': 1174, '0': 488}
- human_action: prompts=100, groups=500, pairs=3000, labels={'0.5': 1659, '0': 727, '1': 614}
- spatial_relationship: prompts=108, groups=540, pairs=3240, labels={'0.5': 1472, '1': 722, '0': 1046}

## Final metrics

- dynamics_degree: Pearson=0.815453 (n=4), test zero-margin=0.684496 CI=[0.634089, 0.733333], delta=0, test tie-aware=0.684496 CI=[0.634089, 0.733333], test tau-b=0.461214, coverage=1.000000
- subject_consistency: Pearson=0.960580 (n=4), test zero-margin=0.584496 CI=[0.549612, 0.618605], delta=0.00030195713, test tie-aware=0.585271 CI=[0.551143, 0.619380], test tau-b=0.382951, coverage=1.000000
- human_action: Pearson=0.904245 (n=4), test zero-margin=0.553333 CI=[0.496111, 0.616111], delta=0, test tie-aware=0.553333 CI=[0.496111, 0.616111], test tau-b=0.265161, coverage=1.000000
- spatial_relationship: Pearson=0.725736 (n=4), test zero-margin=0.505442 CI=[0.461905, 0.545578], delta=0.344889983, test tie-aware=0.525170 CI=[0.476173, 0.572806], test tau-b=0.291660, coverage=0.777778

## Spatial coverage and abstention

Spatial official VBench 1.0 supports only left/right/top/bottom in the locked evaluator path.

24 inside-of prompts from the public preference set are therefore reported as unsupported/abstained, not assigned fabricated scores.

- Records: 1680/2160 supported (77.78%); 480 unsupported.
- Prompts: 84/108 supported (77.78%); 24 unsupported.
- Pairs: 2520/3240 supported (77.78%); 720 unsupported.

## Warnings and limitations

- Model-level Pearson uses only four generators (`n=4`); it is descriptive and no strong model-level confidence interval is claimed.
- Pair-level uncertainty uses prompt-cluster bootstrap; pair rows within a prompt are not treated as independent.
- Binary Human Action and Dynamic Degree scores create many metric ties; tie-aware results depend on dev-calibrated margins.
- Spatial results apply only to the official-supported relation subset and must not be generalized to `inside of`.
