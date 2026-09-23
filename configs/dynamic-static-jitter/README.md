# Dynamic Degree jitter development protocols

## Model retraining after the still-frame counterexample

`vjepa-anchored-v1.json` is a user-authorized **model training** trial, not an
inference-score correction. Keep the frozen encoder and train the same 51,393
head parameters with natural preferences, native-jitter consistency, known still
zeros and coherent weak/strong/reversing motion constraints. No output offset,
rescaling, new download, main32/TEST450 training, or calibration45 media access.
The old mean-latent gauge objective is absent. Only original DEV210/60 is used.
300 steps, one initialization, no validation checkpoint selection or sweep.

1350 added still/pan/reversal/still-jitter features completed with zero failures.
The initial large-tensor training attempt failed its nonfinite-loss guard before
the first ten-step progress message; no checkpoint was produced. Its provenance
and failure are retained, not counted as a trained model. Recovery uses bounded
24-view GPU tensors, concatenates predictions for the identical full-batch loss,
and retains all data, weights and hyperparameters. Output/gradient equivalence
has a unit test. The model still requires the frozen development gates to pass.

Actual root: `/data/chenjiayu/dynamic-structural-motion-20260922/vjepa-anchored-v1/`.
From its `code/` snapshot with the recorded H200 Python/PYTHONPATH/GPU mask:

```bash
python -m scripts.counterfactual.train_vjepa_anchored \
  --config configs/dynamic-static-jitter/vjepa-anchored-v1.json extract \
  --probe-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-probe-v1 \
  --video-root /data/chenjiayu/wenbiao_zhao/vbench-official-v1 \
  --shard 0 --shards 4 --output ../features/shard-0

python -m scripts.counterfactual.train_vjepa_anchored_chunked \
  --config configs/dynamic-static-jitter/vjepa-anchored-v1.json \
  --probe-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-probe-v1 \
  --features ../features \
  --diagnostic-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-static-frame-v1 \
  --output ../training-chunked
```

Run all four extraction shards before training, use fresh outputs for any replay,
and keep the failed `training/` untouched. The optional diagnostic is evaluated
only after the final head has been saved, never in training or model selection.
[Authoritative report and limitations](../../docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-anchored).

## Completed expansion and still-frame diagnostic (old frozen head)

`vjepa-expansion450-v1.json` expands the previously exposed 30 TEST prompts to all
450 MP4s: 900 CF, 450 controls, 1800 inputs per backend, all complete with zero
failures. `expand_vjepa_validation` provides `select/prepare/score/summarize`;
`audit_vjepa_expansion` independently verifies results. Origin **0.680000→0.857778**,
old joint Repair **0.518089→0.516319**. This is coverage expansion, not new unseen
prompts; all 450 human pairs now have both CF endpoints. [Evidence](../../docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-expansion450).

`probe_vjepa_static_frame` / `audit_vjepa_static_frame` separately test one DEV32
frame, kept out of the 450-source denominator. Still **0.456649**, 8px pan
**0.442835**, 32px pan **0.480346**: the old head has a concrete motion-ordering
counterexample, not just an uncalibrated zero. These five diagnostic inputs are
not training examples for the new model. [Evidence and videos](../../docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-static-frame).

## Frozen validation on actual VBench 1.0 (user authorized)

`vjepa-validation-v1.json` freezes the already-trained heads and unchanged
sigmoid mapping before testing. Repeat the original 32 groups/128 inputs;
construct original/control/two 8px seeds for the 90 MP4s in the pre-reserved
120-source/30-TEST-prompt list. The 30 GIFs remain explicitly NOT SCORED under
the native MP4/16-frame/8-FPS protocol; no replacement or timestamps invented.
All 450 official MP4s at those 30 prompts also receive natural preference
evaluation, with labels read only after all predictions are complete.
Total: 848 inputs per backend, no training or rescaling.

Use `python -m scripts.counterfactual.validate_vjepa_motion --help`
(`select/prepare/score/summarize`); construction reuses the unchanged
`official_video_jitter` builder with the selection's frozen `construction.json`.
Actual isolated root is
`/data/chenjiayu/dynamic-structural-motion-20260922/vjepa-validation-v1/`;
local receipts live at `output/dynamic-static-jitter/vjepa-validation-v1/`.
Source selection and protocol are fixed before media inference. This tests the
unchanged *relative* score; absolute motion-strength calibration is still absent.
The 45 calibration-reserved videos remain unopened. These test prompts are
now marked exposed and must not become an iterative tuning set.
[Full protocol and acceptance scope](../../docs/plans/2026-09-22-dynamic-structural-motion-repair-goal.md#vjepa-validation-plan).

Completed: **848/848 inputs per backend, zero failures, 122 exact encoding
controls**. On held-out 90 sources, Origin **0.700000→0.900000**, joint Repair
**0.517112→0.514804**, MAE **0.012016**. On existing DEV32, Origin
**0.593750→0.781250**, Repair **0.515726→0.508504**. The frozen-relative-scale
10% numerical check passes; absolute-strength certification does not follow.
Natural preference concordance is 81.87%→85.96% over 171 ordered pairs, with
279 human ties reported separately; the paired difference CI includes zero.
All 120 new legacy quality flags remain scored; the 30 excluded GIFs remain
in `selection/excluded_sources.jsonl`. [Authoritative results, uncertainty,
ablation and limitations](../../docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-validation).

### Reproduce selection, construction and frozen inference

Use the recorded `code/` execution snapshot and the existing H200 `vbench`
Python environment; weights and heads remain in the earlier `vjepa-probe-v1/`.
No `train` step belongs to this validation. `select --help`, `prepare --help`
and `score --help` list all required paths (place `--config` before the subcommand).
Metadata-only selection from the repository root:

```bash
.venv/bin/python -m scripts.counterfactual.validate_vjepa_motion \
  --config configs/dynamic-static-jitter/vjepa-validation-v1.json select \
  --pool data/processed/e0_scoring_manifest.csv \
  --reserved configs/dynamic-static-jitter/sources.v1.jsonl \
  --probe-sources output/dynamic-static-jitter/vjepa-probe-v1/selection/sources.jsonl \
  --dev-sources configs/dynamic-static-jitter/sources.local-texture-dev32-mp4-v1.jsonl \
  --output output/dynamic-static-jitter/vjepa-validation-reproduction/selection
```

Copy that selection to a fresh task root on the model host. Run the unchanged
builder four times with `(shard, source-start, limit)` equal to `(0,0,23)`,
`(1,23,23)`, `(2,46,23)`, `(3,69,23)` (the last slice has 21 sources).
For example, from the task's `code/` directory, with `python` resolving to
the recorded H200 environment:

```bash
python -m scripts.counterfactual.official_video_jitter \
  --sources ../selection/counterfactual_sources.jsonl \
  --config ../selection/construction.json \
  --data-root /data/chenjiayu/wenbiao_zhao/vbench-official-v1 \
  --split test --source-start 0 --limit 23 \
  --ffmpeg /absolute/path/to/pinned-ffmpeg \
  --output ../construction/shard-0

python -m scripts.counterfactual.validate_vjepa_motion \
  --config configs/dynamic-static-jitter/vjepa-validation-v1.json prepare \
  --selection ../selection --construction ../construction \
  --dev-manifest /absolute/path/to/unchanged-dev32/candidates.jsonl \
  --video-root /data/chenjiayu/wenbiao_zhao/vbench-official-v1 \
  --probe-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-probe-v1 \
  --output ../inputs

CUDA_VISIBLE_DEVICES="<one-available-GPU-UUID>" \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=.:packages/audit-core/src:packages/audit-models/src:metrics/dynamic-degree/src \
python -m scripts.counterfactual.validate_vjepa_motion \
  --config configs/dynamic-static-jitter/vjepa-validation-v1.json score \
  --backend vjepa --inputs ../inputs/inputs.jsonl \
  --probe-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-probe-v1 \
  --video-root /data/chenjiayu/wenbiao_zhao/vbench-official-v1 \
  --raft-weight /data/chenjiayu/.cache/vbench/raft_model/models/raft-things.pth \
  --upstream /data/chenjiayu/dynamic-static-jitter-20260922/VBench \
  --shard 0 --shards 4 --output ../scores/vjepa/shard-0
```

Run shard 0/1/2/3 in four isolated GPU processes, each using logical `cuda:0`;
wait for complete receipts before repeating with `--backend origin` and fresh
`scores/origin/shard-N` outputs. The DEV32 manifest is the byte-identical
`local-texture-dev32-mp4-8px-v1/candidates.jsonl` (SHA fixed in the config).
`prepare` also needs the previous probe feature manifests for leakage checks.
Use the recorded FFmpeg 7.0.2 binary SHA from each construction receipt.
All output directories must be fresh; never overwrite the completed run.
The recorded manifests contain some `../construction/...` paths, so inference
must run from their original task `code/` context; do not rewrite frozen paths.

### Recompute and independently audit existing predictions (CPU only)

After both backends are complete, from the repository root:

```bash
.venv/bin/python -m scripts.counterfactual.validate_vjepa_motion \
  --config configs/dynamic-static-jitter/vjepa-validation-v1.json summarize \
  --inputs output/dynamic-static-jitter/vjepa-validation-v1/inputs/inputs.jsonl \
  --scores output/dynamic-static-jitter/vjepa-validation-v1/scores \
  --human-pairs data/processed/pairwise_master_split.csv \
  --output output/dynamic-static-jitter/vjepa-validation-v1/analysis-reproduction

.venv/bin/python -m scripts.counterfactual.audit_vjepa_motion_probe \
  --validation --root output/dynamic-static-jitter/vjepa-validation-v1 \
  --human-source data/processed/pairwise_master_split.csv \
  --output output/dynamic-static-jitter/vjepa-validation-v1/audit-reproduction.json
```

The independent audit checks the authoritative `analysis/summary.json`, not a
newly selected checkpoint. Local copies omit videos and dense token caches;
those remain on H200. The 45 calibration media are not needed by these commands.

## Completed bounded trial: frozen V-JEPA 2.1 video features

User-authorized official weight download and development only. The frozen
`vjepa-probe-v1.json` records the official source commit, observed checkpoint
SHA-256/size, prompt-disjoint split, preprocessing, augmentation, two loss arms,
fixed training length and feasibility checks. The sole experiment entry is
`python -m scripts.counterfactual.vjepa_motion_probe --help` (`select/extract/train`).
The metric readout is experimental and is **not** wired into the public CLI.

Selection: exclude every main32/reserved-test prompt; hash-split the remaining
21 DEV prompts into 14 train/4 validation/3 calibration-reserved prompts
(210/60/45 official MP4s). Only train and validation media are extracted, with
all native 16 frames/8 FPS, no crop or frame skip, and two new 8px coordinate
warp views per source. The 45 reserved calibration videos remain unopened.
Natural-only and joint-invariance heads share initialization and exactly 100
updates. No validation checkpoint selection, backbone finetuning or sweep.
Sigmoid outputs are **uncalibrated relative scores**, not the old physical
intensity units. The later frozen validation above reports the main32/Origin
10% check as a numerical diagnostic only, not absolute-strength acceptance.

```bash
.venv/bin/python -m scripts.counterfactual.vjepa_motion_probe \
  --config configs/dynamic-static-jitter/vjepa-probe-v1.json select \
  --pool data/processed/e0_scoring_manifest.csv \
  --pairs data/processed/pairwise_master_split.csv \
  --exclude configs/dynamic-static-jitter/sources.local-texture-dev32-mp4-v1.jsonl \
            configs/dynamic-static-jitter/sources.v1.jsonl \
  --output output/dynamic-static-jitter/vjepa-probe-reproduction/selection
```

Actual isolated H200 root:
`/data/chenjiayu/dynamic-structural-motion-20260922/vjepa-probe-v1/`.
Subdirectories: `assets/` (pinned clean source and weights), `code/` (execution
snapshot), `selection/`, `features/shard-{0,1,2,3}/`, `training/`.
Run `extract` once per GPU-bound shard (`CUDA_VISIBLE_DEVICES=GPU-...`, physical
4–7; `--source-root assets/vjepa2 --checkpoint assets/vjepa2_1_vitb_dist_vitG_384.pt
--selection selection --video-root <official-root> --shard N --shards 4`).
Run `train --selection selection --features features --output <fresh-output>`
only after all four completion ledgers pass coverage/hash/repeatability checks.
Use the H200 `vbench` environment and set `PYTHONPATH` to the snapshot root,
`packages/audit-models/src` and `metrics/dynamic-degree/src`; no dependency sync
or network access occurs inside the experiment script. Each phase requires a
fresh output directory. Large token caches and weights remain outside Git.

Completed bounded trial: 810/810 features, zero failures, two 51,393-parameter
heads, exactly 100 steps each. Joint validation score **0.566153→0.564112**,
MAE **0.009302** (33.31% below natural-only); ordered accuracy 20/23 clean,
19/23 under either perturbation. Development feasibility passes, **not final
motion-intensity repair or main32 acceptance**. [Full report, uncertainty and
negative findings](../../docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-probe).

Independent recomputation (no encoder/training):

```bash
.venv/bin/python -m scripts.counterfactual.audit_vjepa_motion_probe \
  --root output/dynamic-static-jitter/vjepa-probe-v1 \
  --output output/dynamic-static-jitter/vjepa-probe-v1/audit-reproduction.json
```

On H200, `--verify-heads` reloads the two saved heads with safe strict state
loading and replays all 1620 learned scores from existing frozen feature caches.
It makes no encoder calls and does not read additional media or labels.

## Previous ablation: larger SAM support; joint goal still fails

`scoring.outer-support-dev32-v1.json` changes only smallest-containing SAM
ownership to largest-containing ownership, keeping residual-first gates,
the full-grid intensity and frozen tau unchanged. All 191 inputs replayed:
**0.560659→0.678265**, signed drift **62.72%** of Origin's, still above 10%.
Compared with residual-first, 22 CFs decrease, 37 are unchanged and 5 increase;
all 95 originals are unchanged. Larger groups do not certify physical parts.

The 95 originals cover only 23 Dynamic DEV human pairs: 4 ordered, 19 tied.
The ordered comparison is Origin 3/4, candidate 4/4, NOT sufficient natural
motion validation. A separate `natural-motion-proxy-review.focus9-v1.json`
records actual all-frame inspection of nine originals, prior exposure included.
No true-static label is inferred from a low score or prompt.
[Evidence and limitations](../../docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-outer-support).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.counterfactual.replay_residual_reversal \
  --config configs/dynamic-static-jitter/scoring.outer-support-dev32-v1.json \
  --output output/dynamic-static-jitter/dev32-outer-support-replay
.venv/bin/python -m scripts.counterfactual.audit_natural_motion_response \
  --scores output/dynamic-static-jitter/dev32-outer-support-replay/scores.jsonl \
  --output output/dynamic-static-jitter/natural95-response-replay
```

## Previous candidate: residual-first reversal, same frozen scale

`scoring.residual-reversal-dev32-v1.json` adds a second analysis branch after
separating protected regional affine and per-point mean motion. All old gates,
the 144-point population, continuous aggregation and calibrated tau stay fixed.
It consumes only same-video model evidence, not partners or construction data.

All 128 DEV32 inputs plus all 63 natural controls completed CPU cache replay:
Repair **0.560659→0.681470**, versus previous **0.560659→0.683035**. Only 7/64
CFs decrease; the other 57 and all 95 originals remain exactly unchanged.
The remaining drift is **64.43%** of Origin's, so the 10% goal still fails.
This small development improvement is not validated physical-motion repair or
a public-default promotion. [Results and limitations](../../docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-residual-reversal).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.counterfactual.replay_residual_reversal \
  --config configs/dynamic-static-jitter/scoring.residual-reversal-dev32-v1.json \
  --output output/dynamic-static-jitter/dev32-residual-reversal-replay
```

The replay requires the completed DEV32 and natural63 caches and fit artifacts
under `output/dynamic-static-jitter/`. It revalidates inputs, reconstructs the
previous decomposition, checks unchanged scores, and writes fresh evidence.
Neither the frozen inputs nor earlier runs are overwritten.

## Current score scale: independent natural calibration

Origin remains the original official per-video boolean; Repair remains
continuous. A separately fitted monotone scale now maps raw intensity `I` to
`I/(I+tau)`, with **tau = 0.12684953311437752 short-side lengths/second**,
offset 0 and fixed exponent 1. Zero maps to zero; there is no motion threshold
or hard clipping. This aligns numerical ranges, not the physical meaning of
intensity with a binary dynamic fraction.

`scoring.natural63-calibration-v1.json` declares the method before new inference.
`sources.natural63-calibration-v1.jsonl` selects all 21 remaining DEV prompts,
one SHA256-ranked MP4 for each of LaVie/ModelScope/VideoCrafter. Neither the
DEV32 prompts/UIDs nor reserved TEST prompts/UIDs may appear. The score-blind
selector and receipt preserve this separation. All 63 natural originals were
scored, no failures or replacements, without constructing any new CFs.

Calibration matches this separate cohort's original Origin mean (36/63), then
freezes the scale for all DEV32 originals, controls, CFs and raw ablations.
`calibration.natural63-intensity-v1.json` preserves the compact fitted parameter
and complete fit artifact SHA; `scoring.dev32-calibrated-intensity-v1.json`
binds its fit protocol and the prior raw-intensity evaluation summary.
No offset was fitted to make DEV32 Repair slightly higher, and no CF outcome
was used in fitting. Public/default Repair is not changed by this experiment.

Initial [calibrated results](../../output/dynamic-static-jitter/dev32-cotracker3-calibrated-intensity-v1/SUMMARY.md):
Origin **0.593750→0.781250**, Repair **0.560659→0.683035**. Baseline is 0.033091
lower, not slightly higher. The signed drift is **65.27%** of Origin's, failing
the 10% goal. Raw intensity and all 128 Origin records remain unchanged.
Natural-motion noninferiority, new CF human review and formal holdout remain
NOT RUN. See the [calibration report](../../docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-calibrated-scale).

Reproduction (fresh output directories; immutable inference snapshot under
`output/dynamic-static-jitter/code-natural63-calibration-v1/`):

```bash
# On H200, from the execution snapshot: originals only, no video generation.
python -m scripts.counterfactual.select_intensity_calibration \
  --sources configs/dynamic-static-jitter/sources.natural63-calibration-v1.jsonl \
  --video-root /data/chenjiayu/wenbiao_zhao/vbench-official-v1 \
  --output /data/chenjiayu/dynamic-structural-motion-20260922/natural63-calibration-inputs-v1
python -m scripts.counterfactual.launch_mp4_dev32 --calibration-natural63 \
  --task-root /data/chenjiayu/dynamic-structural-motion-20260922 \
  --output /data/chenjiayu/dynamic-structural-motion-20260922/natural63-calibration-scoring-v1

# After retrieving the completed cache, on local CPU; fit never reads DEV32 scores.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.counterfactual.calibrate_motion_intensity fit \
  --run-root output/dynamic-static-jitter/natural63-calibration-scoring-v1 \
  --execution-root output/dynamic-static-jitter/code-natural63-calibration-v1 \
  --manifest output/dynamic-static-jitter/natural63-calibration-inputs-v1/candidates.jsonl \
  --sources configs/dynamic-static-jitter/sources.natural63-calibration-v1.jsonl \
  --config configs/dynamic-static-jitter/scoring.natural63-calibration-v1.json \
  --output output/dynamic-static-jitter/natural63-calibration-fit-replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.counterfactual.calibrate_motion_intensity apply \
  --calibration output/dynamic-static-jitter/natural63-calibration-fit-replay/calibration.json \
  --evaluation output/dynamic-static-jitter/dev32-cotracker3-intensity-v1 \
  --config configs/dynamic-static-jitter/scoring.dev32-calibrated-intensity-v1.json \
  --output output/dynamic-static-jitter/dev32-calibrated-replay
```

The H200 paths above are the actual completed run, not overwrite targets:
choose a fresh task root for new inference. The local replay commands need
the full finished cache, not partially synchronized evidence. Four isolated
GPU workers are used, subject to the existing free-memory checks and asset pins.

## Raw continuous intensity: representation retained beneath calibration

The user explicitly requires **Origin unchanged; Repair NOT binary**.
`scoring.dev32-cotracker3-intensity-v1.json` is the unscaled private research
protocol. Its raw intensity is the integrated top-5%-grid displacement divided by native
short side and elapsed motion time, in short-side lengths/second. No motion
threshold, score clipping, or data-fitted rescaling. Official VBench's original
per-video boolean and batch dynamic fraction remain untouched. The old
`scoring.dev32-cotracker3-boolean-v1.json` is superseded, retained for explicit
historical reproduction only; the launcher needs `--legacy-boolean` to use it.
Public scoring defaults have not been promoted to this research candidate.

The user authorized replacing the eight untimed DEV GIFs with official DEV
MP4s. `sources.local-texture-dev32-mp4-v1.jsonl` retains 24 originals and adds
eight score-blind hash-ranked replicates, preserving eight prompts × four
sources. `select_mp4_dev32.py` and the `.selection.json` document the rule;
no reserved TEST video was opened. All 128 planned inputs were scored; 42
construction flags remain in the ledger, without score-based filtering.

The retained [raw continuous results](../../output/dynamic-static-jitter/dev32-cotracker3-intensity-v1/SUMMARY.md)
and [report](../../docs/counterfactual-reports/dynamic_static_jitter.md#最新32-组官方-mp4origin-原版repair-连续运动强度)
give Origin 0.593750→0.781250 (dynamic fraction) and Repair
0.255291→0.365000 (short-side lengths/second). These are different quantities;
do not interpret their raw drift ratio as a calibrated 10% acceptance test.
Repair still increases substantially. The same-unit raw-track ablation is
0.255291→0.374620, so the current decomposition removes only 8.06% of that
mean increase. Natural-motion validity, new CF human review and formal holdout
remain unverified.

`replay_mp4_dev32_intensity.py` rechecks the completed model cache, replays the
decomposition and changes only Repair aggregation, with zero new inference.
It preserves all 128 Origin records and all 32 exact encoding controls. New
outputs must use a fresh directory. Reproduction command is in the report.

## Current decision: fixed 8px intervention; Repair redesign

The user visually accepted the displayed five-video, two-seed 8px cohort on
2026-09-22. Keep 8 native pixels; do not weaken the intervention to help Repair.
[review.local-texture-dev5-8px-v1.json](review.local-texture-dev5-8px-v1.json) binds
this post-results development review to the ten exact candidate identities and
hashes. It does not approve unseen examples or claim that Repair passed.

Preserve the original automatic statuses (4 qualified / 6 rejected). For this
reviewed cohort, Jacobian <0.5 alone no longer vetoes semantic acceptance; use
all ten interventions for the new development goal. Origin remains 40%→80%, and
the old dev-v4 Repair has only 4/10 valid pairs. Numerical scores and historical outputs have
not changed. The scorer and native analysis now accept `--review <review.json>`;
they verify its bound identities and preserve automatic qualification flags.
The metric itself receives only the video and model/config, never the review.
See the [new Repair goal](../../docs/plans/2026-09-22-dynamic-structural-motion-repair-goal.md)
for video-only motion decomposition, coverage and motion-sensitivity safeguards.

### G1/G2 development candidates (none has passed the joint goal)

Latest user acceptance update: use the **32 DEV source groups' batch mean
scores**, with `delta = mean(CF) - mean(base)`, and require
`delta_repair <= 0.10 * delta_origin` when Origin increases, on a fixed
comparable score scale. Do not substitute MAE, average positive-only increments,
or per-video pass/fail. Each source has equal weight; average multiple seeds
within source first and report each seed separately. The five-source diagnostic
is not this 32-group evaluation. The MP4 DEV32 comparison is now complete under
the continuous protocol above; the unscaled cross-unit ratio was not evaluated,
while the later frozen 0–1 scale gives a 65.27% drift ratio and fails the 10% goal.
Missing groups cannot be dropped, and true-motion preservation remains
required. See the updated goal.

`tracker.cotracker3-preparation-v1.json` is an asset/reference specification for
`probe_feature_tracks prepare --tracker-kind cotracker3-offline`, **not a frozen
Repair protocol or download permission**. The user subsequently explicitly
authorized its download and the fixed DEV comparison, which has now completed.
It pins the official checkpoint SHA/size/revision and the unchanged 20 inputs;
`tracker.cotracker3-source-v1.json` binds 31 existing local Python source files.
The new local-only adapter refuses missing or mismatched assets, uses safe strict
state loading, and does not change v2 or public scoring defaults. Real loading
parity now checks all 188 state tensors plus a 16-frame/12x12-grid official input:
state/visibility match exactly, maximum coordinate difference is 0 pixels.
This is implementation parity, not physical-motion correctness.

The 16 original SIFT requests in
`output/dynamic-static-jitter/cotracker3-allphase-dev5-v1-requests/` each contain
all 20 videos, with one distinct native query start per request. They bind the
immutable `code-cotracker3-query-prep-v1` snapshot and are **not predictions**.
All SIFT positions are preserved. A separate fixed 12x12 cell-center grid arm
uses `prepare --query-detector grid --grid-size 12`, again all 16 starts.
The [preparation report](../../docs/counterfactual-reports/dynamic_static_jitter.md#纯本地接入与全起点请求准备仍无模型实验)
records 320 calls' query geometry, control checks and source identities.

Before every new execution, reverify available GPU memory and local asset hashes.
Run `probe_feature_tracks infer` with the bound requests and explicit
local `--tracker-root` / `--tracker-weight`. The request determines v2 versus v3;
unknown families fail rather than silently falling back. Of 320 prepared calls,
128 exceed the existing default 2048-point guard, with a maximum of 3353 points.
The completed run explicitly used `--max-queries 4096`: maximum actual query
count was 3353, with peak allocated GPU memory about 5.02 GiB (SIFT) / 1.76 GiB
(grid). This does not certify arbitrary 4096-query videos. Never subsample or batch joint queries to evade
this guard. Missing/model failures remain failures, not low motion scores.
Both arms completed 320/320 calls, with zero failures and 80/80 exact encoding
controls each. These are 20 videos, not 640 independent samples. The authorized
model comparison is complete, but the known train miss remains. Replaying the
existing protected decomposition as a failure ablation leaves all ten CF
conditional speeds above base; its paired diagnostic MAE is 0.044164→0.034739.
Visibility plus image bounds is not an independent correspondence certificate;
all scores remain null, and no G1/G2 acceptance or default promotion is claimed.
The [completed comparison](../../docs/counterfactual-reports/dynamic_static_jitter.md#cotracker3-实测双查询臂完成但仍未修复)
links the single authoritative summary, full outputs and local verification.

The immutable execution snapshot is `code-cotracker3-comparison-v1`; use a fresh
output and one available GPU per process. For each arm, run all shards 0–3:

```bash
CUDA_VISIBLE_DEVICES=<available-GPU-UUID> OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.counterfactual.run_tracker_request_batch \
  --requests <bound-16-phase-request-directory> \
  --tracker-root <pinned-local-source> --tracker-weight <verified-local-checkpoint> \
  --output <fresh-arm-shard-directory> --shard <0-through-3> --shards 4 \
  --max-queries 4096
```

`audit_cotracker3_comparison` checks both complete arms against their requests and
execution snapshot, verifies every cache and all controls, then replays the
unchanged regional-reversal diagnostic with the same-video SAM cache. Supply
`--run-root`, `--sift-requests`, `--grid-requests`, `--execution-root`,
`--region-run`, and a fresh `--output`. Failures/missing phases are not silently
omitted. Natural calibration, independent physical correspondence and formal
holdout remain incomplete; do not rerun or open them on this download permission.

Latest priority: inspect natural motion before adding another scorer. The
[mechanism report](../../docs/counterfactual-reports/dynamic_static_jitter.md#机制复核先确认真实运动再设计-repair2026-09-22)
contains CPU diagnostics of the same 20 cached inputs and all-frame displays.
The original train visibly moves although its dense flow is nearly zero: neither
Origin=0 nor high background correspondence coverage establishes static truth.
Temporal reversal and local affine support are not sufficient classifiers.
Independent DINO descriptors have since been tested on the same 20 inputs:
direct nearest-neighbor displacement still increases on all ten CFs. This is
not a Repair. SAM automatic regions have now completed 20 videos/320 frames;
multi-hypothesis regional translation with spatial cross-validation is implemented
and its four CPU shards have completed 20 videos/1,080 frame pairs, with zero
runtime failures; all outputs were transferred, verified and merged. All scores
remain null. A confidently wrong train-direction example prevents
promotion. Full region tracks, articulation/rotation and scoring remain unfinished.

The new diagnostic entry point is `probe_structure_correspondence` with
`structure.dino-reciprocal-dev-v2.json`, the existing DINO ViT-B/16 source/weight, and the
same manifest/review. It extracts block-9 keys and last-block tokens, all native
phases, explicit patch/pixel geometry, mutuality/ambiguity and three-frame checks.
No prompt, original/CF pair, displacement field, or seed reaches the model.
All records have null scores; attention is not a semantic segmentation mask.

```bash
uv run --no-sync python -m scripts.counterfactual.probe_structure_correspondence \
  --manifest /path/to/local-texture-dev5-8px-v1/candidates.jsonl \
  --review configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json \
  --config configs/dynamic-static-jitter/structure.dino-reciprocal-dev-v2.json \
  --dino-root /path/to/local/facebookresearch_dino_main \
  --dino-weight /path/to/local/dino_vitbase16_pretrain.pth \
  --device cuda:0 --output output/dynamic-static-jitter/<fresh-feature-run>
```

The current code uses reciprocal antisymmetric subpixel refinement, avoiding
one-sided refinement's nonzero motion on identical features. Use the corrected
`structure.dino-reciprocal-dev-v2.json` for current runs. To reproduce the
completed CPU cache replay, replace the DINO options with
`--replay-source <completed-feature-run> --device cpu` and use a new output.
Replay checks feature hashes, media, geometry and full-cohort completeness;
it does not infer new descriptors or duplicate the large feature arrays.
The original one-sided diagnostic requires its immutable execution snapshot,
not silently substituting current code. An SSH-interrupted three-record replay
is kept separately; the full `v2b` replay has 20/20 diagnostic records.

### Automatic regions and joint translations (DEV diagnostics)

`probe_motion_regions` uses an existing **SAM ViT-H** checkpoint and an explicit
local official checkout, not MobileSAM or a language-conditioned detector. It
independently proposes regions on every native-time sampled frame, preserving
all returned proposals. SAM's threshold-stability statistic is not temporal
tracking confidence. Region masks, including all small proposals, are bit-packed
losslessly and hash-bound to the completed descriptor run.

```bash
uv run --no-sync python -m scripts.counterfactual.probe_motion_regions \
  --manifest /path/to/local-texture-dev5-8px-v1/candidates.jsonl \
  --review configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json \
  --config configs/dynamic-static-jitter/structure.sam-regions-dev-v1.json \
  --feature-run /path/to/completed-feature-run \
  --sam-root /path/to/local/segment-anything \
  --sam-weight /path/to/local/sam_vit_h_4b8939.pth \
  --device cuda:0 --output output/dynamic-static-jitter/<fresh-region-run>

uv run --no-sync python -m scripts.counterfactual.probe_regional_motion \
  --feature-run /path/to/completed-feature-run \
  --region-run /path/to/completed-region-run \
  --config configs/dynamic-static-jitter/structure.regional-motion-visible-dev-v2.json \
  --shard 0 --shards 4 --output output/dynamic-static-jitter/<fresh-motion-run>/shard-0
```

Repeat the CPU command with shard indices 1/2/3 and separate outputs; each video
is assigned once. No new descriptor or SAM inference occurs in this replay.
All 1/2/3/4 lag start phases are kept. The initial regional v1's fixed out-of-view
penalty could pin whole-frame subpixel motion to zero; it was intentionally
interrupted, and its two complete records remain historical evidence. Current
v2 instead reports conditional visible-support loss and its >=80% overlap gate.
Do not reproduce the old algorithm using current code or call this correction
a successful scientific Repair. Entirely unobserved motion stays null.

`merge_regional_motion --inputs <shard-0> <shard-1> <shard-2> <shard-3> --output
<fresh-merged-directory>` refuses running, missing, duplicate or incompatible
shards and checks every phase and whole-frame control. It preserves failed/null
records; even a complete merge is a diagnostic, not a passed motion score.

The next native-image cross-check avoids DINO matching: unfiltered native RGB
region templates search the full translation domain, with target-SAM overlap,
independent default SIFT reciprocal ratio matches, and sparse three-frame identity
closure. It is still a diagnostic with null scores, not a public/default Repair.
The train's first frame pair has the correct direction; full-clip correctness and
robustness are not established. Use byte-identical original media, never previews:

```bash
uv run --no-sync python -m scripts.counterfactual.probe_native_region_motion \
  --manifest /path/to/local-texture-dev5-8px-v1/candidates.jsonl \
  --review configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json \
  --region-run /path/to/sam-regions-dev5-8px-v1 \
  --config configs/dynamic-static-jitter/structure.native-regions-dev-v1.json \
  --video-root /path/to/byte-identical-official-originals \
  --video-root /path/to/local-texture-dev5-8px-v1/videos \
  --shard 0 --shards 4 --output output/dynamic-static-jitter/<fresh-native-shard-0>
```

Run shards 1/2/3 with fresh separate outputs; the same strict
`merge_regional_motion` entry accepts these records. `feature_provenance_sha256`
is explicitly null because these features are newly computed from RGB, not loaded
from a DINO feature cache. No parameters encode the intervention amplitude/phase.
All SAM regions and camera controls remain, including flat/unobservable regions.
Mask overlap and SIFT counts are not motion accuracy or valid-video coverage.

The v1 run completed all 20 videos / 1080 frame pairs and passed strict merging
and all five encoding-control equalities; every score remains null. Whole-clip
train correspondence is still ambiguous. `structure.native-regions-visible-dev-v2.json`
adds target-region visibility to the native RGB objective and continuous refinement:
new foreground occlusion must not drag the background. Unshifted overlap proposes
an identity only; motion still searches the complete domain for each proposed
target. Same hypotheses/overlap/precision/phase settings; missing support remains
missing. The actual bicycle first-frame-pair-window check improved background
alignment while preserving object translation. Full v2 completed/merged 20 videos,
1080 pairs, 37,653 region pairs with no runtime failure and five identical encoding
controls. There are 3,570 missing regional translations; all video scores remain
null. This does not establish final score invariance or natural motion retention.

### Same-video mask recovery and spatial/temporal witness checks

`probe_prompted_regions prepare` binds one post-hoc DEV source region/frame pair
to the completed native run, original SAM cache, byte-identical video and exact
decoded pixels. `infer` uses only that video's target frame and the existing
local SAM ViT-H weight/source. It preserves every box/point single/multimask
alternative; no SAM confidence becomes a motion certificate.

```bash
uv run --no-sync python -m scripts.counterfactual.probe_prompted_regions prepare \
  --manifest <bound-dev-manifest> --review <bound-user-review> \
  --native-run <complete-native-v2-merged-run> --region-run <complete-sam-run> \
  --candidate video_000004 --start 2 --lag 1 --region 0 \
  --video-root <byte-identical-originals> --output <fresh-request.json>

uv run --no-sync python -m scripts.counterfactual.probe_prompted_regions infer \
  --request <request.json> --region-cache <sam-run>/evidence/video_000004.npz \
  --sam-root <existing-segment-anything-checkout> --sam-weight <existing-vit-h-weight> \
  --device cuda:0 --output <fresh-prompted-run>

uv run --no-sync python -m scripts.counterfactual.audit_prompted_regions \
  --probe-run <prompted-run> --region-run <complete-sam-run> \
  --video-root <byte-identical-originals> --proposal-source rgb-only \
  --keypoint-method sift --output <fresh-rgb-only-audit>
```

The train pilot returned 48 masks; its RGB-only ablation keeps 32. The body mask
can be recovered, but the best RGB alignment still stays near zero. A single
independent SIFT location instead favors leftward motion; it cannot certify the
full region. Optional `--keypoint-method rootsift` / `akaze` are diagnostic
comparisons, not upgrades. Their four witnesses still select wrong/ambiguous
displacements in this case. RootSIFT is a reweighting of the same SIFT descriptors,
not an independent detector. Earlier immutable audit snapshots used the legacy
`sift_matches` JSON key for all methods; `keypoint_method` is authoritative. The
current writer uses the explicit `spatial_matches` key.

`probe_regional_geometry --analysis common-sift --source-run <native-v2-merged>
--region-run <complete-sam-run> --output <fresh-common-run>` replays the original
AMG regions without prompted augmentation. The completed 20-video run leaves
23,790/37,653 region pairs unranked; these nested-region counts are not valid-video
coverage. All five encoding controls match and all scores remain null.

The next diagnostic compares SIFT, RootSIFT and default AKAZE at the same ratio
test (0.75), every phase and every source region. Fresh SIFT must reproduce the
bound native baseline exactly. Three-frame identity composition checks a
correspondence through all available third frames; it does not test constant
velocity, suppress reversals or use an 8px/pixel-error threshold. Missing or
contradictory identities stay explicit. Agreement remains uncalibrated evidence,
not ground truth. No new SAM/flow inference, paired-clean mask transfer, or
reserved-test access occurs.

```bash
uv run --no-sync python -m scripts.counterfactual.probe_sparse_identity \
  --manifest <bound-dev-manifest> --review <bound-user-review> \
  --native-run <native-v2-merged> --region-run <complete-sam-run> \
  --video-root <byte-identical-originals> --video-root <bound-cf-videos> \
  --output <fresh-sparse-identity-run>

uv run --no-sync python -m scripts.counterfactual.summarize_sparse_identity \
  --source-run <finished-sparse-identity-run> --output <fresh-diagnostic-summary> \
  --prompted-run <prompted-run> --region-run <complete-sam-run> \
  --rgb-audit <rgb-only-audit> --video-root <byte-identical-originals>
```

Summary refuses running/incomplete/changed sources; it checks every video,
method, phase and encoding control. The optional prompted-case display shows
every common witness, not a manually selected positive example. These commands
produce diagnostic null scores only; full scoring/acceptance remains unfinished.

The complete three-method replay has now finished all 20 videos / 3240
method-pairs with no runtime failure and five exact encoding controls. SIFT,
RootSIFT and AKAZE can rank respectively 13,863 / 14,128 / 10,135 region pairs
before temporal checks, and 12,635 / 13,112 / 9,444 afterwards (each denominator
37,653). The prompted train case loses the correct SIFT witness for lack of a
third-frame match; a near-stationary AKAZE mismatch survives. These counts are
not correspondence accuracy, confidence calibration or score acceptance.

### Automatic feature queries and regional input conditioning (failed train pilots)

`probe_feature_tracks` reuses the exact earlier CoTracker2 source/weight. It
prepares every distinct SIFT location in the chosen native frame (same-location
orientations grouped), not just already-matched or manually marked moving points.
The four train versions at query frame 2 have 482 / 482 / 486 / 486 queries.
All 16 native frames remain input; no paired-clean positions reach inference.

```bash
uv run --no-sync python -m scripts.counterfactual.probe_feature_tracks prepare \
  --manifest <bound-dev-manifest> --review <bound-user-review> \
  --tracker-reference <old-local-trajectory-run>/provenance.json \
  --candidates video_000004 video_000005 video_000006 video_000007 --query-frame 2 \
  --video-root <byte-identical-originals> --video-root <bound-cf-videos> \
  --output <fresh-feature-query-request.json>

uv run --no-sync python -m scripts.counterfactual.probe_feature_tracks infer \
  --request <feature-query-request.json> --tracker-root <existing-co-tracker-source> \
  --tracker-weight <existing-cotracker2.pth> --device cuda:0 \
  --output <fresh-feature-track-run>

uv run --no-sync python -m scripts.counterfactual.audit_feature_tracks \
  --source-run <finished-feature-track-run> --case-candidate video_000004 \
  --source-key 272 --video-root <byte-identical-originals> --output <fresh-track-audit>
```

`--source-key 272` is a disclosed **post-hoc display only**, not an inference
query selector. The pilot keeps all automatic queries. Upstream forces the
query-frame positions/visibility, so their exact match cannot establish tracking
accuracy. Its pre-query backward completion is not an independent reverse-cycle
certificate. Out-of-view predicted coordinates remain unclamped.

For a separate regional-conditioning ablation, prepare a fresh request with
`--region-run <complete-sam-run>` and pass the same argument at inference. Every
source-frame SAM region is retained. Each region gives a PCA-aligned fixed
context=1.5 observation window mapped to 384x512; the affine transform stays
constant over the entire video and is inverted for native displacement. It does
not track a crop center or alter the original videos. Regions without queries
or sufficient geometry remain explicit, and overlapping regions cannot be
treated as independent score observations.

Both four-video pilots finished on H200 GPU 7, with no runtime failure and exact
encoding controls. Whole-frame feature seeding still misses the car-connection
motion. Regional conditioning ran 144 of 162 region proposals (18 insufficient)
and also misses it, sometimes with positive model visibility. Neither is a
passed repair or a valid-video coverage improvement. Their explicit evidence
and immutable execution snapshots are linked in the main Dynamic report.

`inspect_natural_motion --sources configs/dynamic-static-jitter/sources.v1.jsonl
--video-root <official-copy-root> --output <fresh-display-root>` exports every
native frame of predeclared `dev` sources only. It neither opens reserved test
videos nor creates motion annotations, scoring inputs, or calibration anchors.
Explicit `--allow-unknown-gif-timing` permits **display only** of GIF frame order
with null timestamps/FPS; it never guesses playback defaults for speed calibration.
All eight DEV GIFs lacked frame delays. All 32 contact sheets (24 MP4 + 8 GIF)
received an all-frame assistant proxy inspection, recorded in `natural-motion-proxy-review.dev32-v1.json`;
this is not human/blind annotation or a frozen low/high-motion calibration.

`probe_regional_geometry --source-run <complete-native-merged-run> --output
<fresh-geometry-probe>` replays every regional sparse correspondence as
translation/similarity/affine proposals, with ordinary and Huber estimators and
held-out spatial locations. Duplicate orientations at the same source location
share unit weight and stay in the same fold. Rotation/scale use pointwise motion,
not centroid drift; no temporal filter or final score is introduced. The first
20-video replay completed but 19,251/37,653 region pairs had no SIFT evidence.
These counts include controls and nested masks; they are not valid-video coverage.
Models cannot recover missing identities, and full-rank fits are not certified
correct motion. The native RGB/geometry/temporal fusion remains unfinished.

```bash
uv run --no-sync python -m scripts.counterfactual.analyze_flow_mechanism \
  --manifest /path/to/local-texture-dev5-8px-v1/candidates.jsonl \
  --review configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json \
  --source-scores /path/to/dense-tv-x2-dev5-8px-v1-scores/scores.jsonl \
  --cache-provenance /path/to/dense-path-common-dev5-8px-v1-probe/provenance.json \
  --output output/dynamic-static-jitter/<fresh-mechanism-run>
```

Run where the manifest's immutable MP4s and dense caches exist; no models are
loaded. Optional `--figures` uses an already-installed matplotlib and exports
display-only montages/trajectory plots. The numerical runner and tests do not
require matplotlib. All 20 records retain null scores and source insufficiency;
reliable-only measurements are conditional diagnostics, not full-video estimates.
Every lag start has fixed single-window ownership, with every phase retained.

The later controlled flow-model comparisons use the same
`trajectory.dense-dev-v1.json`, with the already-installed torchvision
`raft_large_C_T_SKHT_V2` local checkpoint. Add these research-runner options:

```text
--repair-variant dense-correspondence
--repair-flow-backend torchvision
--repair-flow-weight /path/to/local/raft_large_C_T_SKHT_V2-ff5fadd5.pth
--repair-flow-scale 1
```

Keep `--raft-weight` bound to the original VBench checkpoint for Origin. A
separate fresh run with `--repair-flow-scale 2` enlarges only the model's spatial
input; vectors return to original scoring-pixel units. No video, time sampling,
threshold or score denominator changes. Each run scored the same 20 inputs;
both have only 8/10 valid CF pairs, with the snowy-bike source still insufficient.
Higher correspondence support also increases the legacy jitter path length;
these are reliability ablations, not finished Repairs. Models are never downloaded.

`trajectory.dense-dev-v1.json` uses existing RAFT forward/backward fields and
deformable patch alignment, with no learned visibility head. The research runner
uses `--repair-variant dense-correspondence` and the same `--review`; 14/20
records and 6/10 CF pairs were valid. It has no public CLI variant yet. This is
a correspondence/observed-path ablation, not a complete nuisance-aware scorer.

`decomposition.dev-v1.json` defines a video-only second-derivative smoother plus
appearance-weighted, leave-one-point-out affine residual support. Repeated
reversal alone does not trigger suppression; unsupported small regions retain
their residual. `probe_structural_decomposition` reuses the original local-window
CoTracker caches (not the dense RAFT run). Its 20 records are **diagnostic_only,
score=null**: raw / temporal-only / structure-only / combined all-track quantities
include uncertain tracks and cannot establish scoring coverage or Repair success.
The first combined probe still inflated the train; natural periodic-motion and
human-preference validation remain incomplete.

```bash
uv run --no-sync python -m scripts.counterfactual.probe_structural_decomposition \
  --manifest output/dynamic-static-jitter/local-texture-dev5-8px-v1/candidates.jsonl \
  --review configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json \
  --source-scores output/dynamic-static-jitter/local-trajectory-dev5-8px-v1-scores/scores.jsonl \
  --config configs/dynamic-static-jitter/decomposition.dev-v1.json \
  --output output/dynamic-static-jitter/<fresh-probe-run>
```

Run where the original manifest video paths exist. The probe is CPU-only, checks
input/cache identities and raw-path replay parity, and retains source scoring
insufficiency. Mathematical trajectory tests are algorithm checks, not substitute
videos or natural-motion validation. The full evidence is in the linked report.

The second hypothesis, `decomposition.common-mode-dev-v1.json`, uses time-weighted
SVD of residual velocities plus explicit affine, same-direction, local-part and
small-region safeguards. No fixed intervention phase/frequency is provided.
Eligible modes retain their affine projection to protect simultaneous camera
motion. Reuse the command above with `--variant common-mode`, the new config and
a fresh output directory. Its 20 diagnostic-only records also failed to resolve
inflation: MAE 0.016379, versus 0.013689 for local support. Low evidence on the
dominant train residual is preserved, not bypassed to obtain smaller differences.
The new module is not a public/default Repair; natural motion and holdout remain
unvalidated. See the report for per-source results and per-mode retention reasons.

The same probe now also accepts a bound `dense-correspondence` source score run.
It validates cached fixed-grid flow replay, then builds **new advected paths**
with `dense_paths.py` and pair-local deformable-patch/cycle checks. Geometric
visibility is explicitly only a proxy. The new paths do not equal CoTracker
tracks or fixed-grid flow; do not directly compare their MAEs as the same
estimator. Applied to the 2× source caches, local support / common-mode full
diagnostic MAEs were 0.030696 / 0.030664, still not successful. All records retain
`diagnostic_only`, null scores, source insufficiency, input and cache identities.

### Local appearance and local deformation (DEV; not scoring)

`probe_local_appearance` retains every automatic SIFT location/scale, every
containing source SAM region, the whole-frame control, and factors 1/2/4. It
searches the full translation domain with masked RGB NCC and keeps all three
forward and reverse candidates. The train four-version, frame 2→3 probe finished;
some coupling points relocate correctly, but global mismatches under 8px remain.
Exact reverse closure is explicitly **not** a correctness label.

```bash
uv run --no-sync python -m scripts.counterfactual.probe_local_appearance \
  --manifest <bound-8px-candidates.jsonl> \
  --review configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json \
  --region-run <complete-sam-run> --video-root <official-dev5-native> \
  --video-root <bound-control-and-CF-videos> \
  --candidates video_000004 video_000005 video_000006 video_000007 \
  --starts 2 --lags 1 --shard <0-to-3> --shards 4 \
  --output <fresh-parent>/shard-<0-to-3>

uv run --no-sync python -m scripts.counterfactual.audit_local_appearance \
  --source-run <finished-four-shard-parent> --execution-root <exact-execution-snapshot> \
  --manifest <bound-8px-candidates.jsonl> --region-run <complete-sam-run> \
  --video-root <official-dev5-native> --video-root <bound-control-and-CF-videos> \
  --output <fresh-analysis>

uv run --no-sync python -m scripts.counterfactual.probe_local_deformation \
  --source-run <finished-four-shard-parent> --manifest <bound-8px-candidates.jsonl> \
  --region-run <complete-sam-run> --video-root <official-dev5-native> \
  --video-root <bound-control-and-CF-videos> --factors 4 \
  --shard <0-to-3> --shards 4 --output <fresh-deformation-parent>/shard-<0-to-3>

uv run --no-sync python -m scripts.counterfactual.audit_local_deformation \
  --source-run <finished-deformation-parent> --proposal-run <finished-four-shard-parent> \
  --execution-root <exact-deformation-execution-snapshot> \
  --case-candidate video_000004 --case-source-keys 259 272 310 \
  --output <fresh-deformation-analysis>
```

The optional last case keys only select an explicitly posthoc explanation from
the already completed all-point experiment; they never select inference queries.
The deformation pilot compares translation and affine refinement at factor 4,
retaining all three original proposals and freezing each proposal's visible
support. Lower fitted loss is not held-out motion evidence. It completed with
substantial mismatches remaining; both pilots have `score=null`. See the report
for all denominators, insufficient fits, byte-identical controls and timings.

### Distinct appearance peaks and source self-ambiguity (DEV correspondence)

`probe_local_appearance --search ambiguity` first extracts actual discrete local
maxima, retaining up to 12 hypotheses instead of using all three slots on the
shoulder of a broad NCC peak. It also correlates each source patch with its own
frame, searching remote lookalikes with support intersection at most .25/.5/.75
of the source support. These are relative support-overlap levels, not known
jitter amplitudes. Target/source visibility remains at least .8. All alternatives
and the legacy greedy candidates are saved. The global top candidate generally
does not change merely by retaining more alternatives.

This variant's reverse searches are **NOT RUN**; old reverse closures must not
be transferred. `--factors` explicitly declares the feature-relative support
scales (default remains 1/2/4). The full adjacent-phase train diagnostic uses
factor 4, all automatic points and all their containing SAM regions plus the
whole-frame control. It is a correspondence study, not a full Repair test.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m scripts.counterfactual.probe_local_appearance --search ambiguity --factors 4 \
  --manifest <bound-8px-candidates.jsonl> \
  --review configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json \
  --region-run <complete-sam-run> --video-root <official-dev5-native> \
  --video-root <bound-control-and-CF-videos> \
  --candidates video_000004 video_000005 video_000006 video_000007 \
  --starts 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --lags 1 \
  --shards 4 --shard <0-to-3> --output <fresh-run/shard-N>

uv run --no-sync python -m scripts.counterfactual.audit_local_appearance \
  --source-run <complete-four-shard-run> --execution-root <exact-execution-snapshot> \
  --manifest <bound-8px-candidates.jsonl> --region-run <complete-sam-run> \
  --video-root <official-dev5-native> --video-root <bound-control-and-CF-videos> \
  --no-display --output <fresh-appearance-analysis>

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m scripts.counterfactual.probe_appearance_consensus \
  --source-run <complete-four-shard-run> --source-execution-root <appearance-snapshot> \
  --output <fresh-consensus-run>

uv run --no-sync python -m scripts.counterfactual.audit_appearance_consensus \
  --source-run <complete-consensus-run> --appearance-run <complete-four-shard-run> \
  --execution-root <consensus-snapshot> --output <fresh-consensus-analysis>
```

Consensus uses positive `cross NCC - remote source NCC` margins at the declared
.5 support-intersection level; both linear and squared weights are reported.
Multiple hypotheses from one queried location contribute their maximum weight
at a candidate displacement; feature scales at the same location are averaged,
not extra votes. The integer localization radius is one pixel, not a motion
floor. All/even/odd queried-location folds are explicit; their patches can
overlap, so agreement is not independent physical truth. The independent auditor
recalculates weights, supporters and local maxima directly, without the raster
Hough code. Every result remains `score=null` pending real correspondence and
full-motion validation; the five-source/all-ten-CF acceptance scope is unchanged.

### Three-frame path association and fixed-reference comparison (DEV)

`probe_appearance_paths` consumes one completed all-adjacent-phase appearance
shard. `--mode detected` (default) associates every first-step hypothesis with
every compatible automatic middle-frame query, retaining all second hypotheses.
Its association radius is feature-relative, not a known jitter amplitude. The
snapped first displacement is rechecked; the source template is also compared
with frame three. Paths sharing an endpoint can have different intermediate
motion and are not merged or assigned zero motion.

`--mode reference` removes the need for a middle re-detection: the unchanged
source template is searched globally in frame three, keeping 12 distinct peaks.
All first x reference alternatives are retained. Three pairwise correlations
are computed on the SAME source-index pixels visible at all three placements.
No template replacement, velocity/reversal constraint, even-phase selection,
paired clean input, or construction field is used. Missing evidence stays
missing. This is a correspondence comparison, **not a completed Repair**.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m scripts.counterfactual.probe_appearance_paths --mode <detected-or-reference> \
  --source-shard <complete-appearance-run/shard-N> \
  --source-execution-root <appearance-execution-snapshot> \
  --manifest <bound-8px-candidates.jsonl> --region-run <complete-sam-run> \
  --video-root <official-dev5-native> --video-root <bound-control-and-CF-videos> \
  --output <fresh-path-run/shard-N>

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m scripts.counterfactual.audit_appearance_paths \
  --source-run <complete-path-run> --appearance-run <complete-appearance-run> \
  --execution-root <path-execution-snapshot> \
  --appearance-execution-root <appearance-execution-snapshot> \
  --manifest <bound-8px-candidates.jsonl> --region-run <complete-sam-run> \
  --video-root <official-dev5-native> --video-root <bound-control-and-CF-videos> \
  --output <fresh-path-analysis>
```

Run all four original train-shard identities, not just a favorable case. The
auditor checks every path's membership, complete enumeration, offsets and
ranking, then independently recomputes the pixels for every top1 group. It
does not recompute every path's pixels or certify physical identity. Exact
encoding controls, full phase coverage, missing cases and null scores are
explicit. The two modes have different counting units: query/support paths vs
deduplicated source-support paths; their raw counts are not accuracy measures.

### Disjoint same-region spatial witnesses (DEV; not a motion veto)

`probe_appearance_context` reuses every proposal from a complete adjacent-phase
appearance shard. With core radius `r=factor*SIFT_size`, two witnesses occupy
`(r,2r]` and `(2r,4r]` in the same source SAM region, or the separately retained
whole-frame control. The source supports do not overlap. Witness pixels do not
propose or fit a displacement. SAM/SIFT and target images remain shared, so
this is not statistically independent evidence.

Every original hypothesis plus an explicit nonduplicated zero control is
evaluated. Candidate/zero comparisons use the same visible source pixels, with
the unchanged .8 visibility threshold. NCC, RGB MSE, zero controls and overlap
are retained separately. Core, core+inner, core+outer and core+both rankings use
the minimum involved NCC; missing evidence stays missing. A context can belong
to a different moving part or become occluded, so disagreement is NOT a license
to suppress the seed motion. These outputs deliberately have `score=null`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m scripts.counterfactual.probe_appearance_context \
  --source-shard <complete-appearance-run/shard-N> \
  --source-execution-root <appearance-execution-snapshot> \
  --manifest <bound-8px-candidates.jsonl> --region-run <complete-sam-run> \
  --video-root <official-dev5-native> --video-root <bound-control-and-CF-videos> \
  --output <fresh-context-run/shard-N>

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m scripts.counterfactual.audit_appearance_context \
  --source-run <complete-context-run> --appearance-run <complete-appearance-run> \
  --execution-root <context-execution-snapshot> \
  --appearance-execution-root <appearance-execution-snapshot> \
  --manifest <bound-8px-candidates.jsonl> --region-run <complete-sam-run> \
  --video-root <official-dev5-native> --video-root <bound-control-and-CF-videos> \
  --output <fresh-context-analysis>
```

Run all four original shard identities. The auditor independently reconstructs
all masks by radial inequalities, verifies candidate membership and every
ranking, then rechecks the distinct top1 alternatives of all four variants plus
the zero control from literal RGB pixels. Non-winning pixels are not all
recomputed. Full cohort/phase coverage and exact encoding controls are checked;
none of these checks certify physical identity, invariance or a final Repair.

### Native line/junction correspondence (DEV; not a score)

`probe_line_structure` reads the exact reviewed 20-video cohort, all native
frames and all start positions for lags 1/2/3/4. It retains every OpenCV LSD
segment, globally matches canonical RGB strips in both endpoint directions,
then associates intersections through reciprocal line identities. An intersection
may move along a stationary boundary; no region-wide translation is assumed.
Missing/ambiguous matches remain missing, never zero. The evaluator sees no
pairing, seed, review or manually selected point. No new model weights are used.

```bash
# Four disjoint workers; repeat with --shard 1, 2, 3 and distinct fresh outputs.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m scripts.counterfactual.probe_line_structure \
  --manifest output/dynamic-static-jitter/local-texture-dev5-8px-v1/candidates.jsonl \
  --review configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json \
  --video-root output/dynamic-static-jitter/official-dev5-native \
  --video-root output/dynamic-static-jitter/local-texture-dev5-8px-v1/videos \
  --shard 0 --shards 4 --output /fresh/run/shard-0

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m scripts.counterfactual.audit_line_structure \
  --source-shard /fresh/run/shard-0 --source-shard /fresh/run/shard-1 \
  --source-shard /fresh/run/shard-2 --source-shard /fresh/run/shard-3 \
  --execution-root /exact/execution/snapshot \
  --manifest output/dynamic-static-jitter/local-texture-dev5-8px-v1/candidates.jsonl \
  --review configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json \
  --video-root output/dynamic-static-jitter/official-dev5-native \
  --video-root output/dynamic-static-jitter/local-texture-dev5-8px-v1/videos \
  --output /fresh/audit
```

The completed `line-structure-dev5-8px-v1/shard-{0..3}` outputs bind the immutable
`code-line-structure-dev5-v1` execution snapshot. The independent audit remeasures
all accepted line correlations and all intersection/temporal geometry. It also
checks one predetermined source query against every target per frame pair, not
every rejected query. All 20 inputs / 1080 pairs completed, but sparse geometric
coverage and train correspondence remain insufficient. All scores are null;
see the [full negative result](../../docs/counterfactual-reports/dynamic_static_jitter.md#线段与交点对应完整五源的稀疏几何仍不足).
Changing a distance ratio solely to rescue the inspected example is not accepted.

### Cross-region and crossed region/time explanations (DEV; no suppression)

`probe_image_plane_modes` reuses the complete 20-video 2× torchvision RAFT cache
and each video's own SAM partition. It uses all 15 native transitions and a
32×32 query grid; missing correspondence is not filled with zero. Affine and
three Gaussian spatial bases are compared with an entire region held out.
Spatial extrapolation was insufficient. The subsequent `probe_region_time_modes`
learns temporal bases only from other regions and fits a target point's amplitude
only on other time folds. All three folds are evaluated; no odd/even-frame
selection, known frequency or amplitude is a scoring input. Ranks 0/1/2/3 remain
explicit, including severe negative results for higher ranks.

Both full-cohort probes completed with byte-bound inputs, zero runtime failures
and five exact encoding controls. Train/horse/giraffe interventions have useful
conditional predictive evidence, but bicycle does not show the same result and
snow remains severely under-observed. These are **not** motion-removal results,
new Repair scores, natural-periodic-motion validation or increased G1 coverage.

```bash
uv run --no-sync python -m scripts.counterfactual.probe_image_plane_modes \
  --manifest <bound-8px-candidates.jsonl> \
  --review configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json \
  --source-scores <original-dense-tv-x2-scores.jsonl> \
  --flow-provenance <bound-dense-path-probe-provenance.json> \
  --flow-cache <complete-byte-identical-dense-evidence> --region-run <complete-sam-run> \
  --video-root <official-dev5-native> --video-root <bound-control-and-CF-videos> \
  --output <fresh-image-plane-run>

uv run --no-sync python -m scripts.counterfactual.audit_motion_explanations \
  --source-run <finished-image-plane-run> --execution-root <exact-execution-snapshot> \
  --output <fresh-image-plane-analysis>

uv run --no-sync python -m scripts.counterfactual.probe_region_time_modes \
  --source-run <finished-image-plane-run> --output <fresh-region-time-run>

uv run --no-sync python -m scripts.counterfactual.audit_motion_explanations \
  --source-run <finished-region-time-run> --parent-run <finished-image-plane-run> \
  --execution-root <exact-region-time-execution-snapshot> --output <fresh-region-time-analysis>
```

Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` as in the actual local runs and
preserve immutable execution snapshots. The flow provenance binds the existing
20 cache hashes and source-score SHA, not a regenerated flow model or new weights.
All numerical comparisons use common observed point/time subsets and report the
full denominator; conditional predictive error reduction is not accuracy or a
counterfactual pass. No default CLI Repair changes or automatic commits/pushes.

### Boundary checks and region-protected reversal ablation (DEV; not full scores)

`probe_motion_boundaries --variant boundaries` fits the first three temporal
modes on observed pairs and measures one-sided motion jumps across the video's
own reference SAM boundaries, at grid strides 1/2/4. Missing stencils remain
missing. Smoothness alone is not a nuisance label: real flexible deformation
can have the identical statistic.

`--variant regional-reversal` tests a posthoc DEV candidate. A dominant,
reversing, distributed and cross-region/time-predictable mode can contribute a
removal only in the intersection of two constraints: each region's affine
projection at every time and each point's time-weighted mean velocity stay
unchanged. Degenerate/small regions and unknown pairs permit exactly zero
removal. The gates and projection tolerances are recorded in provenance; there
is no construction amplitude, seed, phase, paired clean or prompt input.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m scripts.counterfactual.probe_motion_boundaries --variant boundaries \
  --source-run <finished-image-plane-run> --output <fresh-boundary-run>

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m scripts.counterfactual.probe_motion_boundaries --variant regional-reversal \
  --source-run <finished-image-plane-run> --output <fresh-reversal-run>

uv run --no-sync python -m scripts.counterfactual.audit_motion_boundaries \
  --source-run <either-finished-run> --parent-run <finished-image-plane-run> \
  --execution-root <that-runs-immutable-code-snapshot> --output <fresh-analysis>
```

Both variants completed all 20 videos. Reversal v1 failed its independent exact
small-region safeguard audit and is preserved; v2 fixes the projection support
and reruns the full cohort, with independent least-squares constraint checks
and exact encoding controls. Six CF conditional speeds decrease but remain far
above their originals. `score=null` everywhere, old valid CF pairs remain 8/10;
conditional model-flow speed is not a full Repair score or an invariance pass.
True train correspondence, snow coverage, natural periodic/nonrigid protection
and the original joint acceptance requirements remain open.

### G1 local-window reproduction

`trajectory.local-dev-v1.json` preserves all dev-v4 thresholds/preprocessing and
uses one-second windows every half second. Transition ownership is fixed by
temporal context, not scores/confidence; overlap never doubles the denominator.
The old motion accumulator remains solely as a controlled reliability ablation.
The first 20-record run had 13 valid / 7 insufficient scores; only 5/10 CF pairs
were valid, so the coverage gate failed. Subsequent decomposition diagnostics
above have not produced a validated complete Repair.

```bash
uv run --no-sync python -m scripts.counterfactual.score_static_jitter \
  --manifest output/dynamic-static-jitter/local-texture-dev5-8px-v1/candidates.jsonl \
  --review configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json \
  --repair-variant local-trajectory \
  --config configs/dynamic-static-jitter/trajectory.local-dev-v1.json \
  --tracker-root /path/to/local/co-tracker-checkout \
  --tracker-weight /path/to/existing/cotracker2.pth \
  --raft-weight /path/to/existing/raft-things.pth --upstream /path/to/locked/VBench \
  --backend both --output output/dynamic-static-jitter/<fresh-g1-run>
```

Select an available CUDA device before the command. The manifest uses its
original absolute video paths; a relocation is allowed only via `--video-root`
with identical file bytes. Native analysis uses the same `--review`, the 8px
construction config, and new scores, writing a fresh output directory. Do not
combine `--review` with `--stress-test` or `--smoke`. Standalone CLI integration:
`dynamic-degree --audit --audit-variant local-trajectory --trajectory-config
<config> --tracker-root <source> --tracker-weight <weight> --video <video>`.

## Historical pilot and reproduction: automatic qualification before user review

User-requested 8 px extension: `construction.local-texture-dev5-8px-v1.json`
uses exactly the same sources, fields, seeds, timing and Repair config, changing
only the displacement amplitude. It keeps the original quality gates. At 8 px,
256×256 fields have minimum Jacobians around 0.44, below the conservative 0.5
qualification threshold, while still nonfolding and within bounds. They remain
**rejected in the original automatic ledger**. To examine all five
videos, use explicit `--stress-test` on the scoring and native-analysis commands.
This development-only option permits verified, nonfolding rejected local warps;
it does not change metric computation or quality status. Analysis retains strict
qualified-only `tables` separately from all-source `stress_tables`. The latter
contains all ten examples subsequently accepted by the user; retain the old
field names as provenance, and cite the separate review when using this cohort.
Neither construction acceptance nor these scores establish Repair success.
Use fresh run names `local-texture-dev5-8px-v1{,-scores,-analysis}`.

For display, `preview_local_texture_jitter --build <build-dir> --output <new-dir>
--amplitudes 8 --seeds 1701 2904 --stress-test` produces original / 8 px seed 1701 /
8 px seed 2904 triptychs. Asterisks identify the historical automatic rejections,
not the subsequent user review decision.

The latest user clarification is **local spatial jitter inside the image, not
additive pixel noise or brightness flicker**. Use the complete original official
video, never a repeated still frame or generated video. This is a five-video dev
pilot, not an independent validation or a new validated default Repair.

- `sources.local-texture-dev5-v1.jsonl`: five distinct dev prompts in existing
  manifest order, generator rotation LaVie / ModelScope / VideoCrafter / LaVie /
  ModelScope. Native MP4 only; no score-based selection.
- `construction.local-texture-dev5-v1.json`: 1/2/4 native-pixel maximum local
  displacement, two seeds, plus original and zero-edit encoding control = 40
  candidates. Main illustration is fixed beforehand at 2 px, seed 1701.
- `local_texture_jitter.py`: smoothly varying, zero-spatial-mean displacement;
  zero image-boundary displacement; alternating zero-temporal-mean phase. Each
  source frame is independently remapped once, without cumulative deformation.
  It adds no RGB noise or whole-frame translation. Linear interpolation can
  alter texture contrast; this is a limitation to inspect, not an intensity-noise
  intervention. Maximum displacement, minimum Jacobian, bounds and the actual
  displacement field are saved. A pixel's color changes only through resampling.
- `trajectory.dev-v4-tracker-denoise.json`: unchanged existing candidate for this
  pilot. No tuning to these five cases. Original and Repair retain different
  units: Boolean dynamic decision vs short-side lengths/s.

Native dimensions, all source frames and timestamps are preserved. Original MP4
files are scored directly, not replaced by the encoding control. Lossless RGB
H.264 counterfactuals are decoded and checked against the intended pixels. Raw
source checksums and every rejected/failed candidate are retained.

```bash
uv run --no-sync python -m scripts.counterfactual.official_video_jitter \
  --sources configs/dynamic-static-jitter/sources.local-texture-dev5-v1.jsonl \
  --config configs/dynamic-static-jitter/construction.local-texture-dev5-v1.json \
  --data-root /path/to/vbench-official-v1 --split dev \
  --output output/dynamic-static-jitter/local-texture-dev5-v1 \
  --ffmpeg /path/to/existing/ffmpeg

uv run --no-sync python -m scripts.counterfactual.score_static_jitter \
  --manifest output/dynamic-static-jitter/local-texture-dev5-v1/candidates.jsonl \
  --config configs/dynamic-static-jitter/trajectory.dev-v4-tracker-denoise.json \
  --tracker-root /path/to/local/co-tracker-checkout \
  --tracker-weight /path/to/existing/cotracker2.pth \
  --raft-weight /path/to/existing/raft-things.pth \
  --upstream /path/to/locked/VBench --backend both \
  --output output/dynamic-static-jitter/local-texture-dev5-v1-scores

uv run --no-sync python -m scripts.counterfactual.analyze_official_video_jitter \
  --manifests output/dynamic-static-jitter/local-texture-dev5-v1/candidates.jsonl \
  --scores output/dynamic-static-jitter/local-texture-dev5-v1-scores/scores.jsonl \
  --config configs/dynamic-static-jitter/construction.local-texture-dev5-v1.json \
  --output output/dynamic-static-jitter/local-texture-dev5-v1-analysis
```

Set the environment and CUDA mask as in the model runtime instructions. On H200
the isolated task code uses `output/<run-id>` instead of the local prefix above.
This batch is development only. Low absolute changes alone do not prove a useful
Repair: retained native motion and score/evidence coverage also matter. Fast
alternation may alias under a scorer's frame subsampling; record native and
sampled cadence when interpreting negative results.

## Historical official-video photometric pilot (superseded construction)

An earlier correction on 2026-09-22 required unaltered VBench 1.0 source videos.
The first implementation used appearance-only noise/flicker. The subsequent user
clarification above supersedes that intervention; its results remain historical.

`construction.official-dev-v1.json` and
`scripts/counterfactual/official_video_jitter.py` preserve native resolution,
frame count, FPS, timing, and content. Outputs use RGB lossless H.264; the builder
checks decoded pixels exactly against intended pixels. Every source has 38
records: original, zero-edit encoding control, and 36 appearance interventions.
Original MP4s are scored directly. GIFs need a verified format adapter because
the locked upstream infer only accepts MP4s or image directories. Timing
disagreement is rejected, never silently resampled. No exposure headroom or
geometric transformation is applied.

```bash
uv run --no-sync --extra models python -m scripts.counterfactual.official_video_jitter \
  --sources configs/dynamic-static-jitter/sources.v1.jsonl \
  --config configs/dynamic-static-jitter/construction.official-dev-v1.json \
  --data-root /path/to/vbench-official-v1 --split dev \
  --output output/dynamic-static-jitter/official-dev \
  --ffmpeg /path/to/existing/ffmpeg
```

Do not reuse synthetic calibration or claim its gates apply to original motion
videos. The primary criterion is now absolute paired invariance, not one-sided
score suppression. See the [current goal](../../docs/plans/2026-09-22-dynamic-static-jitter-goal-prompt.md).
Independent scoring waits for a newly frozen official-video protocol.

## Historical synthetic development (superseded as primary experiment)

These files belong to the new Dynamic Degree static-appearance-intervention goal,
not to the historical FPS-resampling table in `CONSOLIDATED.md`.

- `sources.v1.jsonl`: metadata-only selection, 32 development videos (8 prompts ×
  4 generators) and 120 reserved test videos (30 prompts × 4 generators). One
  hash-ranked replicate per prompt/generator. All variants inherit the source
  split. These are previously scored VBench E0 natural inputs, **not entirely
  unseen natural videos**; the new interventions have not previously been tested.
- `construction.dev-v1.json`: first pilot, original pixel range, speeds 0/1/2/4
  pixels/frame. Preserved after clipping rejections and a no-positive-Origin pilot.
- `construction.dev-v2.json`: image-independent `[48, 207]` exposure headroom on
  every clean and perturbed base, speeds 0/2/4/8. The two alternating seeds now
  start with opposite phases; the first pilot's odd seeds had identical phases.
- `trajectory.dev-v1.json`: uncalibrated CoTracker2 candidate. **Not a frozen
  independent-test protocol**. No epsilon, dynamic threshold, or success claim
  can be inferred from this development configuration.
- `trajectory.dev-v2-highpass.json` and `trajectory.dev-v3-bandpass.json`: explicit
  development descriptor comparisons on the **same** cached CoTracker2 predictions;
  neither is an independently validated final repair. Full pilot evidence and
  failures are retained in the [report](../../docs/counterfactual-reports/dynamic_static_jitter.md).
- `trajectory.dev-v4-tracker-denoise.json`: fixed spatial Gaussian preprocessing
  for the tracker plus the v3 descriptor. A starting candidate for the corrected
  official-video experiment, not a validated repair on official originals.

Each source generates 191 candidates: 37 static (clean + 6 families × 3 doses ×
2 seeds), 148 matched translation controls (4 speeds × 37), and 6 reversal controls
(3 speeds × clean/strong pixel noise). Translation uses a real-image textured
card of constant area over a fixed background; it is not natural articulated
motion. Real natural videos require a separate external-validity evaluation.

Construction rejects and failures remain in `candidates.jsonl`; generated rejected
videos are retained. Input qualification uses no Origin/Repair scores. Encoding:
16 frames, 8 FPS, 256×192, H.264 CRF 12, yuv420p, identical for every variant.
The builder verifies decoded timestamps, shape, perturbation retention, clipping,
and low-frequency structural correlation. These checks are **not human approval**.

## Reproduction

Use new output directories. Do not write frozen `data/`, `results/`, `splits/`, or
`runs/`. The original source root is read-only. Model weights are never downloaded.

```bash
uv run --no-sync python -m scripts.counterfactual.static_jitter build \
  --sources configs/dynamic-static-jitter/sources.v1.jsonl \
  --config configs/dynamic-static-jitter/construction.dev-v2.json \
  --data-root /path/to/vbench-human-preference/videos \
  --split dev --output output/dynamic-static-jitter/dev-build

uv run --no-sync dynamic-degree --audit --audit-variant trajectory \
  --trajectory-config configs/dynamic-static-jitter/trajectory.dev-v1.json \
  --tracker-root /path/to/local/co-tracker-checkout \
  --tracker-weight /path/to/existing/cotracker2.pth \
  --video /path/to/video.mp4 --gpu 0
```

Batch CLI uses `--video-dir` with numbered videos. Use `--both` plus the existing
`VBENCH_AUDIT_UPSTREAM` and `VBENCH_AUDIT_RAFT_WEIGHT` to run the locked official
backend too. The candidate ignores text metadata and never receives a clean
counterpart, construction seed, mask, or motion label.

For incremental experiments, `python -m scripts.counterfactual.score_static_jitter
--help` documents checkpointed per-video scoring, evidence NPZs, SHA-checked
relocation, and identical-provenance resume. `analyze_static_jitter --help`
documents paired tables, prompt-cluster intervals, and development-only calibration.
`--smoke` is development-only and must never be reported as full coverage.

The first independent test must wait for a versioned frozen protocol that records
the selected model/source/checkpoint/config hashes, dev calibration, analysis
manifest, epsilon, and classification threshold. No test-time tuning is permitted.
