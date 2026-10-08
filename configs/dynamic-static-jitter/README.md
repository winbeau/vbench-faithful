# Dynamic aligned-v1 training and construction

Only the selected aligned-v1 recipe and its required probe/anchored initialization stages are retained. Historical candidate recipes are available in the [pre-cleanup tree](https://github.com/winbeau/vbench-faithful/tree/8939690/configs/dynamic-static-jitter). Parent-stage validation failures remain documented; retaining the parent does not select it for inference.


## Completed evaluation of the accepted aligned head on450 sources

`vjepa-aligned450-v1.json` pins the accepted `aligned-v1` checkpoint and the same
1800 existing videos:450 official originals,900 local8px CFs,450 encoding
controls. **1800/1800 completed, zero failures**, fresh decode/encoder forward
for every input; all feature hashes equal previous caches, oldjoint replay error0,
all450 controls exact. Origin is the same hash-pinned previous result, not a new
RAFT run. No training, remapping, reconstruction, download or calibration45 read.

Current Repair **0.629251→0.623035**, delta−0.006216, MAE0.021731; Origin
**0.680000→0.857778**. Natural preference base/CF1701/CF2904 each144/171.
The batch-increase check passes, but28/900 CFs drop by more than0.1 and the worst
drop is0.377932. Not per-video invariance; original DEV gates remain unchanged.
Default unchanged; no further tuning. [Full current results](../../docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-aligned450).

Task root: `/data/chenjiayu/dynamic-structural-motion-20260922/vjepa-aligned450-v1/`.
From its `code/` snapshot, use the H200 Python interpreter recorded below and
set `VBENCH_AUDIT_WORKSPACE` to that absolute `code/` directory, the same
PYTHONPATH and one physical4–7 UUID as in the training recipe. For each shard0–3,
run on its own UUID with a fresh output directory:

```bash
python -m scripts.counterfactual.score_vjepa_aligned450 \
  --config configs/dynamic-static-jitter/vjepa-aligned450-v1.json \
  --previous-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-expansion450-v1 \
  --probe-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-probe-v1 \
  --trained-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-aligned-v1 \
  --video-root /data/chenjiayu/wenbiao_zhao/vbench-official-v1 \
  --shard 0 --output ../reproduction-scores/shard-0
```

To reproduce statistics on the already completed, copied run from the local
repository root (fresh outputs; never overwrite original analysis):

```bash
.venv/bin/python -m scripts.counterfactual.summarize_vjepa_aligned450 \
  --config configs/dynamic-static-jitter/vjepa-aligned450-v1.json \
  --root output/dynamic-static-jitter/vjepa-aligned450-v1 \
  --previous-root output/dynamic-static-jitter/vjepa-expansion450-v1 \
  --human-pairs data/processed/pairwise_master_split.csv \
  --output output/dynamic-static-jitter/vjepa-aligned450-v1/analysis-reproduction

.venv/bin/python -m scripts.counterfactual.audit_vjepa_aligned450 \
  --root output/dynamic-static-jitter/vjepa-aligned450-v1 \
  --previous-root output/dynamic-static-jitter/vjepa-expansion450-v1 \
  --human-pairs data/processed/pairwise_master_split.csv \
  --output output/dynamic-static-jitter/vjepa-aligned450-v1/audit-reproduction.json
```

The saved `analysis/summary.json`, `pairs.jsonl`, `human_pairs.jsonl` and
`independent-audit.json` retain all means, intervals, human ties and failures.
Four-card wall time approximately93.85s including loading/parity; single-shard
times93.30/91.55/89.85/85.85s. Media:1800×2s. No new independent holdout claim.

## Origin-scale supervision inside model training (`vjepa-aligned-v1`)

The user's “flicker” here means **the existing 8px local texture-coordinate
jitter**, not luminance flicker. `vjepa-aligned-v1.json` freezes one 300-step
continuation of the anchored head. Retain all existing supervision and add the
squared difference between the **TRAIN210 native sigmoid mean** and the
**TRAIN210 unmodified Origin mean**, weight1. No per-video binary imitation,
posthoc mapping, validation teacher in the loss, TEST450 fitting, or new weights.
All eleven parent DEV gates remain, plus DEV native mean within0.05 of Origin.
Only one fixed final checkpoint; no sweep or validation checkpoint selection.

The 270 original MP4s were actually rescored with pinned official RAFT:
TRAIN210 Origin **0.638095**, DEV60 Origin **0.800000**, zero failures. This
population difference is retained; the test mean0.68 is not a training target.
Training uses existing2160 feature views; no new counterfactual construction.
The user's five still/motion controls are evaluated only after saving the head.

Completed: DEV60 Repair **0.657326→0.651620**, MAE0.010285; still/pan8/pan32
user control **0.007435/0.351916/0.554858**, still+jitter0.013868. Original gates
10/12 pass: natural preference17/23 and DEV mean gap0.142674 fail. The user
subsequently **accepted scores in the0.6 range and requested closure**. Stop
tuning; this is a post-result scale acceptance, not a retroactive gate pass.
Do not edit the frozen config or receipts. At training handoff new-head TEST450
was NOT RUN; the subsequently authorized evaluation is now completed above.
Default unchanged. [Training result](../../docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-aligned).

Task root: `/data/chenjiayu/dynamic-structural-motion-20260922/vjepa-aligned-v1/`.
From its `code/` snapshot, use the recorded H200 Python interpreter. Set the
workspace explicitly (required because the isolated snapshot has no root
packaging files), PYTHONPATH and a single available physical4–7 GPU UUID:

```bash
export VBENCH_AUDIT_WORKSPACE=/data/chenjiayu/dynamic-structural-motion-20260922/vjepa-aligned-v1/code
export PYTHONPATH=.:packages/audit-core/src:packages/audit-models/src:metrics/dynamic-degree/src
export CUDA_VISIBLE_DEVICES=GPU-490b4a76-6210-31b9-4e03-838a113cf5f4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python -m scripts.counterfactual.score_vjepa_origin_dev \
  --config configs/dynamic-static-jitter/vjepa-aligned-v1.json \
  --probe-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-probe-v1 \
  --video-root /data/chenjiayu/wenbiao_zhao/vbench-official-v1 \
  --upstream /data/chenjiayu/dynamic-static-jitter-20260922/VBench \
  --raft-weight /data/chenjiayu/.cache/vbench/raft_model/models/raft-things.pth \
  --shard 0 --output ../origin-dev-reproduction/shard-0
```

Repeat for shards1/2/3 on their own UUIDs. All four must complete before training;
the original run's `origin-dev/` is immutable. A reproduction uses a fresh teacher
and output directory, not an overwrite:

```bash
python -m scripts.counterfactual.train_vjepa_aligned \
  --config configs/dynamic-static-jitter/vjepa-aligned-v1.json \
  --probe-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-probe-v1 \
  --anchored-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-anchored-v1 \
  --origin-root ../origin-dev-reproduction \
  --diagnostic-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-static-frame-v1 \
  --output ../training-reproduction
```

Audit the actual run's copied small artifacts from the local repository root:

```bash
.venv/bin/python -m scripts.counterfactual.audit_vjepa_aligned \
  --root output/dynamic-static-jitter/vjepa-aligned-v1 \
  --probe-root output/dynamic-static-jitter/vjepa-probe-v1 \
  --anchored-root output/dynamic-static-jitter/vjepa-anchored-v1 \
  --output output/dynamic-static-jitter/vjepa-aligned-v1/audit-reproduction.json
```

`verify_vjepa_aligned` independently reloads the two checkpoints on H200 and
replays all480 DEV views plus five controls from hash-checked cached tokens.
Neither audit/replay is an additional training trial or a fresh encoder run.
The initial four Origin workers failed at import because the explicit workspace
was absent; the failed launches are recorded in `origin-launch-failure.md`.
They produced no scores/model and changed no source, parameter or gate on retry.

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
has a unit test. Recovery **completed all 300 updates**, with no checkpoint
selection. The new head fixes the user control (still/pan8/pan32:
**0.008533/0.134990/0.293819**) and achieves 60/60 on each of three controlled
motion orderings in DEV60. However, natural preference falls **20/23→18/23**,
below the predeclared ≥19/23 gate: **overall acceptance failed (10/11 pass)**.
The model is not promoted to default. At training handoff, new-head TEST450 was
NOT RUN; the subsequent user-authorized frozen evaluation is completed above.
The prior 0.518089→0.516319 numbers belong only to the old frozen joint head.

Actual checkpoint: `training-chunked/anchored.pt`, SHA-256
`8e10add01e050417baeccd191d80525afa3a7f6d60754de549703427823dd045`.
Training receipt, full scores and 300-step curves are in `training-chunked/`.
`independent-audit.json` verifies 1350 new feature receipts, 4320 score values,
all loss terms and gates; this verifies integrity, not scientific acceptance.
`reload-verification/receipt.json` records an independent strict checkpoint
reload: all 4320 DEV predictions and five controls repeat exactly (zero latent
and score error); all seven parameter tensors changed. No training in the replay.

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

Independent CPU statistics audit from the repository root (use a fresh output):

```bash
.venv/bin/python -m scripts.counterfactual.audit_vjepa_anchored \
  --root output/dynamic-static-jitter/vjepa-anchored-v1 \
  --probe-root output/dynamic-static-jitter/vjepa-probe-v1 \
  --output output/dynamic-static-jitter/vjepa-anchored-v1/audit-reproduction.json
```

`verify_vjepa_anchored` independently reloads both heads on the model host and
checks all 2160 cached views (4320 predictions), plus the five user controls.
It does not train or re-encode video. Mask one free GPU and run from the `code/`
snapshot with the same PYTHONPATH; use a fresh output directory:

```bash
python -m scripts.counterfactual.verify_vjepa_anchored \
  --root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-anchored-v1 \
  --probe-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-probe-v1 \
  --diagnostic-root /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-static-frame-v1 \
  --output /data/chenjiayu/dynamic-structural-motion-20260922/vjepa-anchored-v1/reload-reproduction
```

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
[Full protocol and acceptance scope](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/plans/2026-09-22-dynamic-structural-motion-repair-goal.md#vjepa-validation-plan).

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
