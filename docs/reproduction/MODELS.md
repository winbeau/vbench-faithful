# Selected model release

The current trained models live in the public Hugging Face model repository
[`winbeau/vbench-faithful`](https://huggingface.co/winbeau/vbench-faithful).
[`model-release.json`](../../configs/reproduction/model-release.json) pins its
commit, file sizes, SHA-256 hashes, and local installation paths. All seven trained
weights preserve the paper selection in [`release.json`](../../configs/reproduction/release.json).

## Hub layout

```text
adapters/<dimension>/adapter_config.json
adapters/<dimension>/adapter_model.safetensors
heads/dynamic_degree/aligned.pt
configs/paper-methods.json
configs/qwen-base.json
configs/training/<dimension>.json
configs/dynamic_degree/*.json
provenance/selection/<dimension>.json
provenance/training/
provenance/evaluation/
provenance/dependencies.json
licenses/vjepa2.txt
manifest.json
README.md
```

| Trained asset | Selected version | Step |
|---|---|---:|
| Scene adapter | v8 | 300 |
| Human Action adapter | v9 | 300 |
| Spatial Relationship adapter | v8 | 600 |
| Multiple Objects adapter | v6 | 900 |
| Object Class adapter | Paper final | 300 |
| Color adapter | Paper final | 300 |
| Dynamic head | aligned-v1 | 300 |

The first four use the best retained checkpoint by a 24-example development
probe, with latest-step tie breaking. Action's higher-scoring step 150 had already
been pruned. Object, Color, and Dynamic are fixed final selections without a
development checkpoint comparison. Original selection records and training
provenance remain in the model repository.

The seven trained weights total **1,048,141,156 bytes**. The shared Qwen3-8B base,
tokenizer, frozen V-JEPA encoder, and other visual backbones are referenced with
pinned identities. Subject and Background repairs and the seven accelerated
dimensions require no newly trained weights. Each dimension uses the same
selected implementation as before this layout change.

## Download and evaluate

Use the [README quick start](../../README.md#quick-start) for the complete
evaluation runtime. Its existing command also downloads the new model release:

```bash
output/bootstrap/bin/python scripts/restore_paper_runtime.py \
  --output output/runtime --downloads output/runtime-downloads \
  --allow-official-fallback
```

For research input restoration with selected models, use:

```bash
output/bootstrap/bin/python scripts/prepare_reproduction.py \
  --output output/reproduction --include-models --allow-official-fallback
```

The downloader installs adapters into `adapters/<dimension>/` and the Dynamic
head into `models/dynamic_degree/aligned.pt` inside the chosen output directory.
This keeps existing evaluator and training paths compatible. It downloads
each adapter's configuration and weights, and places Spatial's training
configuration beside its adapter to retain the four-direction parsing contract.
Tokenizers come from the shared Qwen base. Files with different existing contents
are preserved and reported as an error. The restoration receipt records the selected model repository and commit;
runtime `--resume` also checks the selected model manifest identity. Evaluation
preflight and cache keys include PEFT configurations and Spatial's four-direction
configuration. Missing or changed metadata is rejected before model inference.

Frozen research inputs still come from `xju-arlab/vbench-repair`. Original training
source archives, Dynamic metadata, and the unmodified frozen V-JEPA checkpoint
remain pinned in the earlier `xju-arlab/vbench-model` release. These references
preserve historical evidence and shared dependencies. They do not substitute
different trained checkpoints for this release.

To fetch only this Hub repository, use `hf download winbeau/vbench-faithful
--local-dir output/models`. For a reproducible download, add `--revision` with the
full commit from `model-release.json`. This command alone does not install shared
base models, video backends, or runtimes. The Hub model card includes a PEFT loading
example; complete scoring uses the project's schemas, visual evidence, and formulas.

## Release verification

Published model commit: [`91d6f98`](https://huggingface.co/winbeau/vbench-faithful/commit/91d6f98bbd30f9eec232e7820d3c65a449ad4801).
GitHub integration: [`f7b298d`](https://github.com/winbeau/vbench-faithful/commit/f7b298da24c91df944e42f822a7aa2d192d4507e).
The [publication receipt](../publication/faithful-model-release.json) records:

- All 46 uploaded files passed remote size/content-identity checks. Anonymous
  access works; all 14 runtime files were downloaded with an empty Hub file cache,
  installed through the recovery mapper, and SHA-256 verified.
- Six safetensors files each contain 504 FP32 tensors / 43,646,976 parameters.
  The seven-tensor, 51,393-parameter Dynamic head passed strict CPU loading.
- CPU validation: 1,146 passed, 3 skipped; locked uv checks and all 16 package
  help entries passed.
- On H100 GPU 3, a fresh Spatial run used the same 32 two-second videos
  (16 frames at 8 FPS), the selected checkpoint, and the restored configuration.
  It completed in 57.31 seconds including setup, with 32/32 defined scores and
  mean 0.11677368474221925. All 11 previously defined scores matched exactly;
  the other 21 were newly scored by the selected parser, without score imputation.

The earlier H100 installation lacked `training_config.json`, which silently
selected the legacy Spatial prompt/schema. Its historical 11/32 coverage and
all-16 timings remain historical evidence. This release restores the original
selected four-direction configuration and verifies it before execution. The
published weight bytes and scoring formulas are unchanged. This validation ran
one GPU dimension; it did not retrain models or establish new all-16 timings.

The H100 test tree was `13d9e82670ed9b09627b5777ad8448715c00d43b`, using VBench
`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`. Its temporary snapshot is identified in
the receipt; the only untracked entry was the launcher `.venv` symlink. Results
are under `/root/wenbiao_zhao/vbench-repair-eval-20261008/faithful-model-release-20261008/eval/`.
