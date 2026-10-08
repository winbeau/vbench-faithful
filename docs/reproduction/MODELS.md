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
runtime `--resume` also checks the selected model manifest identity.

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
