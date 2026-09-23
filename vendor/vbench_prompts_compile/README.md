# Inference and training source snapshot

This captures the current local vbench-prompts-compile source, including uncommitted
repair interfaces used by the research workspace. `source-manifest.json` records
every file hash; the Git HEAD alone does not describe the dirty snapshot.

Install the pinned environment with uv and the train extra when loading adapters.
Use scripts/predict.py or scripts/batch_predict.py with the appropriate task,
base-model path and one adapter directory. Video scores additionally require the
VBench Repair video backends and their separately managed pretrained weights.
