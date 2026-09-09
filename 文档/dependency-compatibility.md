# Dependency compatibility audit

## Inspected environment

| Item | Observation | Consequence |
|---|---|---|
| uv | `0.11.6` | available |
| default `python` | not found | use the existing vbench interpreter explicitly |
| usable interpreter | Python `3.10.20` at `/home/msy625/miniconda3/envs/vbench/bin/python` | compatible with the planned `>=3.10,<3.13` bound |
| PyTorch | `2.13.0+cu130` | installed in the existing environment, but not validated against the reference fork |
| torchvision | `0.28.0+cu130` | installed, but not validated against the reference fork |
| transformers | `4.33.2` | matches the reference requirements line |
| decord | `0.6.0` | installed |
| OpenCV | `4.11.0` | installed as `cv2` |
| NumPy | `1.26.4` | satisfies the reference `<2.0.0` constraint |
| CUDA | unavailable; `torch.cuda.is_available()` is false and device count is 0 | official model execution is blocked; no CPU fallback |
| Detectron2 | not installed in the existing vbench environment | Spatial Relationship GRiT inference is blocked |

## Reference dependency evidence

The clean reference fork `/home/msy625/vbench1` at commit `13dee903cc97e2633ed6e8f50dea61bc90717935` declares broad dependencies in `requirements.txt`, including NumPy `<2.0.0`, SciPy, OpenCV, PyTorch-related packages, `openai-clip`, decord, and `transformers==4.33.2`. The root `setup.py` performs a CUDA-enabled PyTorch check. Dynamic Degree additionally imports `easydict`, `tqdm`, and bundled RAFT code.

These observations are not enough to select final uv versions: the reference requirements are not a complete per-metric dependency closure, model code includes vendored third-party code, and the installed PyTorch build cannot be exercised without CUDA. No model dependency was invented or added to a lock file.

## Current decision

- The workspace packages currently depend only on `audit-core` so their CLI and protocol can be tested without downloading model stacks.
- Official backends fail explicitly until their per-metric closure, license handling, CUDA combination, and weight hashes are approved.
- Spatial Relationship additionally requires the locked fork's vendored GRiT/CenterNet2 source and a compatible Detectron2 build. The reference requirements comment out an unpinned Detectron2 Git dependency, so no version is invented in this workspace.
- `uv.lock` is intentionally absent. Generate it only after the model dependency conflict is resolved and the chosen interpreter/environment has been validated.
- Model files must remain outside Git and `uv.lock`, with SHA-256 recorded in run metadata.
