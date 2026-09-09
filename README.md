# vbench-audit

This repository is the workspace scaffold for extracting eight VBench 1.0 dimensions into independently maintainable packages.

Current state:

- CLI, input validation, metadata matching, GPU validation, output schemas, and audit stubs are implemented across the workspace.
- Spatial Relationship now includes a locked-fork official adapter and an ordered-role, identity-first audit backend. Its model-free contracts pass; real GRiT execution remains blocked by the currently missing CUDA/Detectron2/weight runtime.
- Dynamic Degree now includes the locked RAFT official adapter plus background-aware affine decomposition, timestamp-normalized motion rate, structured continuous intensity/coverage, prompt routing, and Audit-only short-video compatibility diagnostics. Its real RAFT parity remains blocked by unavailable CUDA.
- Human Action now includes the locked filename-driven UMT/K400 Official baseline and an explicit-target Audit backend with continuous target-class and lightweight temporal-window evidence. Real UMT parity remains blocked by unavailable CUDA.
- Other official model backends remain intentionally blocked until their reference and dependency closures are verified.
- No model weights are downloaded and no `uv.lock` is generated yet.

See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md), [`文档/architecture.md`](文档/architecture.md), and [`文档/upstream-mapping.md`](文档/upstream-mapping.md).
