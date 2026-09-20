# Temporal Style M1

This package currently delivers the independent uv project and the shared CLI,
metadata, batch, output, and provenance contracts. `vbench` and `audit` return
`status=not_implemented` with `score=null`; no ViCLIP/BPE model is imported and
no checkpoint is downloaded. The official adapter will call the pinned
upstream `compute_temporal_style` in a later milestone. Style text extraction
and ranking remain package-local work for that milestone.

M1 verification on 2026-09-20: `uv build --package temporal-style`, the
workspace and isolated-wheel `temporal-style --help` checks, and the package's
five pure tests passed. CUDA, ViCLIP/BPE weights, and numeric parity were not
run.
