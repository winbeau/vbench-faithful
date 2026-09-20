# Object Class M1

This package currently delivers the independent uv project and the shared CLI,
metadata, batch, output, and provenance contracts. `vbench` and `audit` return
`status=not_implemented` with `score=null`; no GRiT model is imported and no
checkpoint is downloaded. The official adapter will call the pinned upstream
`compute_object_class` in a later milestone. Label normalization and instance
evidence remain package-local work for that milestone.

M1 verification on 2026-09-20: `uv build --package object-class`, the
workspace and isolated-wheel `object-class --help` checks, and the package's
five pure tests passed. CUDA, GRiT weights, and numeric parity were not run.
