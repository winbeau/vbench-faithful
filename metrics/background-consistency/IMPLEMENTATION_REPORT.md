# Background Consistency M1

This package currently delivers the independent uv project and the shared CLI,
metadata, batch, output, and provenance contracts. `vbench` and `audit` return
`status=not_implemented` with `score=null`; no model is imported and no
checkpoint is downloaded. The official adapter will call the pinned upstream
`compute_background_consistency` in a later milestone. The repair module is a
separate package-local boundary and does not import another metric.

M1 verification on 2026-09-20: `uv build --package background-consistency`,
the workspace and isolated-wheel `background-consistency --help` checks, and
the package's five pure tests passed. CUDA, VBench weights, and numeric parity
were not run.
