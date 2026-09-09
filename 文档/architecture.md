# Architecture

The root project owns the workspace contract and explicitly depends on every metric and `audit-core`. Each metric owns its CLI and algorithm adapter; metrics never import one another. `audit-core` owns only shared input validation, metadata matching, device parsing, scheduling, result schemas, and output writing.

```text
CLI -> audit-core.inputs -> audit-core.devices -> metric backend
                                      |                 |
                                      v                 v
                               run metadata       per-video result
                                      \                 /
                                       -> summary/output
```

The current implementation is a dependency-light scaffold. `audit` returns `not_implemented`. Official model backends remain explicit blockers until the reference dependency closure, CUDA compatibility, model weights, and licenses are verified.
