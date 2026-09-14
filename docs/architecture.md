# Architecture

The root project owns the workspace contract and explicitly depends on every metric and `audit-core`. Each metric is an independent distribution under `metrics/<kebab-name>/src/<import_name>`; metrics never import one another. `audit-core` owns only shared input validation, metadata matching, device parsing, scheduling, result schemas, and output writing.

```text
CLI -> audit-core.inputs -> audit-core.devices -> metric backend
                                      |                 |
                                      v                 v
                               run metadata       per-video result
                                      \                 /
                                       -> summary/output
```

The default uv environment is dependency-light and does not install model runtimes. Scientific imports needed by pure algorithms are direct metric dependencies; model closures are available through each package's `models` extra and the root `models` aggregate. `uv sync --locked --group test` adds the CPU scientific test group. Official model execution remains unverified where CUDA, model weights, compiled dependencies, or upstream runtime conditions are unavailable. Third-party model constructors may download their default pretrained assets when invoked, so controlled runs must provide local checkpoints and external runtime configuration.

The authoritative upstream identity is configured in `configs/upstream.toml` and resolved at runtime. Outputs are written under `output/<metric>/<backend>/<run-id>/` by default, with provenance recording the code, lock, input, model and upstream identities when available.
