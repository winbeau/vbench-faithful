# CLI contract

All metric entry points use:

```text
uv run <metric> (--vbench | --audit | --both)
                (--video FILE | --video-dir DIR)
                [--output DIR] [--gpu [IDS]]
                [--metadata FILE] [--seed INT]
```

The scaffold validates mutual exclusion, video naming and ordering, metadata lookup, GPU syntax, and output isolation. A vbench invocation requires CUDA; no CPU fallback is provided. An audit invocation writes an explicit `not_implemented` result and exits non-zero.
