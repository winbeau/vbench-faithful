# Overall Consistency Implementation Report

## 1. Official source verified

The locked reference is `/home/msy625/vbench1`, remote `https://github.com/msy625/VBench.git`, branch `master`, commit `13dee903cc97e2633ed6e8f50dea61bc90717935`. VBench 2.0 and beta implementations were excluded. The bundled `VBench1.0_paper.pdf` is a 284-byte HTML 404 response, so it was not treated as paper evidence; the implementation facts below come from the locked VBench 1.0 source.

Direct sources:

- `vbench/overall_consistency.py`
- `vbench/utils.py`
- `vbench/third_party/ViCLIP/viclip.py`
- `vbench/third_party/ViCLIP/viclip_text.py`

Official flow: Decord loads the video; `read_frames_decord_by_fps(..., num_frames=8, sample="middle")` uniformly partitions the full timeline and takes each interval midpoint, padding short videos with the final sampled frame. `clip_transform(224)` resizes/center-crops and applies CLIP mean/std normalization. ViCLIP (`ViClip-InternVid-10M-FLT.pth`) encodes one global video feature and the complete prompt with its native 32-token context behavior. Both features are L2-normalized, and their matrix-product scalar is cosine similarity. There is no threshold. Per-video similarities are averaged arithmetically; distributed aggregation gathers video records and recomputes the same equal-weight per-video mean.

## 2. Research mechanism implemented

Core contract: Overall Consistency should preserve holistic video-text alignment while remaining explicitly sensitive to critical semantic violations.

Core hypothesis: a single holistic similarity may permit strongly matched semantic content to compensate for a localized critical violation. This implementation operationalizes that hypothesis; it does not establish that the failure occurs empirically.

## 3. Repair

The Repair keeps the official global branch and official ViCLIP representation:

```text
S_global = cosine(e_V, e_P)
s_k      = cosine(e_V, e_condition_k)
S_mean   = mean(s_1, ..., s_K)
S_min    = min(s_1, ..., s_K)
S_cond   = alpha * S_mean + (1 - alpha) * S_min
S_repair = (1 - lambda) * S_global + lambda * S_cond
```

Defaults are `alpha=0.5` and `lambda=0.5`; both are configurable CLI parameters, recorded in diagnostics, and are not claimed to be tuned or optimal. Negative cosine values are retained. The video encoder runs once per video; full-prompt and condition text use the same official ViCLIP text encoder and checkpoint.

## 4. Condition source

`semantic_conditions` at the metadata root has priority, followed by the same field inside `dimension_metadata`. Strings and `{text, type}` records are accepted. Empty strings are removed and duplicates are removed in first-seen order without weakening phrase binding. Invalid containers or entries generate a diagnostics warning and conservatively fall back to `[full_prompt]`. Missing or zero usable conditions also use full-prompt fallback; in that case Repair equals Official global similarity mathematically.

No LLM, parser, specialist evaluator, detector, or additional backbone is used.

## 5. Diagnostics

Repair output records `video_path`, `prompt`, `official_global_score`, `semantic_conditions` with text/type/source/score, `condition_scores`, `condition_mean`, `condition_min`, `weakest_condition`, `condition_score`, `repair_score`, `num_conditions`, `condition_source`, `warnings`, `alpha`, and `lambda`.

## 6. Tests

Focused tests cover locked-source identity and constants, official adapter/full-prompt behavior, middle sampling, L2 normalization, cosine (including negative values), dataset mean, explicit/fallback/malformed/duplicate conditions, one-time video encoding, exact violation-sensitive formulas, synthetic counterfactual contracts, and CLI modes/config/output.

Validation command:

```text
UV_CACHE_DIR=/tmp/vbench-audit-uv-cache uv run --with pytest pytest -q 指标/overall_consistency/tests
```

First-pass result: `23 passed, 3 warnings`. The warnings are upstream deprecation warnings for `pkg_resources` and old timm import paths. Package import, locked official module import, and CLI `--help` also pass. The real ViCLIP checkpoint is absent from the configured cache, so real CUDA/video parity was not run; adapter, direct locked sampling/normalization helpers, cosine, aggregation, and same-encoder contracts were tested without downloading model weights. `setuptools<81` is pinned because the unchanged VBench 1.0/OpenAI CLIP source imports the removed `pkg_resources` API.

## 7. Compatibility

Official mode directly calls the locked `vbench.overall_consistency.overall_consistency()` function. Repair calls the same source module's frame reader, transform, `get_vid_features`, and `get_text_features`. Dataset output uses audit-core schemas and preserves equal-weight per-video aggregation. No separate software-level repair mode was introduced; the second-pass corrections are recorded below.

`audit-core changed: no`.

## 8. Known limitations

- Reliable experiments should provide explicit, proposition-preserving semantic-condition annotations; automatic decomposition is intentionally absent.
- ViCLIP capability and calibration are unchanged.
- `alpha` and `lambda` defaults are not validated optima.
- Temporal partial validity is intentionally not repaired.
- Fixed ViCLIP text context behavior may truncate long prompts exactly as Official does.
- No empirical superiority over VBench is currently claimed.

## 9. Second-pass audit

The second pass found and fixed three implementation issues without changing the frozen method: full-prompt fallback now preserves the exact Official prompt string instead of stripping boundary whitespace; mathematically identical single-condition/equal-score and alpha/lambda boundary cases return the exact operand rather than accumulating a one-ULP blend difference; and diagnostics now serialize `fallback_reason` plus the weakest condition's text, type, source, and actual score. Tests were added for the exact full-prompt invariant, alpha/lambda validation and boundaries, aggregation monotonicity, deterministic first-occurrence weakest ties, top-level versus `dimension_metadata` precedence, malformed fallback, JSON metadata propagation, K=1/K=4 scoring, and fallback/weakest diagnostics.

Final focused test result: `38 passed, 3 warnings, 28 subtests passed`. The warnings remain unchanged upstream `pkg_resources` and timm deprecation warnings.

Official mode remains a direct call to the locked VBench function. The audit-core was not changed. Multi-device CLI execution uses audit-core round-robin assignment but evaluates shards sequentially in one process; it preserves equal per-video weighting and does not claim distributed parallelism. The workspace lockfile diff is not solely attributable to Overall Consistency: besides its required upstream runtime dependencies, it contains resolver reconciliation for other workspace members, including Human Action and Motion Smoothness. No unrelated dependency was added to Overall Consistency's own `pyproject.toml`.

`REAL_MODEL_PARITY = NOT RUN`: no `ViClip-InternVid-10M-FLT.pth` exists in the searched local project/cache locations. A future parity smoke can be run directly with the locked checkpoint and a small video through `overall-consistency --mode official`; that mode itself invokes `vbench.overall_consistency.overall_consistency()` and emits its per-video score. Remaining empirical risks are ViCLIP capability/calibration, semantic-condition annotation quality, and the unvalidated default alpha/lambda values, not a reason to alter the frozen formula before counterfactual validation.
