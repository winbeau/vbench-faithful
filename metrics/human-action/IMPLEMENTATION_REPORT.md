# Human Action implementation report

## Official audit summary

VBench 1.0 论文将 Human Action 定义为：判断视频中的人物是否准确执行文本指定动作；测试 prompt 从 Kinetics-400 选取 100 个低语义重叠动作。锁定源码路径为 `compute_human_action -> load_dimension_info -> human_action`。`load_dimension_info` 虽读取 prompt，但调用方只保留 `video_list`；真实 target 由 `basename.lower().split('-')[0].split('person is ')[-1].split('_')[0]` 得到。因此 prompt/metadata 不参与最终 label matching。MP4 由 Decord 解码，`get_frame_indices(..., sample="middle")` 均匀采 16 帧，短视频重复最后帧；随后 resize 256、center crop 224、转为 `C,T,H,W` 并按 ImageNet 均值方差归一化。模型是 UMT ViT-L/16 的 400 类闭集动作分类器，不是 video-text matching。模型产生一次全 clip sigmoid logits，取 Top-5，概率先四舍五入到 4 位，只保留 `>=0.85` 的类别，再与 filename target 精确匹配。每视频输出 bool，dataset score 是 bool 均值。论文也明确描述 16 帧、UMT/K400、Top-5 和 0.85；本地 PDF 是无效 404 HTML，因此通过官方 arXiv HTML 副本核对。

## Confirmed evidence

Source-confirmed facts are the filename target extraction, discarded prompt record, 16-frame middle sampling, UMT/K400 closed-set classification, rounded Top-5/0.85 gate, video boolean, and dataset mean.

The supplied controlled experiment confirms an implementation artifact that is exposed by the custom-input path, not a semantic definition failure: ten validated positives score 10/10 only under correct filenames and 0/10 under wrong or neutral names, while explicit metadata targets give 10/10 for all three filename conditions. These historical results were not rerun here.

The source also confirms an information bottleneck: one clip-level prediction is immediately collapsed to a thresholded boolean. It does not by itself prove errors for action context, preparation, brief execution, or weak temporal grounding.

## Core contract

Use the explicit prompt/metadata action as the semantic target, evaluate target-specific visual execution evidence, and preserve both evidence strength and its temporal distribution.

## Core failure

Target-conditioned execution evidence is not preserved end to end: target binding is replaced by a filename shortcut, then a single whole-video K400 prediction is collapsed through rounded Top-5/0.85 matching before continuous or temporally localized evidence can be inspected.

Primary manifestations are: (1) confirmed filename-controlled target changes; (2) a source-confirmed inability to distinguish different temporal evidence profiles that produce the same final boolean. Whether this causes human-disagreement on context/preparation/brief execution remains experimental.

## Selected repairs

- R0 prerequisite - explicit `target_action`, with a strict exact-K400 prompt fallback; no filename fallback.
- R1 core - preserve raw target-class sigmoid probability, rank, and Top-5 diagnostics instead of using only accepted-set membership.
- R2 lightweight extension - run the same UMT on at most four deterministic contiguous temporal windows and report source-frame-count-weighted mean target probability plus coverage at the preserved 0.85 threshold.

R1/R2 produce structured evidence. No mapping from mean probability and coverage to one benchmark scalar has been independently calibrated, so Audit returns `score=null`. Action-state transition/FSM logic was rejected because no cross-action state protocol or confirmed failure supports it. No VLM, detector, pose model, optical flow, or ensemble was added.

## Preserved / Modified / Added

Preserved: locked UMT architecture and weight, Kinetics-400 labels, 16-frame middle sampler, resize/crop/normalization, sigmoid output, Official Top-5/0.85/exact-match behavior, Official filename extraction, workspace CLI/GPU/output protocol.

Modified in Audit only: filename target becomes explicit metadata/prompt target; boolean-only evidence becomes continuous target-class evidence; one whole-video observation gains deterministic temporal windows.

Added: upstream identity gate, structured diagnostics, target parser, temporal aggregation, per-video failure isolation, multi-GPU round-robin workers, and opt-in real parity test.

## Tests

The Human Action suite has 21 tests: 20 pass and the real CUDA/UMT parity test is skipped unless explicitly enabled. Tests cover locked SHA, Official filename parsing and rounded Top-5 thresholding, source-helper-equivalent 16-frame sampling, corrupt-video isolation, filename dependency/invariance, exact target binding, continuous target evidence outside Top-5, temporal mean/coverage, brief-versus-persistent synthetic behavior, structured Audit evaluator output, CLI single/batch/both/GPU/run-id/interruption, and metadata-before-CUDA validation.

These tests use synthetic probabilities or fake backends and do not establish UMT's real action-recognition ability.

The package and `audit-core` build as wheels from a temporary source copy, install into a temporary Python 3.10 environment, and expose a working `human-action --help` console entry point. No build artifacts were generated in the repository.

## Known limitations

- CUDA is unavailable in the current WSL session, so no real UMT inference or parity run is claimed.
- Audit costs one full-clip pass plus up to four window passes; it adds no model but can approach five times Official inference compute.
- Four windows and the Official-derived 0.85 window-coverage threshold are not independently calibrated.
- UMT remains a K400 closed-set classifier. Continuous/window evidence cannot guarantee actual execution rather than correlated context.
- Very short windows repeat frames under the preserved 16-frame sampling behavior.

## Remaining hypotheses

Real counterfactual experiments are still required for context versus execution, preparation versus execution, brief/partial versus complete execution, and weak temporal grounding. Human labels are required before claiming a human-alignment improvement.
