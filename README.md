# VBench Audit

VBench 1.0 的可复现审计工作区。本轮范围 **11 维** = 已实现的 7 维 + 4 个候选维度
（范围决定见 [`docs/plans/2026-09-15-dimension-scope-11d.md`](docs/plans/2026-09-15-dimension-scope-11d.md)）。
2026-09-20 已另接通 background consistency 的官方与候选修复后端，验证进展见下文；
原范围决定时的七维实现清单保留如下。
Object Class / Color 也已完成独立包后端、共享 Qwen3-8B 的独立 LoRA 与四级消融，
见[实现与复现](docs/object-color-repair.md)及[两维实测报告](docs/counterfactual-reports/object_color_repair_20260920.md)。
Object 的 metadata 改写与 Color 的可见性响应分别报告；Color test 仅 5 个合格基底，
Official 同义控制缺失，人工语义审核尚未完成，不据此宣称总体修复优越。
每个维度都是独立包，Python import 名使用下划线，发行包和命令使用 kebab-case：

```text
metrics/                     # 已实现的 7 维
├── dynamic-degree/          → dynamic_degree / dynamic-degree
├── motion-smoothness/       → motion_smoothness / motion-smoothness
├── subject-consistency/     → subject_consistency / subject-consistency
├── scene/                   → scene / scene
├── human-action/            → human_action / human-action
├── spatial-relationship/    → spatial_relationship / spatial-relationship
└── multiple-objects/        → multiple_objects / multiple-objects
packages/audit-core/         → vbench_audit_core
packages/audit-models/       → vbench_audit_models

后续 4 维（已在 configs/upstream.toml 钉住源码，均已有独立包）：
background_consistency、temporal_style、object_class、color
```

`overall_consistency` 已退出本轮范围：它与 `temporal_style` 是逐行同一份 ViCLIP 估计器，
且其 prompt 条件需要人工撰写（八维计划 §12.2）。`metrics/overall-consistency/` 及其 CLI、
测试暂时保留为 legacy，未删除。
未入选的另外 4 维（`temporal_flickering`、`appearance_style`、`aesthetic_quality`、
`imaging_quality`）的源码定位与不入选理由见
[`docs/paper/unaudited-dimensions-triage.md`](docs/paper/unaudited-dimensions-triage.md)。

## 环境

仓库固定 Python 3.11.14（`.python-version`），uv 0.9.17 在 CI 固定，依赖由已提交的 `uv.lock` 管理：

```bash
# Development (model-free)
uv sync --locked

# CPU scientific/contract tests (separate overlay)
uv sync --locked --group test
uv pip install --python .venv --index https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
uv run --no-sync --group test pytest tests metrics
```

模型运行另起一个同步环境执行：

```bash
uv sync --locked --extra models
```

不要在 CPU 测试 overlay 之后再次同步 models extra，以免替换 CPU wheel。

模型权重、CUDA 驱动和 Detectron2/GRiT 等外部构建不由 uv.lock 提供。本机 CPU 测试不等于真实模型 parity；Subject 的 H100 全量 parity 已在下述扩展报告中逐条核验，其余维度以各自运行报告为准。某些第三方模型构造器在传入默认 pretrained 配置时可能联网下载权重；正式运行应预置本地权重并使用对应的本地路径参数。纯算法、输入输出合约及 CLI help 不需要权重。

## VBench 1.0 官方视频目录

VBench 1.0 的 Dynamic Degree 与 Motion Smoothness 复用 Subject Consistency 的 prompt suite 和 sampled videos。官方人类标注中的视频路径也指向各生成器的 `subject_consistency/`；不能据 `question_en` 的残留文本重新判定维度。远端数据根目录使用以下布局：

```text
<data-root>/<generator>/subject_consistency/
<data-root>/<generator>/dynamics_degree -> subject_consistency
<data-root>/<generator>/dynamic_degree  -> subject_consistency
<data-root>/<generator>/motion_smoothness -> subject_consistency
```

其中 `<generator>` 为 `lavie`、`modelscope`、`cogvideo`、`videocrafter`。`dynamics_degree` 是 VBench 官方维度名，`dynamic_degree` 是本仓库 CLI/包别名；两者均链接到同一只读视频目录。可用 `scripts/link_shared_official_dimensions.sh <data-root>` 建立并校验软链接。此共享关系只适用于视频输入，Dynamic Degree、Motion Smoothness 和 Subject Consistency 仍分别读取各自的人类偏好标注并运行各自评分公式。

## CLI

所有入口都必须显式选择 `--vbench`、`--audit` 或 `--both`，并在 `--video` 与 `--video-dir` 中二选一。示例：

```bash
uv run dynamic-degree --vbench --video /data/video_000.mp4
uv run dynamic-degree --audit --video-dir /data/videos --gpu
uv run dynamic-degree --both --video-dir /data/videos --gpu 0,2,4
```

裸 `--gpu` 和不带值的默认选择是当前 CUDA 可见逻辑设备 0；显式列表拒绝重复或越界编号，不会静默切换 CPU。默认结果写入仓库根 `output/<metric>/<backend>/<run-id>/`，每次运行使用新 run-id；`--output DIR` 可替换输出基目录。输入 `data/`、`results/`、`splits/`、`runs/` 为冻结研究内容，不修改、不重算、不删除。

## 规范与文档

开发边界和测试要求见 [`AGENTS.md`](AGENTS.md)，贡献、uv extras 与提交规则见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。架构、CLI、依赖和上游映射分别见 [`docs/architecture.md`](docs/architecture.md)、[`docs/cli.md`](docs/cli.md)、[`docs/dependency-compatibility.md`](docs/dependency-compatibility.md) 与 [`docs/upstream-mapping.md`](docs/upstream-mapping.md)。实施计划在 [`docs/plans/2026-09-14-workspace-refactor.md`](docs/plans/2026-09-14-workspace-refactor.md)。

本轮实际通过的检查及未验证范围见[重构验收记录](docs/plans/2026-09-14-workspace-refactor-verification.md)。补充实验计划与 H200 执行口径见[补充实验计划](docs/plans/2026-09-14-supplementary-experiments.md)。当前范围（11 维）见[十一维范围决定](docs/plans/2026-09-15-dimension-scope-11d.md)；其中 7 在办维度的总体实验协议见[八维实验协议](docs/plans/2026-09-14-experiment-plan-8d.md)；反事实数据集（VBench-CF）的构造口径见[反事实数据集](docs/counterfactual-dataset.md)。

补充实验的可直接执行命令、H200 四卡计时、8 卡恢复记录和当前主表结果见
[补充实验运行手册](docs/supplementary-experiments.md)。主表交付物位于
`figures/supplementary_main_table.{csv,tex,pdf,png}`；逐视频研究输出仍写入
被 Git 忽略的 `output/supplementary_20260914/`，避免把大量中间缓存提交到仓库。

Subject 的表示层修复与完整背景糊化实验见 [数据、协议与复现说明](docs/subject-repair.md)。
[Subject 背景干预稳定性修复](docs/plans/2026-09-20-subject-stability-goal.md)已按用户最新验收要求收尾：
开头背景糊化时，Origin 明显变化、repair 变化约 0.01，且保留主体变化响应；人类偏好为辅助诊断。
[本轮真实运行报告](docs/counterfactual-reports/subject_stability_20260920.md)：官方 **720 条、全部 72 个 prompt、每生成器 180 条**的主实验已完成，241 条数值构造全部评分、0 运行失败。预定 240 条主分析的 Origin/repair 平均绝对变化为 **0.120035/0.010596**，主体平均降分 **0.099319**；背景联合成功 74/240，加主体降分 ≥0.05 为 70/240。用户看到结果后接受“0.01 左右”的均值表现，原严格 ≤0.01 统计仍保留为未通过。479 条构造拒收、最坏 repair 分差 0.149715 均保留。用户随后要求快速收尾，补充单帧评分在 204/241 条完成时停止，部分输出完整保留，不作为全量结果。旧 60 条单首帧的 0.102377/0.001275 与偏弱主体响应均保留，不能替代新主实验。
此前[官方扩展实验](docs/counterfactual-reports/subject_official_extension_20260920.md)已完成 1440 条自然视频、2160 对人工偏好：
test 1290 对上 origin 为 58.53%，原有聚合 repair 为 59.61%，自动定位＋隔离裁剪为 48.76%。
初轮候选整体退化；不能由两条背景示例宣称修复成功。所有官方/原聚合分数分别与历史值在 1e-6 内对齐。
后续保持粗掩码、改用 CLS 并在主体证据不足时回退 origin：统一零平局容差下，test 为 57.05%，origin 为 58.45%。
34 条完整背景干预的平均分差降到 0.053575（origin 0.060218），但配对改善区间跨零，主体响应仍有不足；尚未达到优秀修复。
背景构造也已扩到 288 条候选，其中 34 条通过自动门槛；类别不支持和构造门槛限制了覆盖。
后续接受大体正确的主体粗定位，不因少量边缘误差反复暂停实验；重点检查表示、缺失帧处理和整体效果。
当前 v2 保留主体像素、模糊主体掩码的全部补集，独立 MobileSAM 评分定位与建库掩码隔离；
历史镜像盒协议只归档重放。真实运行结果与质量限制单独报告，未验证项目保留 NOT RUN。
当前 [7 条试跑](docs/counterfactual-reports/subject_region_discrimination_v2.md) 因图像与建库掩码质量不足，
不作为正式主实验；用户只保留演讲者和游泳者为候选，后续先做原图质量筛选。
编码前主体隔离与规范化裁剪已作为 `subject_isolated` 候选变体实现，
[开发消融](docs/counterfactual-reports/subject_isolation_development.md) 显示中位背景扰动减小，
但均值、最坏情况和独立验证仍未达标，尚不宣称修复效果优秀。
后续 [咖啡、吉他两条新候选](docs/counterfactual-reports/subject_isolation_quality2.md) 已完成真实评分：
背景平均绝对分差从 origin 的 0.01449 降至 0.00591，局部时窗主体干预 6/6 降分；
咖啡评分掩码仍漏掉部分身体，两条样本不足以支持总体修复成功的结论。

此前的 [Background Consistency 修复与主体糊化实验](docs/plans/2026-09-20-background-repair-goal-prompt.md)：
糊化主体、保留背景，对比 origin 与 repair 的干预前后分数，同时验证背景变化响应与自然偏好表现。
background 已完成真实 CLIP 后端与独立验证，[正式报告](docs/counterfactual-reports/background_holdout_20260920.md)
覆盖 1,040 条自然测试视频、1,560 对偏好及 188 条基底的 1,504 个干预版本。
完整主体糊化平均绝对分差从 origin 0.016240 降至 repair 0.006733（降低 58.54%）；
自然偏好准确率从 52.12% 提高到 61.22%，三个时窗平均的背景响应保留 88.80%。
18 项冻结门槛全部通过，默认 `repair` 已指向背景 patch 池化与全帧对方案；真实 CLI parity 误差为 0。
仍有单片段失败与粗定位局限，所有旧候选负结果保留在[开发报告](docs/counterfactual-reports/background_development_20260920.md)。

后续构造语义检查发现旧干预存在误选和漏分，以上数值通过不能证明所有编辑正确。
按用户要求，background 原生开发集 680 条已用同一 SegFormer + GrabCut 流水线扩大到 104 类重构：
全空从 450 降至 89，数值构造通过从 7 增至 49，待复核项目显式保留。
评分端的 COCO80 与既有 MobileSAM 自动补充分支仍属开发候选；完整结果、逐步复核范围和未达标项见
[空掩码审计与类别扩展](docs/counterfactual-reports/background_empty_masks_20260920.md)。
后续复用 GRiT 区域描述与 MobileSAM 的评分定位候选，已完成 680 条自然开发视频和 1,020 对偏好：
origin 52.35%、现有 repair 57.84%、COCO80 57.94%、Caption v2 56.67%。
Caption v2 的检出覆盖改善，但未证明对现有 repair 非劣，因此保持开发候选；
详见[定位修复与全量自然评估](docs/counterfactual-reports/background_caption_localizer_20260920.md)。
104 类重构后的 49 条数值候选在暂停时已逐次复核 30 条：12 条可用、18 条拒收、19 条待复核。
十条已完成干预对照，origin / 现有 repair / Caption v2 的全片主体糊化 MAE 为
0.011578 / 0.003054 / 0.002704；origin 达到 0.10 或 0.20 的比例均为 0/10。
仍有非空但错位的评分掩码，类别扩展改善了构造覆盖，尚未达到新的联合修复目标。
按用户要求暂停实验，新增两条接受项尚未评分；[阶段总结与紧凑证据](docs/counterfactual-reports/background_repair_checkpoint_20260920.md)
记录暂停状态、全部已完成结果和剩余工作。多物体 union 仅完成输入探测，未晋升默认。
