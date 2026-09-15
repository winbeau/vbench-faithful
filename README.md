# VBench Audit

VBench 1.0 八个维度的可复现审计工作区。每个维度都是独立包，Python import 名使用下划线，发行包和命令使用 kebab-case：

```text
metrics/
├── dynamic-degree/          → dynamic_degree / dynamic-degree
├── motion-smoothness/       → motion_smoothness / motion-smoothness
├── subject-consistency/     → subject_consistency / subject-consistency
├── scene/                   → scene / scene
├── human-action/            → human_action / human-action
├── spatial-relationship/    → spatial_relationship / spatial-relationship
├── overall-consistency/     → overall_consistency / overall-consistency
└── multiple-objects/        → multiple_objects / multiple-objects
packages/audit-core/         → vbench_audit_core
packages/audit-models/       → vbench_audit_models
```

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

模型权重、CUDA 驱动和 Detectron2/GRiT 等外部构建不由 uv.lock 提供。本机当前只有一张 RTX 4060（8 GiB），没有本项目权重；真实模型 parity 和多卡验收因此尚未验证。某些第三方模型构造器在传入默认 pretrained 配置时可能联网下载权重；本轮没有下载权重，正式运行应预置本地权重并使用对应的本地路径参数。纯算法、输入输出合约及 CLI help 不需要权重。

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

本轮实际通过的检查及未验证范围见[重构验收记录](docs/plans/2026-09-14-workspace-refactor-verification.md)。补充实验计划与 H200 执行口径见[补充实验计划](docs/plans/2026-09-14-supplementary-experiments.md)。八维审计的总体实验协议见[八维实验协议](docs/plans/2026-09-14-experiment-plan-8d.md)；反事实数据集（VBench-CF）的构造口径见[反事实数据集](docs/counterfactual-dataset.md)。

补充实验的可直接执行命令、H200 四卡计时、8 卡恢复记录和当前主表结果见
[补充实验运行手册](docs/supplementary-experiments.md)。主表交付物位于
`figures/supplementary_main_table.{csv,tex,pdf,png}`；逐视频研究输出仍写入
被 Git 忽略的 `output/supplementary_20260914/`，避免把大量中间缓存提交到仓库。
