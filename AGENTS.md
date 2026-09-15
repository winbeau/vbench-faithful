# VBench Audit 研发约定

## 开始前

先阅读 `README.md` 与 `docs/plans/2026-09-14-workspace-refactor.md`。本仓库工作区固定使用 Python 3.11.14（见 `.python-version`）和锁定的 `uv.lock`。

## 目录与边界

- `metrics/<metric>/src/<import_name>/` 只包含该维度的 CLI、适配器、算法和模型封装；metric 之间不能互相导入。
- `packages/audit-core/` 只放输入、元数据、设备、调度、输出和 provenance 等共用基础设施；共用模型适配放在 `packages/audit-models/`，不得放评分公式。
- `configs/` 保存可审阅的配置示例；模型权重、驱动和外部 checkout 不入库。
- `data/`、`results/`、`splits/`、`runs/` 是冻结研究输入/结果，本轮不得删除、重算、改名或改写内容。历史报告中的旧路径可以作为历史事实保留。
- `docs/` 记录当前行为；旧的根级论文计划保留并链接到当前计划。
- VBench 1.0 的 `dynamics_degree` 与 `motion_smoothness` 官方 sampled videos 复用每个生成器的 `subject_consistency/`。远端输入可建立 `dynamics_degree -> subject_consistency`、仓库别名 `dynamic_degree -> subject_consistency` 和 `motion_smoothness -> subject_consistency` 软链接；不得复制成内容分叉的目录，也不得将该视频共享关系误用为评分公式或人类标注共享。

## 当前状态（2026-09-15）

反事实（metamorphic）审计已完成一轮全量测量，结论与产物见
`docs/counterfactual-reports/CONSOLIDATED.md`。要点：

- **数据集** `counterfactual-vbench`：7 维、205 base、815 条派生片段，已发布到
  `xjuIcthub/counterfactual-vbench`；`Overall Consistency` 因需要人工撰写
  prompt 条件（计划 §12.2）本轮未做。构造代码在 `scripts/counterfactual/`。
- **评分** 在 `h100-server` 上进行：Official VBench 1.0 与 Repair 两个后端，
  6 卡并行、一维一维串行；7 维 × 2 后端覆盖率为 100%。
- **评分环境**（H100 上，均在 `/root/wenbiao_zhao/` 下，不依赖他人目录）：
  解释器 `venvs/vbench/bin/python`；锁定上游 checkout `VBench`（`fd18b3d`，
  由 on-box bundle 克隆）；权重 `models/raft/` 与 `~/.cache/vbench/`，scene
  Repair 用本地 `/root/.cache/clip/ViT-B-32.pt`。**物理卡 6 对 nvidia-smi 可见
  但对 CUDA 不可用，可用范围是 1–5。**
- **不要单独引用 pooled CPA。** 七份独立 review（`<dimension>.review.md`）
  指出：同 rank 族（`fps_resampling`、`filename_invariance`）的 tie-margin CPA
  会因 margin 饱和而恒为 1.0 且对符号不敏感；混合族（`temporal_relocation`、
  `weakest_object_visibility`）必须按敏感性半/不变性半分开报；`scene`、
  `human_action`、`spatial_relationship` 三个族的**族设计本身**不成立，结论
  必须连同 review 一起读。
- **当前没有任何维度支持无保留的"Repair 更好"**：`subject_consistency` 按半边
  拆分后最站得住；`dynamics_degree` 的 shipped repair 只是把违约镜像
  （Official `p=+0.49`，repair `p=−0.51`），v2 的 `d/dt**0.5` 才把聚合层
  拉回 `p=−0.011`（但指数是同一批数据上的 default，非独立校准）。
- 真实模型、CUDA 与权重 parity 对冻结 E0 基线**尚未验证**，上述数值是首轮
  测量值，不是复现的官方基线。

## 改动与验证

源码变更后运行受影响包的纯算法测试和合约测试；接口或工作区变更还要运行 `uv lock --check`、`uv sync --locked`、CPU torch overlay 与 `uv run --no-sync --group test pytest tests metrics` 及八个入口的 `--help`。真实模型、CUDA 和权重 parity 未验证时必须在报告中明确写出，不下载权重，也不修改上游 checkout。

H200 正式实验使用物理 4–7 卡时，一个维度完成、校验并合并后再开始下一个维度；单维度可拆成四个隔离 shard，每个进程只看到一张卡并使用逻辑 `cuda:0`。计时报告至少记录视频组定义、视频数量、每个视频的媒体时长、四卡墙钟时间、代码 SHA、上游 SHA、设备和输出路径。

若用户明确要求利用空闲显存加速，允许在不终止既有进程的前提下扩展到更多
物理卡；本轮 Motion Smoothness 的剩余 718 条使用 H200 0–7 卡完成，原有
`sglang` 进程未被 kill。跨卡、跨批次恢复必须按 `video_uid` 去重并检查完整
覆盖后才能更新汇总表。

CLI 新维度可复制对应 metric 目录，保持 `src/` 布局、独立 `pyproject.toml`、直接依赖声明和 `tests/`。模板入口应先实现 `--vbench/--audit/--both`、`--video/--video-dir`、元数据和默认输出，再接入算法，不将另一个 metric 作为运行时依赖。

不自动 commit 或 push；普通修复按用户授权执行。commit 标题使用 Conventional Commits（中文描述可以）：`type(scope): 描述`，重大变更用 `type(scope)!: 描述`，type 为 `feat/fix/refactor/test/docs/build/ci/chore`。提交前可运行 `./scripts/check_commit_message.py "$(git log -1 --pretty=%s)"`，或安装 `.githooks/commit-msg` 到本地 hooks；CI 只检查 PR 标题或新提交范围，不回溯检查旧历史。
