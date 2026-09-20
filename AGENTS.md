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

本轮审计范围已定为 **11 维**：在办 7 维（`dynamic_degree`、`motion_smoothness`、
`subject_consistency`、`scene`、`human_action`、`spatial_relationship`、`multiple_objects`）
加候选 4 维（`background_consistency`、`temporal_style`、`object_class`、`color`）；
`overall_consistency` 退出本轮（估计器与 `temporal_style` 相同），其包只保留为 legacy，
不删除。范围与候选源码 pin 见 `docs/plans/2026-09-15-dimension-scope-11d.md`，
源码定位见 `docs/paper/unaudited-dimensions-triage.md`。

反事实（metamorphic）审计已完成一轮全量测量，结论与产物见
`docs/counterfactual-reports/CONSOLIDATED.md`。要点：

- **`CONSOLIDATED.md` 的 "Raw result" 一节是最新版本的唯一权威表**：任何维度
  重跑后**原地更新**该表（并同步 `table2.csv`/`SUMMARY.md`/`table2.json` 与
  `README.md`），不要另存平行副本。`SUMMARY.md` 由
  `scripts/counterfactual/summarize.py` 从同一份冻结 scores 树重新生成；表下的
  脚注必须保留：`dynamics_degree` 的注释保存了 v1 归档值（0.8444）与 shipped
  v2 的对照，`multiplt_object` 的注释记录 occlusion-only ladder 下组合值等于
  ordered 半边。
- **数据集** `counterfactual-vbench`：7 维、205 base、815 条派生片段，已发布到
  `xjuIcthub/counterfactual-vbench`；`Overall Consistency` 因需要人工撰写
  prompt 条件（计划 §12.2）本轮未做。构造代码在 `scripts/counterfactual/`。
  `output/` 不入库，因此**已发布数据集的 base 选择固定在
  `configs/counterfactual/bases_published.jsonl`**：单跑 `select_bases.py` 只能
  复现 7 维中的 5 维，检测器相关维度（`multiplt_object`、`subject_consistency`）
  还必须再跑 `pick_detectable.py`；两步合起来可 205/205 复现已发布清单，检测器
  资格计数见 `configs/counterfactual/README.md`。
- **P1 实验已完成**（P2 人工验证、P3 可选实验本轮未做），结果在
  `docs/counterfactual-reports/P1_NATURAL_AND_CONTROL_RUNS.md`：自然偏好集上
  Dynamic v2 与 Motion 方向感知修复都**显著差于 Official**（配对 Δ −0.1155、
  −0.3116，后者低于随机），P1.3 的独立 holdout 只在聚合层验证了 alpha=0.5，
  P1.4 的弱目标面积跨 62×，P1.5 的等面积 box 只是**减轻**而非消除位置混淆
  （配对 CI 含 0）。
- **评分** 在 `h100-server` 上进行：Official VBench 1.0 与 Repair 两个后端，
  6 卡并行、一维一维串行；7 维 × 2 后端覆盖率为 100%。
- **评分环境**（H100 上，均在 `/root/wenbiao_zhao/` 下，不依赖他人目录）：
  解释器 `venvs/vbench/bin/python`；锁定上游 checkout `VBench`（`fd18b3d`，
  由 on-box bundle 克隆）；权重 `models/raft/` 与 `~/.cache/vbench/`，scene
  Repair 用本地 `/root/.cache/clip/ViT-B-32.pt`。**物理卡 6 对 nvidia-smi 可见
  但对 CUDA 不可用，可用范围是 1–5。**
- **不要单独引用 pooled CPA。** 七份独立 review（`<dimension>.review.md`）
  指出：同 rank 族（`fps_resampling`、`filename_invariance`）的 tie-margin CPA
  会因 margin 饱和而恒为 1.0 且对符号不敏感；`temporal_relocation` 必须按敏感性
  半/不变性半分开报；`multiplt_object` 的 ladder 已改为 occlusion-only，组合值
  即 ordered 半边，旧的 tie 判据只作为独立的 control 统计量出现；`scene`、
  `human_action`、`spatial_relationship` 三个族的**族设计本身**不成立，结论
  必须连同 review 一起读。
- **唯一站得住的 ordered Repair 赢是 `multiplt_object`**（occlusion-only ladder，
  +0.2400 [+0.155, +0.330]），其不变性契约用计划 §11.4 的 level predicate 判定，
  两个后端都通过。其余维度都不能无保留地说 "Repair 更好"：
  `subject_consistency` 只有**按半边拆开后**的不变性半边站得住；
  `dynamics_degree` 的 shipped repair 只是把违约镜像（Official `p=+0.49`，
  repair `p=−0.51`），v2 的 `d/dt**0.5` 才把聚合层拉回 `p=−0.011`（但指数是同一
  批数据上的 default，非独立校准）；`motion_smoothness` 的方向感知修复只到
  parity。
- **`motion_smoothness` 的默认值在 `4d53fa2` 变了**（未对齐的像素级方向、top-k
  时间聚合 k=3、0.5/0.5 权重），因此 `CONSOLIDATED.md` 的 motion 行与
  `P1_NATURAL_AND_CONTROL_RUNS.md` 的自然集测量（0.3248 对 0.6364）都是
  **`feeb770` 修订版**的数字，引用前必须重跑；新默认值目前只有合成 ladder 与
  单元测试覆盖。
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
