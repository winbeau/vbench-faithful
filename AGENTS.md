# VBench Audit 研发约定

## 开始前

先阅读 `README.md` 与 `docs/plans/2026-09-14-workspace-refactor.md`。本仓库工作区固定使用 Python 3.11.14（见 `.python-version`）和锁定的 `uv.lock`。

## 目录与边界

- `metrics/<metric>/src/<import_name>/` 只包含该维度的 CLI、适配器、算法和模型封装；metric 之间不能互相导入。
- `packages/audit-core/` 只放输入、元数据、设备、调度、输出和 provenance 等共用基础设施；共用模型适配放在 `packages/audit-models/`，不得放评分公式。
- `configs/` 保存可审阅的配置示例；模型权重、驱动和外部 checkout 不入库。
- `data/`、`results/`、`splits/`、`runs/` 是冻结研究输入/结果，本轮不得删除、重算、改名或改写内容。历史报告中的旧路径可以作为历史事实保留。
- `docs/` 记录当前行为；旧的根级论文计划保留并链接到当前计划。

## 改动与验证

源码变更后运行受影响包的纯算法测试和合约测试；接口或工作区变更还要运行 `uv lock --check`、`uv sync --locked`、CPU torch overlay 与 `uv run --no-sync --group test pytest tests metrics` 及八个入口的 `--help`。真实模型、CUDA 和权重 parity 未验证时必须在报告中明确写出，不下载权重，也不修改上游 checkout。

CLI 新维度可复制对应 metric 目录，保持 `src/` 布局、独立 `pyproject.toml`、直接依赖声明和 `tests/`。模板入口应先实现 `--vbench/--audit/--both`、`--video/--video-dir`、元数据和默认输出，再接入算法，不将另一个 metric 作为运行时依赖。

不自动 commit 或 push；普通修复按用户授权执行。commit 标题使用 Conventional Commits（中文描述可以）：`type(scope): 描述`，重大变更用 `type(scope)!: 描述`，type 为 `feat/fix/refactor/test/docs/build/ci/chore`。提交前可运行 `./scripts/check_commit_message.py "$(git log -1 --pretty=%s)"`，或安装 `.githooks/commit-msg` 到本地 hooks；CI 只检查 PR 标题或新提交范围，不回溯检查旧历史。
