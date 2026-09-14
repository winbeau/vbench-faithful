# Workspace 重构验收记录

日期：2026-09-14。6-astra 先编写[实施计划](2026-09-14-workspace-refactor.md)，Luna 完成实现，主代理复核并执行本记录中的最终验收。

## 交付状态

- 本仓库已创建并切换至 `winbeau`；修改保留在工作树，没有 commit 或 push。新增文件仅设置 intent-to-add 以便 Git 显示重命名和完整 diff，没有暂存提交内容。
- 官方参考已 clone 到同级 `../VBench`，origin 为 `https://github.com/Vchitect/VBench.git`，HEAD 为 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`，工作树保持干净。当前来源记录于 `configs/upstream.toml`；没有宣称它与历史 fork 的 `13dee903...` 等价。
- 原八个 metric 保留为 `metrics/<kebab-name>/src/<python_name>/` 独立 workspace 成员；公共基础设施在 `packages/audit-core/`，共用 RAFT 封装在 `packages/audit-models/`。metric 不再通过另一个 metric 获取 RAFT 实现。
- 建立 `AGENTS.md`、贡献和提交规范、可选 commit-msg hook、提交模板、EditorConfig、CPU CI、模型 extras、配置示例及当前架构/CLI 文档。
- 八个入口统一必选模式和输入参数。目录视频按数字排序、轮转分配给非空 GPU 分片，以 spawn 并发运行；默认输出为仓库根 `output/<metric>/<vbench|audit>/<run-id>/`，both 共用 run-id。
- Subject Consistency 的 CLI 官方汇总固定使用单进程 transition 加权公式，不再因请求 GPU 数变化而切换公式；逐视频公式保持不变，formula version 已标识此变化。

## 实际验证

环境为 Python 3.11.14、uv 0.9.17。最终 `uv.lock` 解析 102 个包（包括可选模型依赖），默认开发同步不安装模型运行时。CPU torch 为单独测试 overlay，不属于默认锁定环境。

| 检查 | 结果 |
| --- | --- |
| `uv lock --check` | 通过 |
| `uv sync --locked --group test` | 通过 |
| 默认环境 `uv pip check --python .venv` | 通过 |
| 安装 `torch==2.14.0+cpu` overlay 后再次 `uv pip check` | 通过 |
| `uv run --no-sync --group test pytest --import-mode=importlib -q --tb=short -rs tests metrics` | **297 passed, 3 skipped** |
| 八个 `uv run --locked <metric> --help` | 全部通过 |
| 八个真实 console entrypoint 的 help、缺参和互斥检查 | 48 项通过 |
| dynamic / motion 的 `uv run --locked --package ... ... --help` | 通过 |
| 真实 spawn + Barrier 并发、分片恢复顺序、坏分片/崩溃隔离、启动中断清理 | 纳入已通过测试 |
| 共享 RAFT fake-wrapper、固定上游 AST/纯函数及适配器对照 | 纳入已通过测试，不等于真实模型 parity |
| `ruff check --select F821`、Python compileall、全部 shell 脚本 `bash -n` | 通过 |
| `git diff --check` | 通过 |
| 冻结 `data/`、`results/`、`splits/`、`runs/` 与 HEAD 比较 | 无修改；其中 data/results/splits 的 36 个 tracked 文件保留 |

还对迁移前后的 46 个 metric 测试文件做了 AST 名称比对：原有测试保留；两项来源准入测试随官方来源协议调整了名称/断言。删除的是八个重复命名的测试包标记 `tests/__init__.py`，避免 pytest 错误复用其他 metric 的同名测试模块。

## 验收边界

三项 skip 分别是 Dynamic Degree、Human Action、Spatial Relationship 的真实 GPU/权重 opt-in 对照。本机只有一张 RTX 4060 Laptop GPU（8188 MiB），没有本项目模型权重；没有下载权重、执行真实模型评测或做三卡对照。多进程协议通过不代表 GPU 数值一致性或性能已经通过实测。

模型 extras 已按当前官方推理 import 补齐已知 Python 依赖并通过 resolver，但没有完整安装/验证所有模型环境；Detectron2/GRiT 编译、模型资产及 CUDA 组合仍需配置。Decord 0.6.0 的上游 wheel 元数据警告、第三方构造器可能联网取 pretrained 资源，以及历史研究脚本的外部存储路径，见[依赖说明](../dependency-compatibility.md)。

本轮交付统一目录、依赖声明、CLI 参数和并发调度，并不提供断点续跑。各 metric 的扩展 provenance 字段及中断结果输出仍有差异；计划中完整统一这些扩展字段、保留中断前全部已完成结果的要求，不应视为已验收功能。
