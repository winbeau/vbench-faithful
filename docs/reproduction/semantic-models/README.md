# vbench-prompts-compile

VBench 四维语义接口修复的独立开发与实验仓库。四维原始/变换数据的 Origin、Repair-rule、Repair-model 三方案矩阵已完成：29,660 项、88,980 条评分记录。最新结果统一见[当前四维实验汇总](docs/deterministic/current-summary/README.md)；首轮共同官方公式实验保留在[历史研究报告](docs/deterministic-experiments-report.md)、[历史表格目录](docs/deterministic/matrix-v1/README.md)与[执行验收](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/EXECUTION-2026-09-20.md)。

新增[消融、小模型与泛化评估](docs/deterministic/ablation-v1/README.md)：同数据 0.6B/8B、base/SFT/接口/Scene caption 消融，共 4,584 次预测；架构与抗捷径 WG-CC 指标已记录。发现并修复 Action 接口越界覆盖，当前默认 `repair-v2.1`，旧 `repair-v2` 保留。未见 Action 类别映射、Scene 长文本和 Objects 正例损失仍不支持全面泛化结论；报告保留负结果及后验修复边界。

Spatial 后续已完成[有向后端修复 v2](docs/deterministic/spatial-repair-v2/README.md)：Repair 使用主体/客体配对、有符号几何和冠词归一化；Origin 保持官方行为。原始几何单向成立的 106 视频，Repair 框镜像四格为 0.4442→0→0→0.4442，106/106 符合预期。默认评分启用 v2，`--spatial-backend legacy-official` 可重现首轮结果。

Action 已完成[同义接口修复 v2](docs/deterministic/action-repair-v2/README.md)：固定 v9 adapter 加入既有同义契约归一化。60 对同义表达、1,200 视频中，Origin 0.8892→0，Repair 0.8892→0.8892，逐视频保持一致。推理/评分默认启用；`--action-interface legacy-model` 保留旧链路。此结果是声明词典覆盖域内的接口修复，未重训或证明未见同义表达泛化。

Objects 已实现[相邻帧确认 v2](docs/deterministic/objects-repair-v2/README.md)：Repair 的当前实体框须得到相邻帧同名框支持（IoU≥0.5）。275 个模型复核不可见端点中，仍有正分的视频从 29 减至 15；原始均分 0.2843→0.2617，可见帧检出也有损失。被过滤的检测单列为弃权，全部明确阴性的端点仍为 246；这是抑制孤立误报的工程取舍，持续幻觉尚未解决。评分默认启用，`--objects-backend legacy-official` 可重现旧版。

| 系统 | 输入 | 输出 | 组织 |
| --- | --- | --- | --- |
| Scene | 原始 prompt + 单帧 Tag2Text caption | supported / contradicted / insufficient 三选一 | 独立模型 |
| Spatial | prompt | relationships 三元组数组 | 共享 backbone + spatial LoRA |
| Human Action | prompt | actions 数组，名称对齐 K400 | 同 backbone + action LoRA |
| Multiple Objects | prompt | entities 数组，不限两个 | 同 backbone + objects LoRA |

外部指定维度，硬路由激活一个 adapter。不让 LLM 自选维度；不强制三个解析任务增加末端匹配模型。数量、几何和最终评分由后端算法处理。

## Reproducible development

Python **3.11.14**、uv **0.9.17**；直接依赖 exact pin，运行/开发传递依赖及哈希由 `uv.lock` 冻结。Hatchling 隔离构建闭包另外通过 `build-constraint-dependencies` 固定版本（构建依赖不声称具有 lock 中的制品哈希锁定）。

```bash
uv --version  # 必须为 0.9.17
uv python install 3.11.14
uv lock --check
uv sync --locked
uv run --no-sync pytest
uv run --no-sync vbench-prompts --help
uv run --no-sync vbench-prompts --task spatial
```

最后一个命令只展示 schema 示例，不做推理。基础开发不安装 torch、不访问模型仓库。

训练环境准备（不下载权重、不启动训练）：

```bash
uv sync --locked --extra train
uv run --no-sync python -c 'import torch, transformers, trl, peft; print(torch.__version__)'
```

CUDA wheels 固定为 PyTorch cu126。driver 和 GPU 不由 lockfile 管理。不要在正在训练的环境重新执行不含 `--extra train` 的同步命令，它会移除训练依赖。

## Layout and plans

- [主计划索引](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/plans/README.md)：分阶段门禁、多个子计划。
- [冒烟实施主计划](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/plans/05-smoke-master.md)：数据清洗、训练编码、分层验收三个子计划（已实施，见冒烟报告）。
- [环境与远端部署](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/plans/01-environment.md)
- [数据与标注](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/plans/02-data.md)
- [实际数据盘点](docs/data/README.md)：标注原料、MovieGen提示词、许可与可训练性
- [teacher 试标报告](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/teacher-pilot-report.md)：19次请求的候选、拒绝原因与协议结论
- [冒烟报告](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/smoke-report.md)：T0–T3 验收、真实底座100步资源与未完成项
- [运行手册](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/runbook.md)：造数据/训练命令、路径约定与踩坑记录
- [决策记录](docs/decisions.md)：已确认的底座/Scene/数据/下载决策与待决策项
- [正式训练报告](docs/formal-training-report.md)：8B 四维模型的数据、资源、dev 结果与限制
- [模型卡](docs/model-cards.md)：四个模型的用途、许可与不能声称的内容
- [边界与弥补方案](docs/limitations-and-remediation.md)：五条已知边界的现状、弥补路径、验收与执行顺序
- [阶段性总结](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/STATUS-2026-09-20.md)：四维训练/数据集/效果、确定性数据盘点、Origin-Repair 现状与阻塞
- [训练曲线](docs/curves/8b/)：loss / token accuracy / dev exact match（SVG+CSV+HTML）
- [LoRA 训练](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/plans/03-training.md)
- [评测与集成](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/plans/04-evaluation.md)
- [接口与研究约束](docs/contracts.md)
- [复用来源](docs/upstream.md)
- [验证记录](docs/verification.md)
- [代理约定](AGENTS.md)

## Remote

私有 GitHub：`git@github.com:winbeau/vbench-prompts-compile.git`。

服务器：`ssh rtx4090`；路径：`~/wenbiao_zhao/vbench-prompts-compile`。

初始化完成后更新：

```bash
git pull --ff-only
../tools/uv-0.9.17/uv sync --locked --extra train
```

远端使用仓库专用 SSH 只读 deploy key；不复制个人 token。精确部署与验收结果见环境计划和验证记录。本轮 GPU 5 训练、H100 GPU 4 视觉缓存已按用户授权完成；新任务不自动扩占 GPU。
