# Object Class：后端与验证进展

独立 uv 包，使用 audit-core 的 CLI、输入、分片、输出、provenance；不导入其他 metric。
`--vbench` 直调锁定上游 `compute_object_class`，不改原目标串。
`--audit` 默认 repair：保留 GRiT 全实例、box 与 confidence，确定性别名判定，全采样帧分母。
缺配置、模型失败或 query 不支持均保留空分数与状态；部分覆盖时主 aggregate 为空，子集均值只作诊断。

2026-09-20：wheel/sdist 构建与 help 通过。H100 上 63 个 metadata 反事实请求两后端均完成；
真实 GRiT trace 按原串重放与 Official 63/63 一致，最大误差 0。
14 个 test bases 的大小写对照中，Official 中位绝对变化 1.0，Repair 为 0。
这不是自然同义词漂移率、人工偏好改善或冻结 E0 复现。

共享 Qwen3-8B 的独立 Object LoRA 已训练 300 steps；四级原始 query 视频消融已实测，
LoRA 与确定性修复的 14 个 test 原始视频平均分均为 0.991071，没有证据说明 LoRA 在该小队列增益。
真实四卡 CLI 4 视频 × 2 后端，8/8 对齐，最大误差 0；每进程只见一张卡、逻辑 cuda:0。
最新全量 676 passed / 3 skipped，三个 skip 均属其他维度真实模型条件；单独 wheel 在仓库外安装、
help 和无配置失败输出通过。人工审核为 0，语义 silver 只报参考一致性，不报标注精度。
配置、命令、覆盖率与未验证范围见 [两维执行记录](../../docs/object-color-repair.md)。
