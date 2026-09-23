# 主计划：四维语义适配器

## 已决定

训练两套模型系统：Scene 独立；Spatial/Action/Objects 共用底座、分别训练 LoRA、按维度硬路由。四维正式 adapter 已完成；原始/变换视频的三方案确定性矩阵已完成。

## 子计划与门禁

| 阶段 | 子计划 | 交付与进入下一阶段的条件 |
| --- | --- | --- |
| P0 | [环境](01-environment.md) | 私有仓库、Python/uv/包锁、本地测试、远端同步；不启动训练 |
| P1 | [数据](02-data.md) | 原料数量与许可证核验、冻结划分、人工金标准、schema/词表校验 |
| P2 | [训练](03-training.md) | 确定基模SHA、指定空闲GPU、100步测量；之后才执行正式LoRA训练 |
| P3 | [评测](04-evaluation.md) | 未微调/规则对照、独立测试、错误分析、速度与覆盖率，才考虑接入audit |

P0–P2 已完成环境、数据加工与正式训练。P3 的四维三方案矩阵已验收：全部冻结项均已尝试，主表、家族 CI、覆盖/弃权、失败帧、来源与资源汇总齐全。新增工程长提示词与小模型消融已完成，独立四维人工金标与自然长提示词结论仍不具备，不以模型标注冒充。详见 [verification](../verification.md)与[新增消融](../deterministic/ablation-v1/README.md)；[矩阵执行状态](../EXECUTION-2026-09-20.md)保留历史快照。

## 当前执行记录

已完成[四维消融与泛化评估](16-ablation-generalization.md)：四个新 0.6B final、同数据 0.6B/8B 的 4,584 次预测、base/SFT/接口/Scene caption 消融与最差组双端正确率 WG-CC。完整[结果](../deterministic/ablation-v1/README.md)、[架构及指标](../deterministic/ablation-v1/method.md)保留新测试失败和已有 Objects 正例损失。Action 范围保护 v2.1 已接入默认推理/评分；原冻结消融可逐字重现，后验修复不冒充新盲测。226 passed。

四维当前版本已统一整理为[最新结果汇总](../deterministic/current-summary/README.md)，含原始均分、关键变换、正确性/弃权/检出代价、完整 CSV 和来源 SHA；本次只整理已有结果，没有新实验。Spatial/Action/Objects 使用 v2，Scene 与 Origin 保持原版本。

Objects 后续相邻帧确认已实现并复算，见[固定方案与边界](15-objects-temporal-repair.md)和[v2 结果](../deterministic/objects-repair-v2/README.md)。275 个不可见端点中的正分残留 29→15，原始均分下降 0.02258；弃权与已确认阴性分开，持续误报和可见目标损失均保留。四维其他 63,500 条记录逐字段不变。

Action 同义接口已按用户要求完成，见[实施计划](14-action-synonym-repair.md)和[v2 结果](../deterministic/action-repair-v2/README.md)。既有 60 对词典内容保持不变，接入生产推理和评分；固定 v9 加该接口在全部 1,200 视频上同义分数完全一致。原生模型与接口收益分开报告。

Spatial 方向后端已按用户后续要求完成，见[实施与验收计划](13-spatial-direction-repair.md)和[v2 结果](../deterministic/spatial-repair-v2/README.md)。此版本显式改变 Repair 几何公式，保留 Origin 与 matrix-v1 历史；已在同一缓存上完成四维回归。

本轮已按用户授权完成四维三方案矩阵。交付见 [最终研究报告](../deterministic-experiments-report.md)、[CSV/Markdown/LaTeX 表格](../deterministic/matrix-v1/README.md)、[执行验收](../EXECUTION-2026-09-20.md)；过程与问题见 [评测审计](../evaluation-audit.md)。此项覆盖下文历史阶段的待授权描述。

## 子计划与历史实施记录

[冒烟实施主计划](05-smoke-master.md)索引实施子计划与跨阶段约束：

1. [S1 数据加工与清洗](06-smoke-data.md)
2. [S2 训练代码实现](07-smoke-training-code.md)
3. [S3 分层冒烟与验收](08-smoke-validation.md)
4. [S4 提示词长度泛化](09-length-generalization.md)：四任务必需，跨数据、训练与验收落实
5. [S1.5 teacher 数据构造](10-teacher-data.md)：受控试标与候选入库，预算已用满 20/20
6. [确定性（蜕变）实验](11-deterministic-experiments.md)：四维镜像/反转/遮挡/替换族，Official vs Repair 配对对照
7. [下一阶段行动指南](12-next-stage-action-guide.md)：先修评测口径与来源分组，再做四维原始数据双后端评分、确定性实验与盲测
8. [阶段性总结](../STATUS-2026-09-20.md)：训练/数据/效果、确定性数据、Origin-Repair 分数与阻塞（历史阶段快照；本轮数字以最终报告为准）
9. [Spatial 有向后端修复 v2](13-spatial-direction-repair.md)：有符号几何、正确实体配对、文本序列化及四格镜像验收
10. [Action 同义接口修复 v2](14-action-synonym-repair.md)：冻结模型输出重放、统一归一化、OOV/类别替换控制及全矩阵回归
11. [Objects 相邻帧确认 v2](15-objects-temporal-repair.md)：孤立检测抑制、弃权语义、可见帧损失与完整对照，未解决持续视觉幻觉
12. [消融与泛化 v1](16-ablation-generalization.md)：同数据 0.6B/8B、LoRA/接口/证据消融、未见表达与长度挑战、抗捷径指标

实施状态：清洗、teacher 试标、训练薄封装、冒烟（T0–T3）与**正式 8B 四维训练**均已完成，见[正式训练报告](../formal-training-report.md)、[模型卡](../model-cards.md)、[试标报告](../teacher-pilot-report.md)、[冒烟报告](../smoke-report.md)、[运行手册](../runbook.md)、[决策记录](../decisions.md)。真实 Tag2Text 数据与固定 final 重训已完成。尚不具备人工金标准与独立长提示词泛化结论；本轮确定性实验不等于视频质量收益验证。工程样本、teacher候选与独立研究金标准三者分离。

## 已决策（详见[决策记录](../decisions.md)）

- 正式底座：`Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`，沿用锁定栈；0.6B 除历史冒烟外，新增同数据规模消融，未据新测试替换正式底座。
- Scene：保持生成式三标签 SFT＋LoRA，推理端固定标签边界约束，不改分类头。
- 当前 Scene/Objects 标注已按本轮授权使用独立用途账本；早期 pilot 的 20/20 限额保留为历史，不限制后续已授权用途。
- 本轮 GPU 已确认：rtx4090 GPU 5 训练、H100 GPU 4 视觉缓存。四模型使用固定步数 final，不按 dev/test 选 checkpoint。

## 后续研究边界（不阻塞本轮已授权交付）

- 新一轮训练、4,096 长度评估与 8B 产物公开发布策略；不自动扩展本轮预算。
- 独立金标准规模与来源；论文披露口径。
- 未见同义表达与更丰富 OOV 的独立研究；当前 K400 别名、空结果与 other 协议已冻结，不按测试错例扩充。
- 多关系聚合、Action top-5 的替代评分、超过两个对象的聚合。

## 工作边界

本轮用户已授权完成四维矩阵、逐步 commit/push，并在两服务器 pull --ff-only。相邻 vbench-audit 的冻结 data/results/splits/runs 只读；原始数据、图片、模型和预测不进 Git。早期子计划中的待授权文字是当时的记录，不覆盖本轮明确授权。
