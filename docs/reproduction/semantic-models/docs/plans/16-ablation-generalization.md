# 四维消融与泛化评估 v1

本轮用户授权消融、抗过拟合指标和小模型架构比较。沿用已确认的 rtx4090 GPU 5；训练前已检查空余 46,392 MiB。只用现有本地权重与数据，不调用付费标注。以下方案在本轮新测试推理前冻结；完成情况另见结果报告。

## 固定实验

- 以 Qwen3-0.6B 对照当前 Qwen3-8B。0.6B 重新训练四个独立 adapter，逐任务复用当前 8B 的 train/dev 文件、seed、LoRA r=16/alpha=32、学习率与固定 600/300/300/900 步；只评 final。旧 0.6B 数据不同，不参与同条件比较。
- 每种规模比较 base（关闭 LoRA）与 SFT，以及原始 JSON 与既有 Repair 接口；不根据本轮失败扩充词典。Scene 另做去 caption 和固定错配 caption 消融，保留原始金标签，因此该对照测量信息移除的代价。
- 底座都是 dense decoder Transformer；三解析维度共享冻结底座、硬路由独立 LoRA；Scene 独立。规模对照不等于不同模型家族、联合 adapter 或分类头对照。实际参数量、训练耗时、推理资源单列。
- 已有四维后端消融按旧固定样本复算/汇总；明确标为见过的回归集，不作为本轮未见测试。

## 新评估与来源隔离

新建、冻结工程契约挑战集：每维 24 个语义家族，包含 canonical、未加入词典的改写、改变目标的反例、组合要求及 201–400 词的 head/middle/tail 位置压力测试。不同变体归同一 family。长文本只增加拍摄风格要求，不增加物体、动作或场景条件。Scene 另外平衡 supported/contradicted/insufficient。

对全部四任务当前 train/dev 检查规范化文本重合；工程 canonical 的自然重合允许作为 ID 参照，改写与长文本单列。记录来源和模板重复，工程构造不称独立人工金标准。补充固定来源的自然语言诊断（SNLI 官方 test 的场景相关子集，按 premise 族、类别平衡选择），不能等同真实 Tag2Text 分布，也不能排除基模预训练见过公开语料。已有 dev 只作训练分布诊断，不用于选择 checkpoint 或方法。

## 预先固定指标

主指标 **WG-CC（worst-group correct-and-consistent）**：每个维度×变换组，只有原始与变换端点都符合各自语义目标才计成功。组内先按语义 family 计一次，再取所有预声明组的最低成功率。缺失、解析失败、超预算、对确定目标输出 other/空答案均计失败。语义改变的控制必须输出新目标，不能复制原输出。

同时报告 WG-CC 的 **同时 95% Wilson 下界**：每个组用 Bonferroni 分配 alpha，再取各组下界最小值。它以 family 独立为近似，不能消除共享人工模板相关性；因此称本挑战集保守诊断，不称部署保证。另报普通 exact accuracy、有效输出覆盖率、最差组、每组样本数、长减短的 family 配对 bootstrap CI（2,000 次，seed=20260919）。长文本非劣效参照固定为 −0.03；不通过即报告不通过，不根据结果改阈值。

必备反作弊对照：恒定空/other、复制 canonical 输出、Oracle。Oracle 应为 1；常量和复制输出应在语义改变组失败。Scene insufficient 是合法确定标签，不把它一概算弃权。工程集与自然集、训练分布与 OOD、raw 与接口结果分表，禁止用纯一致性、低分或选择性删样本代替正确性。

上述指标只能暴露过拟合/捷径，不能在逻辑上防止所有过拟合。强泛化主张还需要新的盲测人工金标、多训练 seed 与独立数据源；本轮单 seed 规模消融不能代替它们。

## 冻结与复现

保存数据、配置、代码、adapter SHA；运行推理前写不可覆盖的 manifest，所有结果保留全分母。模型与原始数据/预测在忽略目录，Git 仅收代码、配置、聚合表和方法说明。执行纯 CPU 单测、完整 pytest、CLI help、uv lock --check 与 diff 检查。依照 CheckList 的行为测试思路与 worst-group 诊断组织结果，不据此声称实施了 Group DRO 训练。

方法来源：[CheckList](https://aclanthology.org/2020.acl-main.442/)、[Group DRO 的最差组泛化分析](https://arxiv.org/abs/1911.08731)、[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)。
