# 最小接口与研究约束

本仓库继承讨论结论，独立保存当前协议；相邻audit仓库的中期文档仅作背景。

## 模型A：Scene

输入：原始prompt + 单帧Tag2Text caption，无前置场景compile。

```json
{"prompt": "A person walks on a beach.", "caption": "People walking along a sandy seashore."}
```

输出单个标签：`supported` / `contradicted` / `insufficient`。不是输出三个标签列表，也不要求JSON包装。

仅判断prompt中的场景要求；不是全prompt相似度、动作或画质评分。VBench论文 arXiv:2311.17982v1 第18–19页用ocean vs river说明场景区分，并按成功帧比例评分。候选最小聚合为supported记1，其他记0，另报不足率；正式验收前确定。

## 模型B：三个LoRA

外部指定任务；各任务样本输入仅prompt。词表属于冻结配置与标注约束，不要求每条输入额外携带视觉预测。模型无末端match。

Spatial：

```json
{"relationships": [{"subject": "cat", "relation": "left", "object": "dog"}, {"subject": "dog", "relation": "below", "object": "bird"}]}
```

Action：

```json
{"actions": ["playing guitar"]}
```

Objects：

```json
{"entities": ["cat", "dog", "bird"]}
```

## 不变约束

- 对象名对齐实际GRiT标签，动作名对齐锁定K400，未知不可强配。
- Spatial保留有序关系，当前支持 left/right/above/below，多关系保持逐关系均值；复杂身份/属性无法表达时不能宣称完整满足。
- Action前端修复不自动修复Top-5评分；采用目标概率等方案需单独消融。
- Objects检查目标是否同帧共同出现，不统计视频所有物体；目标不限两个，不按and简单切分。实例数量count不是当前任务必需字段。
- 超过两个对象是我们的扩展，不能声称当前Official适配原样支持。
- 视觉漏检、UMT误分类、Tag2Text幻觉不能由文本接口保证修复。
- 模型只做语义接口，计数、检测去重、几何及最终聚合由代码完成。

## 待定的协议边界

未知动作、歧义、无场景要求、不支持关系、空输出、同类多实例指代、重复实体与动作选择规则均需P1标注指南明确后才生成正式数据。当前CLI仅展示正常样本，不构成完整schema验证器。

## Spatial 确定性后端 v2

`scripts/score_matrix.py` 默认 `--spatial-backend repair-v2`。Origin 固定回放官方公式；Repair-rule 和 Repair-model 共用 `spatial_repair.py`：先去实体名前 a/an/the 并映射既有公共词表，再按不同检测实例的主体/客体有序配对，按方向符号过滤，保留官方主轴与 IoU=0.1 权重。未知名称不模糊强配，坏框计缺失，不修改检测器输出或原始模型 JSON。

多实例使用有效有序对最大值；同帧可能同时支持相反关系，不保证这种歧义情况单独镜像一定掉分。16 帧固定分母、缺失记零。镜像几何与同步交换方向的数学等变性、原始正确/错误方向的条件变化、真实视频重跑检测的误差分别报告，见 [v2 结果](deterministic/spatial-repair-v2/README.md)。`legacy-official` 保留首轮纯接口实验的几何及实体序列化。

## Action 同义接口 v2.1

正式推理入口与 `score_matrix.py` 默认 `--action-interface repair-v2.1`。固定 adapter 输出后使用 `action_repair.py`：首先保留并归一化模型短语，再在声明的单句主语/动作结构内匹配 K400 原名或既有同义词。跨句、契约外主语形式和不支持的连词结构使用模型输出，不把风格描述中的 and/without 当作动作；没有明确已知正动作时保留模型空结果。原生 JSON、接口前后目标、scope_guard 原因都保留；不读视觉预测、变换类型、参考答案或原句分数。词典内容与 matrix-v1 原先的 60 对完全相同。

完整类别名优先于并列拆分，多动作去重；已知＋未知保留已知项与 other，other 计零并单报已知类得分。歧义、否定、观看/意图等受限语境不凭关键词强配；非法/缺失模型输出不能由规则掩盖为成功。词典未完全覆盖时保留模型输出路径并标记未覆盖，不承诺开放表达全正确。

`Repair-model` 在此版本指 **v9＋确定性接口**；`repair-v2` 精确保留修补前语义，`legacy-model` 保留未接接口的历史路径。原封闭词典结果见 [Action v2](deterministic/action-repair-v2/README.md)，新消融与 v2.1 后验回归见 [ablation-v1](deterministic/ablation-v1/README.md)。`evaluate_smoke.py` 默认仍为 legacy-model，以免把接口收益误记为原生模型能力。UMT top-5、0.85 阈值、四位小数舍入及最终平均不变。

## Objects 相邻帧确认 v2

`score_matrix.py` 默认 `--objects-backend repair-v2`，Repair 两方案共用 `objects_repair.video_scores(entities, frame_detections)`。固定 v6 解析和 GRiT 缓存不变。要求当前帧的每个实体都存在，且各自至少一个框与前一帧或后一帧的同名框 IoU≥0.5；边界不环绕，不跨缺帧、不填补漏检。全部实体仍须在当前帧共同出现；重复实体去重，不额外要求实例数量。

评分函数不接收遮挡参数、可见性标签或原视频配对信息。每帧为 supported / absent / unconfirmed / missing 之一；仅 supported 记 1，其余计 0，分母固定 16。absent 仅表示原检测没有满足合取，并非人工证实对象不存在；unconfirmed 表示检测存在但未获相邻帧支持，单列弃权；非法框/缺帧单列缺失。

不可见端点的“无正分”可包含 unconfirmed；“全部明确拒绝”不能包含 unconfirmed 或 missing。验证敏感性还要求两端无弃权，避免仅靠失去原始正分或确认不足通过验收。原始正分保留、可见帧检出、同批背景对照和持续误报单列，见 [Objects v2](deterministic/objects-repair-v2/README.md)。此项改变 Repair 的视觉证据确认规则，不是 LoRA 能力提升；`legacy-official` 保留旧单帧标签合取，Origin 始终使用旧行为。
