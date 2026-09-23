# S2：训练代码实现子计划

状态：已实现并通过 T1/T2 冒烟（见[冒烟报告](../smoke-report.md)）。复用已锁定TRL 0.19.1、PEFT 0.15.2与Transformers 4.52.4，参考docs/upstream.md固定入口，未改依赖锁、未自写优化器训练循环。

## 1. 拟新增模块

| 文件 | 职责 |
| --- | --- |
| data.py | JSONL校验、任务输入渲染、规范target序列化 |
| training.py | 解析配置、加载固定模型/tokenizer、构造LoRA与SFTTrainer |
| inference.py | 单底座加载三个命名adapter，硬路由，严格输出解析 |
| scripts/train_adapter.py | spatial/action/objects单任务训练入口 |
| scripts/train_scene.py | 独立Scene训练入口，不与解析adapter共用训练状态 |
| scripts/evaluate_smoke.py | 分任务预测、格式/语义指标、失败覆盖率 |
| configs/smoke/*.json | 四任务小规模配置，权重路径必填，不写个人绝对路径 |

模块位于src/vbench_prompts_compile/；正式训练入口不能在--help时导入权重或初始化CUDA。

## 2. 默认实现路线

- 三个解析任务：同一冻结Qwen3-8B底座候选，各自重新初始化LoRA，训练产物独立。
- Scene工程首版建议使用独立实例的三标签SFT＋LoRA，以复用现有训练栈；接口保持单标签，不把它纳入三adapter解析路由。这是待执行的默认实现选择，不声称比分类头更好。
- Scene若后续改小型encoder/sequence-classification，另立对照配置，不静默改变输入/标签协议。
- CPU微型模型从本地配置随机初始化，用同一数据/collator/PEFT路径做单步；它不代表Qwen真实模板或8B资源行为。

## 3. 数据渲染与loss

- prompt-only任务加固定任务指令，输入不附视觉预测；Scene传原prompt和caption，两者边界清楚。
- 模型实际target：前三任务json.dumps规范序列化；Scene裸标签字符串。
- 固定chat template和enable_thinking=False；训练推理完全一致。
- 用prompt/completion格式和completion_only_loss=True；用单元测试逐token验证提示部分mask=-100、回答部分有监督、EOS被正确处理。
- 不只检查配置字段存在；实际构造一个batch检查labels，禁止全部labels为-100。
- 长度超限拒绝并记录，不截断丢掉JSON右括号或答案。小样本关闭packing。

## 4. 训练配置及可靠性

- 起点BF16、rank16、alpha32、all-linear、SDPA、梯度检查点、microbatch1、累积4、max_length512或1024；正式取值经GPU试跑确认。
- 显式seed、max_steps=100、logging_steps=10、save_steps=50；不自动wandb联网。
- 本地模型路径或完整revision必填；默认local_files_only和HF离线，找不到权重即报错，不能隐式下载。
- 验证实际可训练参数仅为预期adapter；底座参数训练前后校验不变。
- 每任务新输出目录，配置/代码/数据/lock/model/tokenizer identity随checkpoint保存。
- resume只接受相同task、数据hash、底座revision和adapter配置；不匹配拒绝恢复。
- OOM立即记录batch、长度、峰值显存与错误退出，不偷偷换GPU、降精度或改样本。

## 5. adapter路由与推理

- spatial/action/objects分别加载命名adapter，外部task选择；相同底座版本才能共同加载。
- set_adapter切换是模型可变状态；初版串行请求或加锁，禁止并发请求互相切adapter。
- 未知任务、缺adapter、底座revision不符显式失败；不fallback到别的任务。
- temperature/采样策略固定；工程版使用确定性解码和短max_new_tokens。
- 解析失败记录invalid_output，原文保留于ignored运行日志；不得修成空数组掩盖失败。
- JSON语义按有序三元组/集合评估，不能以JSON键顺序或列表无关顺序导致误判。
- 空结果/未知类别规则未冻结前，只用明确有答案的工程fixture；这不等于正式系统已支持开放分布。

## 6. 必需离线测试

schema正常/异常、四task渲染、mask/EOS/超长拒绝、LoRA底座冻结与梯度、adapter保存重载、路由切换与缺失拒绝、resume identity、指标分母与格式失败。

基础CPU测试不强制安装torch；训练栈测试单独标记，缺extra时显式skip，并在报告中区分已运行与跳过。所有训练栈测试必须使用本地随机微型模型，无网络无真实权重。
