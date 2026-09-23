# P2：复用LoRA训练框架

此阶段尚未实施；当前CLI只是schema示例。

## 复用与模型

使用官方TRL SFTTrainer + PEFT，见 [固定来源](../upstream.md)。不从零写训练循环，不vendoring完整框架。

- 三个解析adapter分别从完全相同、SHA固定的冻结backbone初始化，不把一个任务训练后的adapter继续当另一个任务起点。
- 推理共享底座，按任务选择一个adapter；不默认merge三个adapter。
- Scene独立训练：候选为三标签SFT或sequence-classification head，正式训练前确定；不由当前依赖选择暗中锁死架构。
- 模型、tokenizer、chat template均记录revision和哈希；正式下载权重另行执行。

## 起始配置建议

8B、BF16 LoRA、rank16/alpha32、all-linear、SDPA、gradient checkpointing、microbatch1、长度512–1024、gradient accumulation、packing=False。这不是已验证最优参数。

先使用常规LoRA，暂不引入bitsandbytes、FlashAttention自编译、DeepSpeed。显存不足时再审阅并锁定QLoRA依赖。

## 数据与loss

JSON任务仅监督completion；TRL completion_only_loss=True。显式使用一致的no-thinking模板，不依赖默认行为。不假设模板自带assistant mask。不要截断掉目标JSON；超长样本单独统计和处理。

## 运行门禁

1. 固定数据manifest、split、模型SHA及参数。
2. 检查空闲GPU并明确物理卡；每进程仅看一张卡，逻辑cuda:0。
3. 100步试跑，记录峰值显存、实际token吞吐、batch、长度、loss、墙钟；不以参数量猜测时长。
4. 三个adapter可在不同空闲卡独立训练；多卡显存不会自动合并。
5. 实际GPU算子与反传通过不代表完整模型训练已通过。

## 输出

checkpoint/adapters、tokenizer引用、config、数据/代码/锁文件哈希、seed、硬件、训练日志、dev指标、最佳checkpoint选择依据。测试集不用于早停或超参选择。
