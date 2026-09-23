# 训练栈复用来源

采用已发布、保守的同代版本组合，不声称是最新版本或安全审计结论。

| 包 | exact pin |
| --- | --- |
| torch | 2.7.1+cu126 |
| transformers | 4.52.4 |
| trl | 0.19.1 |
| peft | 0.15.2 |
| accelerate | 1.7.0 |
| datasets | 3.6.0 |

- 官方TRL仓库：https://github.com/huggingface/trl
- 参考release：v0.19.1
- 固定commit：accf7383a33c618a2edc7205107120b5f32e28a3
- [可复用入口 trl/scripts/sft.py](https://github.com/huggingface/trl/blob/accf7383a33c618a2edc7205107120b5f32e28a3/trl/scripts/sft.py)
- 上游许可证Apache-2.0；本轮未复制源码。若后续复制部分入口，保留许可证与归属。

通过安装包复用SFTTrainer和PEFT，仅写薄的配置、数据与revision封装。该release的examples/scripts/sft.py是迁移占位，不作为训练实现复制。

需要补充：上游示例未完整固定tokenizer/config revision，项目入口必须同时固定模型和tokenizer。completion_only_loss、no-thinking模板需用单元测试检查，不把模型默认行为当协议。

依赖release元数据：

- https://pypi.org/pypi/trl/0.19.1/json
- https://pypi.org/pypi/transformers/4.52.4/json
- https://pypi.org/pypi/peft/0.15.2/json
- https://pytorch.org/get-started/previous-versions/

版本解析通过与真实训练通过是不同结论。基础测试、远端import和CUDA结果见verification.md。后续如需升级，独立PR审阅兼容性并重建lock，不运行pip install -U破坏环境。
