# S3：分层冒烟与验收子计划

状态：T0/T1/T2/T3 已执行，结果与未完成项见[冒烟报告](../smoke-report.md)。正式训练、独立金标准与长度分桶验收仍未执行。

## T0：数据/CLI纯CPU

- 固定Python3.11.14、uv0.9.17，uv lock --check和uv sync --locked。
- 运行旧测试及新增schema/分组/去重/数据生成测试；所有入口--help不加载模型。
- 同一输入清洗两次，样本hash与分组一致；输出build目录独立。
- 验证raw未改、许可隔离、无未知target类别、计数闭合、样本目标由输入支持。
- 确认MovieGen仍为unlabeled，SNLI仍为auxiliary_nli，不被训练loader误读。

## T1：随机微型模型CPU闭环

在已有train extra环境，构建本地随机tiny causal LM/tokenizer，无下载。

1. 四任务各2–8条明确工程fixture，运行forward/backward和至少一个optimizer step。
2. 检查loss有限、有adapter梯度、预期adapter参数发生变化、底座不变。
3. checkpoint保存再加载；eval模式相同输入logits在声明容差内一致；不要求CPU微型模型有语义能力。
4. 模型B加载三个独立命名adapter，逐一切换；输出归属和加载参数验证，不要求每个随机adapter必然生成不同文本。
5. 检查Scene独立运行目录和模型对象，单标签解析与三类固定映射。
6. 检查resume兼容、异常输入、缺adapter、不合法输出和超长样本失败路径。

验收：证明训练/保存/路由实现正确，不宣称任务准确率。

## T2：真实底座单卡100步（前置确认后）

### 前置

- 明确两个系统基模及不可变revision、tokenizer/template版本；准备合法的本地权重。
- 用户确认具体GPU使用；执行前nvidia-smi核验空闲显存，不依赖初始化时的快照，不kill任何已有进程。
- 只使用已放行engineering/weak样本；不以许可待定Flickr或无标签MovieGen凑数。
- 物理卡通过CUDA_VISIBLE_DEVICES隔离，进程只使用cuda:0。一次一个任务，禁止后台自动开四卡训练。

### 运行矩阵

| 任务 | 建议工程规模 | 步数 | 主要检查 |
| --- | ---: | ---: | --- |
| Spatial | 最多400单关系＋约50人工核对多关系 | 100 | 多三元组输出、方向、adapter持久化 |
| Action | 400类标准句候选，经语法/有效性过滤后如实计数 | 100 | K400合法性、ID映射与coverage |
| Objects | 约100手写小样本；或许可/映射已放行子集 | 100 | 1/2/3+对象、非and枚举、集合输出 |
| Scene | 约90手写场景证据对，覆盖三类 | 100 | 单标签、支持/矛盾/不足不混淆 |

这不是正式训练配额。数据不足时使用小fixture验证工程，不复制样本冒充规模。

### 过拟合诊断

另取每任务8–16条无歧义fixture做可选短程记忆测试，固定最大步数上限（例如200）；用于定位mask/模板/优化问题。训练集指标和独立验证严格分栏。未达拟合不自动等同架构无效，先分析loss、mask和样本歧义；不无限延长训练。

### 记录

代码SHA及dirty diff hash、uv.lock hash、数据build/split hash、底座/tokenizer SHA、adapter配置、seed、设备UUID/物理卡、driver/CUDA、batch/累积/长度、总tokens、稳态tokens/s、100步墙钟、加载时间、峰值allocated/reserved显存、loss/梯度、checkpoint路径与恢复结果。

## T3：接口推理与初步验证

- 同一评估fixture比较未微调和微调结果；冻结生成参数。
- Spatial三元组集合F1与全句正确率；Action类别映射；Objects集合P/R/F1；Scene macro-F1/混淆矩阵。
- 同时报告JSON/标签有效率、失败数和总分母；不得丢掉失败样本。
- 不给100步冒烟设任意“95%即模型很好”的门槛；主要验收loss/梯度/保存恢复/路由正确，语义指标用于定位问题。
- 只有独立人工自然测试才能支持有效性声明；本轮smoke不替代最终视频偏好评测。

## 输出与停止条件

产物：ignored runs/smoke/<run-id>/下配置、日志、预测、指标、adapter与manifest；docs/smoke-report.md保存摘要与复验命令。

出现数据泄漏、越权联网下载、目标被截断、底座意外更新、adapter串用、NaN、OOM、保存重载不一致时暂停该任务并记录；修复后重跑相关层。正式长训练必须在独立数据与标注验收后另行启动。
