# Object Class / Color：确定性修复与独立 LoRA 实测

日期：2026-09-20。必须连同[方法审查与非声明](object_color_repair_20260920.review.md)引用。
实现与命令见[复现说明](../object-color-repair.md)。本报告只覆盖用户指定的两个主族及其控制，未评测自然人工偏好。

Object 的 metadata 大小写/类别别名干预暴露 Official 的字面匹配脆弱性，确定性修复保持不变。
Color 在 5 个自动合格 test 基底上达到预注册可见性响应门槛；但 25 个候选只接收 7 个，且 Official 的同义改写控制全无返回值。不能据此称两维所有科学验收门槛都通过，或无保留地称 Repair 更好。

## 冻结设计、构造与覆盖

先按规范化原 prompt 的 SHA-256 分组，每维 hash 顺序前 20 test、接着 5 dev、其余 train，再扩写。别名、候选、门槛在评分前冻结；所有拒收、零效应及空分保留，不补选。
Object 只改 `dimension_metadata.object_class.object`，视频字节与原 prompt 相同；不让 prompt-only 模型读取改写目标。

| 队列 | 候选 dev/test | 接收 dev/test | 拒收 | 评测请求 |
| --- | --- | --- | ---: | ---: |
| Object | 5/20 | 2/14 | 9 | 63（同一基底原串/大写/可用首别名/不存在类代理） |
| Color | 5/20 | 2/5 | 18 | 42（每 base 五档可见性 + 同义改写控制） |

Object 资格由独立 Mask R-CNN、阈值 0.7、目标在 ≥12/16 帧定位决定；9 条均未达该门槛。
Color 在同一候选池要求原目标 16/16 帧定位，10 条未达标；另 8 条全遮挡后独立定位器仍能找到目标，施工拒收。此资格不是 GRiT 打分筛选，也不是人工确认。

Color 施工优先 SegFormer-B0 精确类别、限制在独立目标框内，区域不足时 MobileSAM box+中心前景点；覆盖全部目标实例，以掩码外 RGB 中位数填充，1.5 px 只向内羽化。固定帧排列产生 100/75/50/25/0% 嵌套可见性。所有接受版本断言掩码外像素变化为 0、无损 decode 等于施工数组、第二次编码 SHA-256 一致。掩码只用于施工，评分仅看视频和 query。自动全遮挡检查不证明语义掩码完整或人工颜色正确。

| 后端/变体 | Object 有分/保留请求 | Color 有分/保留请求 | Color 空分 |
| --- | ---: | ---: | ---: |
| 直接 Official | 63/63 | 33/42（78.57%） | 9 dropped_by_official |
| binding-only trace 消融 | — | 33/42 | 9 条件分母为空 |
| binding+lexical trace 消融 | — | 40/42 | 2 条件分母为空 |
| full Repair | 63/63 | 42/42（100%） | 0 |

所有上述模型请求运行失败和文本 unsupported 均为 0；不能从这一队列推断总体失败率为 0。
Repair 的 1008/672 个采样帧均有真实 GRiT 返回。Object 记录 2842 实例：255 exact-hit 帧、492 alias-hit 帧、261 other-class 帧；这里重复计入构造 queries，**不是自然词法漂移发生率**。Color 记录 2259 实例：351 支持、207 颜色证据不足、114 无目标帧；后两项都保留在全帧分母。实测两路实例均为 verified-index 关联，IoU/歧义分支由合约反例覆盖，未声称自然数据中验证了所有歧义情形。

## 主命题与控制

统计单位为 base，固定 seed=20260920，10,000 次 base bootstrap。变化侧为中位 |Δ|≥0.20、绝对 CI 下界>0.10；不变侧为中位 |Δ|≤0.05、带符号 CI 含 0。Color Repair 使用有方向的 100%−0% 降分并要求中位 base Spearman≥0.8 或严格单调 base≥70%。这些是预注册 operational gates；CI 含 0 不等于严格等效性证明。

| Test 命题 | n | Official | Repair | 可支持的结论 |
| --- | ---: | --- | --- | --- |
| Object 原目标→大写（主对照） | 14 | 中位绝对 Δ=1.0，95% CI [1,1] | 0，[0,0] | 变化侧/不变侧分别达标 |
| Object 原目标→首个冻结别名 | 13 | 1.0，[1,1] | 0，[0,0] | 达标；主要为复数类名，其中一个 test 类无别名，不是评分失败 |
| Object 不在场类别代理 | 14 | 中位分=0 | 中位分=0 | 双方低分；独立检测未检出不等于人工证实不在场 |
| Color 五档可见性 | 5 | 在有定义的档位上，最大绝对偏离的中位数=0，CI [0,0.6] | 100%→0% 中位降分=1.0，CI [0.625,1.0] | Official 的条件不变性门槛通过但区间宽；Repair 响应门槛通过 |
| Color 同义改写控制 | 5 eligible | 0 个有定义配对，全部丢弃 | 5/5：Δ=0，CI [0,0] | Repair 不变；**双方都平的控制未获验证** |

Color Repair 五个 test bases 均五档严格单调，中位 Spearman≈1。两个 dev bases 的中位降分为 0.5、CI [0,1]，仅一个严格单调，**dev 门槛未通过**；未因此调参或删除零效应。Object 两个 dev bases 的方向与 test 一致。完整 dev/test 结果见 [main.csv](object_color_repair_20260920/main.csv)，逐 query/base 分数、状态见 [per_base.csv](object_color_repair_20260920/per_base.csv)。

Color binding-only 在 test 的条件偏离中位数为 0，CI [0,0.4]；binding+lexical 为 0.125，绝对 CI [0,1]，不满足不变性门槛。它们由同一真实 GRiT trace 经过各自 metric 公式重放，**不是另外两次 GPU Official 测量**。原始 tuple 消费重放与 true Official 42/42 空值一致、33 个有效值误差为 0；Object 原串重放 63/63 误差为 0。

验收时修正了这两个消融的 summary 分母名称（条件率曾误标成全帧率）；实际逐帧分母与所有分数未改，原评分文件 hash 保持一致，修正记录保留在 execution manifest。真实 GPU parity 之后的这项摘要标签修复和独立安装 provenance 回退经本机合约测试验证，未重新运行 GPU 数值试验。

源码行号已于 2026-09-20 对 fd18b3d 复核：Object `vbench/object_class.py:32` 丢弃实例分数并 set，`:33` 吞异常，`:40` 精确成员；Color `vbench/color.py:37` 绑定首实例，`:46` 精确对象，`:47`–`:50` 白名单/子串，`:67` replace 解析，`:81`–`:89` 条件聚合/丢视频。Repair 保留模型证据、核验关联、整词颜色、固定全帧分母；公式属于 metric，不归因于 LLM。

## 冻结词表、训练与四级消融

官方 79 Object + 85 Color 种子先分组；Color 的对象标签来自可无歧义解析的官方模板，颜色来自 aux。固定 24 个训练源真实 GRiT 观测共 384 帧，观测 56 类均为标准 COCO 类名，仅 mouse 是官方 79 suite 外新增，冻结对象词表为 80 类。未观测到不等于模型不支持。别名在测分前定义为确定性表，未经人工确认的自然 GRiT 同义漂移不作频率声明；颜色 20 项，保留 crimson/navy/maroon。

DeepSeek 以 `deepseek-flash`、fingerprint `aeb56401ca74e127821c4f9126dcb669`、UTC `2026-09-20T18:56:59.458537+00:00` 固定本次服务身份；它是 API 身份，**不是不可变权重 revision**。164 gold + 1280 silver，160 个扩写请求成功、4 个 `IncompleteRead` 网络失败隔离；并非已经观察到版本漂移。每源八类覆盖同义、属性、复数、否定、两物体两色、场景颜色、上下位非同义等。200 条人工审核位已留，审核完成 0。跨源重复 prompt 不进入 train；训练源与 dev/test 源不重叠。

Qwen3-8B 同一主干，两个独立 LoRA 各 r=16、alpha=32、dropout=0.05、all-linear、bf16、lr=1e-4、batch=1、accum=4、固定 300 steps、seed=20260920。Object/Color 实际训练样本为 424/465；adapter 参数不共享。推理一次加载两 adapter，分别启用指定 head；base 模式明确禁用 adapter。

以下是**未人工审核参考标签的完整 JSON 一致性**，不是标注精度、真实语义准确率或视频生成质量：

| Test 参考分层 | 参考样本/源组 | 确定性规则 | 同基模未微调 | 独立 LoRA |
| --- | ---: | ---: | ---: | ---: |
| Object 官方 seed | 20/20 | 20/20 | 12/20 | 20/20 |
| Object silver | 160/20 | 136/160（85.00%） | 39/160（24.38%，含 21 无效输出） | 151/160（94.38%） |
| Color 官方 seed | 20/20 | 20/20 | 20/20 | 20/20 |
| Color silver | 160/20 | 131/160（81.88%） | 80/160（50.00%） | 159/160（99.38%） |

无效输出纳入 160 的分母；合法 null 另计，不作为格式错误。按 source group bootstrap 的 CI、null/格式失败数及所有逐样本预测见 [prompt_compilation.csv](object_color_repair_20260920/prompt_compilation.csv) 与 [prompt_compilation_details.json](object_color_repair_20260920/prompt_compilation_details.json)。gold 与 silver 可能共享文本，按层报告，不把合计当独立样本。

160 条 test silver 分别有 157 个 Object / 153 个 Color 不同原始字符串；重复记录按源组保留，表同时报 unique prompt 数。
数据审计发现 `a basin` 在两个 Object 源中被 teacher 标为 bowl/sink，`a sandwich` 的 train silver null 与 test 官方 seed sandwich 冲突。
这两处冲突及参考标签均原样保留，相关 train 记录已被评分前的跨源重复规则排除；训练与 heldout 的 source group 和规范化 prompt 重叠均为 0。
审计见 [integrity_audit.json](object_color_repair_20260920/integrity_audit.json)；这种标签局限也是不报人工语义精度的原因。

视频端四级消融只用原始 query，各级都真正运行 GRiT。表内是固定小队列的平均支持分，**不是准确率，越高不自动越好**：

| Test 原始视频 | n | 旧规则 true Official | 确定性归一/别名 + Repair | 同基模未微调 + Repair | LoRA + Repair |
| --- | ---: | ---: | ---: | ---: | ---: |
| Object | 14 | 0.991071 | 0.991071 | 0.633929 | 0.991071 |
| Color | 5 | 1.000000 | 0.975000 | 0.975000 | 0.975000 |

本队列看不到 LoRA 相对确定性规则的视频分数增量。Object metadata-only 主命题不归功于 LoRA。旧规则→确定性 Repair 包含明确的公式变更；后三列固定视觉公式，才隔离 prompt 编译方式。详见 [video_ablation.csv](object_color_repair_20260920/video_ablation.csv) 与 [逐 base 四级表](object_color_repair_20260920/video_ablation_per_base.csv)。

## 工程、成本与产物

实现验收时的完整工作区测试 **676 passed, 3 skipped**（包含其他并行任务改动）；两 metric 单独安装、各包构建、12 help、uv lock/sync 验收通过。Object/Color 独立提交另对暂存快照验证，结果记录在提交说明。真实 H100 CLI 四卡各 4 视频 × 双后端，最大绝对误差 0；每 worker 只见一张物理卡且用逻辑 cuda:0。本机没有 CUDA 实测。
测试摘要、skip 原因、独立安装位置、构建产物和日志 hash 见 [verification.json](object_color_repair_20260920/verification.json)。

| 测量组 | 硬件 | 请求数 | 实测墙钟 |
| --- | --- | ---: | ---: |
| Object Official / Repair 全族 | H100 物理 5，单卡 | 各 63 | 80.385 / 81.893 s |
| Color Official / Repair 全族 | H100 物理 5，单卡 | 各 42 | 77.277 / 206.333 s |
| Object 四卡公共 CLI | H100 1/3/4/5 | 4 视频、两后端 | 34.358 s |
| Color 四卡公共 CLI | H100 1/3/4/5 | 4 视频、两后端 | 35.367 s |
| Object / Color LoRA 300 steps | RTX4090 物理 5，分别训练 | 424 / 465 文本训练样本 | trainer 393.868 / 388.995 s |

所有视频媒体时长均为 2 s；105 请求对应 51 个实际文件。四卡时间只覆盖表列 parity 子集，不能当作完整族的四卡墙钟；trainer 时间不包含加载、silver 与编译，视频消融时间不含预先语义编译。逐视频尺寸/时长/hash 在 [media.csv](object_color_repair_20260920/media.csv)，全部计时在 [timing.csv](object_color_repair_20260920/timing.csv)，覆盖在 [coverage.csv](object_color_repair_20260920/coverage.csv)。

[execution_manifest.json](object_color_repair_20260920/execution_manifest.json) 保留真实 runtime、CUDA 设备 UUID、上游 SHA/源 hash、GRiT 与施工模型/基模/adapter 身份、逐 run 代码 hash 与路径。H100 含代码快照的旧 checkout HEAD 为 `9f2cf67dee2ee9849953f6c479e23fba6d814715`；它不是本次实验代码的版本标签，实际运行代码以源文件哈希为准。独立 E0 复现、人工偏好一致性、人工标注精度与掩码语义真值不在已验证范围。

可审阅配置与全部候选/拒收/query ID 在 `configs/four_dimension/object_color_family_manifest.json`，raw GPU 输出在 H100 `/root/wenbiao_zhao/vbench-audit/output/object_color_20260920/`；本地保留结果、证据和训练记录镜像，模型权重只在服务器。各表可用 `uv run --no-sync python -m scripts.object_color_delivery` 复算，不单独引用 pooled/tie-margin CPA。
