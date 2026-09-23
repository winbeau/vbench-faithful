# 四维实验审计与执行记录（2026-09-20）

目标仍为 VBench 1.0 上 Spatial 镜像、Objects 遮挡、Scene 同义、Action 同义/类别替换/OOV 的完整三方案矩阵（Origin、Repair-rule、Repair-model）。本记录不是最终实验报告。

## 本轮已核验的阻碍

- 本地起点 `61503a4`。rtx4090 同一提交，但 `scripts/score_deterministic.py` 被删除；部署前保存该差异，禁止 reset。H100 原目录是脚本副本且 `.venv/bin/python` 指向不存在的本地路径；将用 Git bundle 建立独立 checkout，保留原数据和历史产物。
- 旧 test2 492 项全部失败，账本保留 1,571 次请求（DeepSeek 95、chiyi 1,476）。492 项仅 287 个 item ID：交叉配对反复使用同一 partner。不得将 492 称为独立样本/家族。
- 旧 dev/dev2/dev3 共 1,158 行，按 prompt、caption、证据视频、帧号、帧文件、证据来源 prompt 合并为 719 个观测，合并 439 个重复记录；11 个观测的视觉标签有冲突。新文本证据标签独立获取，冲突视觉真值保留为未知。
- 旧 Scene test 的 240 行只有 80 个 sample ID、200 个不同观测；v7 Scene train/dev 分别复用了其中 52/8 个提示词，均来自无场景要求池。新训练集须剔除测试 prompt 与证据两端的所有来源，不能沿用 sample_id 去重或仅查 group_id。
- 当前原始 caption 池 dev/test 各只有 11 个场景提示词；帧扩样不增加独立场景家族。跨配对还会产生共享证据依赖，不能把每条配对或每个无场景提示词都当成独立家族。

## 标注恢复协议

沿用已有供应商、模型和两遍标注加分歧仲裁，不使用学生预测修标签。

- 证据重标：DeepSeek `deepseek-flash`，纯 prompt + caption；冻结 719 个观测；本轮 purpose `scene-evidence-v8`，累计上限 2,500 次请求。
- test2 恢复：chiyi `gpt-5.6-luna`，216 同源 + 216 唯一交叉配对 + 60 既有 gate 确认的无场景提示词；新 purpose `scene-test2-recovery-v8`，上限 1,600 次。历史失败请求不归零，保留在原账本。
- 每次发送前加锁记账，每次响应和每条结果立即持久化；恢复复用已收到的响应；401/403 首次出现即停止整批，避免对 492 项重复扣请求。
- 输出中的缺失保持在冻结清单分母内；返回码 2 表示未完成（包括有意分段运行）。JSONL 与报告区分 planned / completed / missing。
- 记录供应商返回的 token 数和请求数。供应商没有返回实际货币账单，`currency_cost=null`；不能把请求数或估算金额称为实际费用。
- 标签均为 LLM 标注，不是人工金标准。旧已诊断池保留为历史/开发诊断；最终主实验另冻结来源与模型后计算。

## 验证与待完成项

本轮离线回归覆盖：首遍请求后中断的恢复、403 fail-fast、并发预算上限、调低预算、历史 ID 碰撞、冲突视觉标签、唯一跨视频配对、帧号身份、配置更改拒绝续写。

2026-09-20：`uv lock --check` 通过；Python 3.11.14 / uv 0.9.17；`uv run --no-sync pytest -q` 110 passed；CLI 及两个标注脚本 `--help` 通过；`git diff --check` 通过。该验证不代表 GPU 模型、官方评分 parity 或完整矩阵已完成。

仍须完成：来源闭包审计与新训练集、Scene evidence 重训及正确记分卡、Spatial 四方向重训、UMT 接入、四维变换缓存、三方案评分、家族 CI、覆盖/弃权与失败样例、最终表格和报告。

## 来源依赖补充

恢复清单采用互不相交的双场景配对块（最后一块三场景），避免交叉配对形成贯通全部数据的一条环。492 项现为 492 个唯一 ID，11 个证据场景、71 个文本来源，但只有 **5 个独立来源依赖块**（129/92/92/91/88 项）。最终统计必须按来源块重采样，不能报 71 或 492 个独立家族。`scripts/audit_scene_sources.py` 同时检查 prompt、证据 prompt、视频和帧来源。

该审计还发现旧 v7 train/dev 共享 16 个来源（只检查 group_id 曾漏过）；新数据构建必须检查配对两端。冻结 E0 的四维 test 来源数为 Spatial 65、Objects 49、Scene 52、Action 60。`an ocean` 在 dev，ocean→sea 必须单列开发诊断，禁止改写 E0 或冒称盲测。

## 官方 Spatial parity 的新发现

对 H100 上锁定 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490` 的实际源码核验：

1. `get_position_score` 只比较 `abs(x_distance)` 与 `abs(y_distance)`，没有左右/上下符号检验。证据框镜像和同轴方向词互换不能改变这一判据的结果。
2. 官方 `check_generate` 把 object_a 或 object_b 的所有框放入同一列表，允许两个同名对象的框组成配对。旧 `score_origin_vs_repair.py` 却要求 A/B 各一个框，已经改变了官方实现。
3. 因此旧 Spatial 表 B 的 0.108/0.058 也不能直接称为已验证的官方比较。新三方案主表必须复用精确的官方几何公式并单列镜像失败；有向几何只能作为另列的判据消融，不能归因于文本修复。

## 冻结训练配置

`configs/formal/v8/` 沿用 v7 的模型 revision、LoRA r16、学习率 1e-4、batch 1×4、2048 tokens、seed 20260919。Spatial 固定 600 步，Scene 固定 300 步，使用 final checkpoint，不做按 dev 调参或测试选择。

Spatial 新数据保留 2,679 train / 312 dev，关系空间限四方向；保留 unsupported prompt 并监督空结果。under/beneath/underneath、over 与明确 on (the) top/bottom of 作统一文本映射；原始 target 与省略关系保留。训练与推理读取同一 adapter contract，越界输出计格式/覆盖失败。与 E0 spatial test 原始 prompt 的规范化精确重叠为 0；此项不代表近重复审计已完成。

Scene 构建器要求 evidence 标注完整后才产出训练集，测试保留集合包括 test2 配对两端和 E0 全部 test 场景；先划分来源再保留配对，跨 train/dev 配对进入排除清单，重新核验来源闭包。

2026-09-20 最新代码验证：115 passed（含 CPU 微型模型闭环）；未进行新的真实 GPU 训练。具体卡号等待用户按 AGENTS.md 的要求指定。

## 后端缓存与规则验证

新缓存格式保留所有 16 帧：Scene 不再用中间一帧代替视频；Objects 额外保存官方使用的 `det_obj` 标签集合；Spatial 保存实际 resize 后尺寸；Action 接入官方 UMT 架构、变换、sigmoid top-5 与四位小数阈值。逐视频持久化，并保留未运行/失败项的完整分母。正式采样使用冻结 E0 test 的全部 manifest 视频。

规则复算已与从锁定源码提取的原函数比较：2,000 组几何、6,000 帧、Scene 大小写/子串/空词以及 Objects 全合取反例均一致。此项只证明规则 parity；真实 UMT/GPU smoke 尚待执行。脚本 `check_official_parity.py` 在服务器上还会验证工作文件等于提交中的 blob。

冻结协议见 `configs/experiments/deterministic-v1.json`：三方案、16 帧、3% 不变性界、固定 2,000 次家族 bootstrap、完整失败分母；不按新模型结果改阈值。用户已确认 rtx4090 GPU 5 训练、H100 GPU 4 视觉缓存，此分配持续有效。

719/719 证据重标已完成（exit 0），新 Scene 数据为 455 train / 108 dev；train/dev/test2 双端来源重叠均为 0。Scene dev 只有一个连通来源组件，不能据此给出有意义的家族 CI 或选择模型。固定训练步数保持不变。

## 2026-09-20：新版 Scene 记分卡与冻结主清单

- `score_scene_metamorphic.py` 已改为从官方辅助键复算：区分大小写的子串规则；caption evidence 与 visual truth 分域报告。原始构造类型不能充当标签。
- Origin / Repair-rule / Repair-model 比较相同的 supported vs not-supported 二分类；Repair-model 的三分类另列。全清单缺失输出/未知标签计失败并明确为下界；仅已标注的平衡指标同时报告标签覆盖。无场景要求没有官方 key，单独列出适用域与全清单覆盖。
- CI 使用两个来源端点连接出的组件；每次重算同一个宏平均估计量，缺类别的重采样显式计数。少于两个组件不报 CI。新增同义“正确且一致”、raw consistency、copy-original、全条件常数对照。
- `batch_predict.py` 处理同一缓存的全部 16 帧，精确重复文本共享贪心推理；保存完整来源引用、每批持久化结果、模型/代码/输入身份，支持恢复。Scene 固定 batch 8、8 tokens、newline stop，未经标签选参。
- `build_matrix_plan.py` 从真实 manifest 和 E0 test 源冻结 29,660 条矩阵记录：Spatial 65 家族中 49 个方向家族适用（980 视频），16 个 inside-of 家族（320 视频）无四方向/官方元数据目标，仍保留不适用行；Scene 52 家族/1,040 视频，其中严格同义词表适用 10 家族/200 视频；Action 60 家族/1,200 视频，Objects 49 家族/980 视频。
- 词表是显式闭词典对照，不是新的人工金标；Action 类别替换并不自动证明替代动作在视频中缺席。`an ocean` 仍为 dev，未混入 test。
- 本地 `data/deterministic/matrix-v1/manifest.json` 记录原 manifest、官方元数据、E0 split、生成器与词表 SHA；数据不进 Git。Scene 原视频缓存 + 同义/类别替换 + test2 文本计划合计 8,218 个唯一输入、50,450 帧/标注引用，其中 13,440 为词表不适用占位，须与视觉缺失分开。
- 验证：完整 pytest 124 passed；CLI help、uv lock、diff check 通过。真实新 adapter 推理仍待 GPU smoke 与全量完成，不能把纯 CPU 测试作为其结果。

Spatial v8 固定 600 步已真实完成（exit 0）：2,679 条训练记录，train loss 0.029682，训练运行约 980 秒，`base_frozen=true`、`adapter_changed=true`；Scene 300 步已在同一授权 GPU 5 接续启动。H100 Action 1,200/1,200、Scene 1,040/1,040 原视频缓存均完整成功；Spatial/Objects 与变换缓存尚在继续。

## 逐帧视觉变换实现（真实后端 smoke 仍待执行）

`cache_matrix_transforms.py` 复用完整冻结原缓存，按 source video + evidence transform 去重，不按 Repair 输出挑选视频；同时保留所有矩阵行的引用。证据级镜像翻转框坐标；实际视频镜像使用无损 RGB 编码并在官方采样帧上要求像素精确等于原图翻转，不通过则保留失败。变换视频、采样帧、原缓存与代码身份均有 SHA。

Objects 在每帧使用全部匹配目标框，向外取整覆盖奇数/小数边界；level 0 完全恒等；背景对照整体平移原 patch，不裁剪或改变面积，且避开全部已检测对象；非目标对照不能与目标框重叠。无法定位或放不下对照明确记录失败。逐帧记录像素范围、遮挡外像素恒等、框覆盖率、全部 GRiT 描述/框/官方 det_obj 标签和缺失状态；完整端点 PNG 用于后续可见性审核。几何覆盖不等于真实不可见，`visibility_verified` 默认 false，不计作已验证成功。

缓存每个变换后持久化，恢复只复用已绑定身份；导出保留未运行行。空间不足保留至少 20 GiB 后停止，已完成记录不丢失。当前新增的五个算法回归用例通过，完整测试共 129 项；实际视频编码与 GRiT 验证须在服务器执行后另记，尚未将本实现称为完成实验。

## Action 模板暴露审计与固定预算隔离训练

已将四个现用 adapter 的训练文件 SHA 与运行 manifest 比对，全部相符。对 E0 test 原始 prompt 做规范化精确查重：Spatial 0/65、Scene 0/52、Objects 0/49；**Action v5 为 46/60**，均来自 K400 prompt 模板。旧 Action 结果只能作有训练文本暴露的诊断，不能标成独立文本测试。

`prepare_action_isolation.py` 在不读取预测/视觉分数的前提下，删除与原始及所有冻结变换 prompt 相同的训练记录所属完整来源组，以及原始/类别替换所关联的 K400 模板 ID 家族（含重新 teacher 标注但未带 k400_id 元字段的记录）。保留独立自然来源的同类动作示例；类别名称仍是预先声明的 400 类输出词表。原始 test split、视频与变换清单不变。

Action v9 数据为 605 train / 67 dev，删除旧 train 135、dev 14，合计 113 个来源组；train SHA `5d2b6bc69447f8df63c1c416dd7c2194d2aeb898514ff4eefbf66cb3909ba9ce`，dev SHA `a4746be124d26c384b6de064b79615d949507eb86af626c0594efdea99ec523a`。配置完全沿用 v5 300 步超参数，仅输入与输出目录变更，最终使用 final checkpoint；这不是按当前测试分数重新选模型。已有 v5 推理保留为暴露对照，主表待 v9 完成后使用其结果。

真实镜像 smoke 已修复 GIF 解码差异：MP4/GIF × hflip/vflip 四项均通过官方 16 帧精确像素翻转检查。没有 duration 元数据的 GIF 使用 10 fps 表示并保留所有 PIL 帧索引；有非均匀时间元数据的 GIF 暂不支持，会记录失败。报告 `output/upstream-parity/mirror-pixels-v2.json`。H100 已排队 `vpc-matrix-transforms-v1`（shell 2883920），待原缓存 shell 2864476 退出后依次完成 Spatial 和 Objects 变换。

## 统一三方案记分入口

新增 `score_matrix.py`：所有方案使用相同原缓存/变换缓存，逐视频配对；Spatial 继续严格保留官方无符号几何和同名框配对，Objects 用原始 det_obj 标签，Action 对 rounded top-5 sigmoid 分数用固定 0.85，Scene 对全部 16 captions 聚合。三解析任务的多目标输出使用确定性平均关系/动作或对象合取；Action `other` 计零并单列弃权，混合输出另外保留已知类分数。

Repair-rule 的 Scene 字典规范化同时作用于辅助场景键和 caption，字典为变换前声明的严格闭词表；不从测试标签学习规则。三个方案的相同二分类记分卡也使用这一规范化。

每个变换报告事先适用域与完整清单两个分母、原始/变换分数、配对差和来源家族 CI，以及缺失、覆盖、弃权。缺失两端的差为零也不能算不变性通过；全弃权不能算有效不变性成功。原分数为正条件下的下降只是诊断；缺少独立可见性/替代类缺席审核时，validated sensitivity 返回 N/A，不将 raw drop 冒充验证。

默认要求原缓存、变换与预测各自完成报告，未完成必须显式 `--allow-incomplete` 并在 JSON 中标记。生成逐项 JSONL、主表 CSV/Markdown 和完整 JSON；论文格式、失败案例、独立端点审核及最终汇总仍待完成。完整测试 **136 passed**，CLI help、lock check、diff check 通过；尚未宣称已获得全矩阵真实结果。

## Scene test2 真实复评（新 v8 final）

492/492 视觉标注和 492/492 caption evidence 重标均完成，后者 996 请求、240,361 input tokens、6,179 output tokens，provider 未给出实付金额。8,218/8,218 唯一 Scene 输入推理完成，无非法输出或证据缺失；13,440 引用是预先声明的不适用词表域，不是缓存缺失。

在 **相同 432 条官方有定义的场景样本**、5 个来源组件上，caption-evidence 二分类准确率 Origin/Repair-rule 为 **0.8171** [0.7789, 0.8985]，Repair-model 为 **0.9630** [0.9420, 0.9898]；配对提升 **0.1458** [0.0787, 0.1925]。平衡二分类分别 0.7772 与 0.9605。完整 492 条 Repair-model 三分类准确率 **0.8293** [0.7907, 0.8739]，平衡三分类 0.8294；55 条 contradicted 被预测为 insufficient，不能以二分类高分掩盖这一类错误。标签均为 LLM 标注，且家族数很小；此前诊断曝光不能称为全新盲测。

test2 可匹配的严格同义子集只有 38 对、1 个连通来源块，不给 CI。正确且一致率 Origin 0.6579，Repair-rule/Repair-model 均 0.9211；后两者的 copy-original 对照也为 0.9211，不能单凭此宣称模型有超出复制原判定的收益。原始与变换分数的一致性、正确性是不同量。

为覆盖此前已冻结主矩阵的全部严格同义域，新增**补充标签普查**：200 个视频、10 个源场景、全部 3,200 帧 caption，656 个唯一文本对，DeepSeek 请求上限 1,800。该补充计划在 initial test2 记分卡之后建立，选择只依赖此前冻结的同义适用域，不依赖预测/标签/分数；与 test2 单独报告，不改任何模型、阈值或数据 split。脚本 `prepare_scene_matrix_gold.py`，计划 `data/gold/scene-matrix-syn-v1/seed.jsonl`。尚未将补标或新家族 CI 称为已完成。

## Objects 独立可见性复核入口（待真实 API 验收）

原始 980 视频中，505 个在全部 16 帧均有目标定位。因此设置单独 1,800 请求上限，覆盖候选的两次独立视觉判断及必要的第三次仲裁。`annotate_object_visibility.py` 逐条读取变换追加日志；原视频分母仍为 980，几何不完整或无法准备图像的项保留未知状态。`prepare_visibility_contacts.py` 使用官方解码环境重建同一 16 帧，核验原帧、mask 重建、PNG 哈希，生成原始分辨率的 before/after 4×4 图板；不展示 GRiT/adapter 预测或框。

每次 API 分别判定四组 16 个可见性标签：before_target、before_other、after_target、after_other。至少两票同意才形成 visible/not_visible，否则 uncertain。端点不可见要求全部 after_target 为 not_visible；隔离移除案例进一步要求遮挡前两者全部可见、遮挡后另一对象全部可见。所有结论均标为 **vision-model consensus，human_reviewed=false**，不能称为人工金标。每个来源的请求、图像身份、共识与不确定性均持久化，401/403 或预算耗尽停止批次。

四个回归用例验证短/非法数组、单票不冒充真值、分歧无多数保留未知，以及流式读取不解析尚未写完的一行。全测试 140 passed；真实图板与 API pilot 待完成，评分器接入端点标签后再报告已验证敏感性。

## 实体接口序列化与可见性接入

首次 Objects 主矩阵评分前，接口检查发现文本规范名 ski/scissor 与原生检测标签 skis/scissors 不一致。评分入口现从**完整公开 VBench 元数据的 79 个实体**建立规范化的唯一逆映射，且仅对 Repair 的文本输出使用；不是从当前检测结果或样本目标反推答案。歧义映射保留未知原词，未命中名称不替换为标准答案。Origin 标签、检测缓存与合取/几何公式原样保留。回归覆盖已知序列化、未知目标不获答案，以及 Origin 不受规范化影响。

`score_matrix.py --visibility` 核对每条视觉标签与被评分变换记录 SHA 一致。全部目标不可见的模型共识端点，报告完整拒绝率和逐帧拒绝率；缺失帧不能当作拒绝成功。严格隔离移除还要求遮挡前两对象均可见、遮挡后另一对象仍可见；敏感性另条件于原分数为正、两端完整且下降至少 0.03。模型复核数、人工复核数和全计划下界分开保留，不将同一模型多遍共识称为人工金标。

真实 pilot 已通过解码/重建/PNG 哈希核验、两次独立提示与第三次仲裁响应校验；图板展示了框覆盖之后仍留下可见尾巴的反例，其不可见条件没有通过。API 延迟约 17–78 秒/遍，因此只改变执行调度：保留前 10 项 pilot，用互斥 ID 将其余项分成 8 片并发，全部共享原 1,800 请求预算与原脚本/模型/图像/标签规则，拒绝重复 ID 的最终合并仍以 980 为分母。

Scene 补充普查 3,200 帧、200 视频、10 来源家族标注完成：1,369 请求、328,056 input tokens、7,590 output tokens，金额未知。原始 caption 二分类准确率 Origin 0.57219、Rule 0.62438、Model 0.92906；同义正确且一致分别 0.26719 [0.19063,0.34813]、0.62438 [0.49219,0.75031]、0.92125 [0.88438,0.95563]。始终 supported 为 0.73375，模型 copy-original 为 0.92906；模型没有超越自身复制上界，也未达到完美不变性。该普查选择域在主清单中已冻结，但补标签发生在初始 test2 评分之后，单列补充证据。

## 训练来源扩展审计与最终产物身份

`audit_training_sources.py` 覆盖现用四维 train/dev、主矩阵全部原始/派生 prompt、test2 及同义普查配对两端。规范化精确文本、已记录测试视频路径/图像视频 SHA、Scene 标注来源两端均无重合，train/dev group_id 也无交集。固定字符相似度 ≥0.90 或 ≥12 词逐字包含筛查仅标出 Action 的 5 个原始 prompt 家族、2 个类别替换词（例如 cutting/eating watermelon、shaking/shaving head、tying tie/bow tie）；它们是不同 K400 模板类别，不能把字面相似当作同一来源，也不能据此声称语义近重复已被穷尽。原始训练来源保留 VG / MovieGen / K400 等出处，未按评分过滤或重新选模型。

四模型 final 权重均已与对应最后 checkpoint 的权重 SHA 逐字节确认相同：Spatial v8 600、Scene v8 300、Action v9 300、Objects v6 900；同时保存底座全部五个 safetensors 分片、tokenizer、adapter、配置与运行 manifest 的 SHA。RTX 训练 manifest 如实记录既存删除 `D scripts/score_deterministic.py`，没有清理 dirty 状态冒充干净训练。训练峰值显存当时未记录，后续报告应留空，不能拿空闲显存或设备总容量冒充峰值。

原始同缓存均分：Action Origin/Rule 0.88917、Model 0.865；Objects 三者均 0.28431；Spatial 适用 980 视频 Origin/Rule 0.29810、Model 0.28020；Scene 1,040 视频为 0.36382/0.37386/0.68912。均分变化不是人类正确率改善。Spatial 的 3 个原始 prompt 仍输出带冠词的对象字符串（an oven、an apple 等），实际 GRiT 标签匹配失败；该学生输出与判据保持不变。解析表同时区分**评分器实际接受的严格接口匹配**（46/49，0.93878）与额外规范化后的语义匹配（49/49）。ski/skis、scissor/scissors 是已声明的序列化等价，不在严格接口指标中误罚；冠词剥离没有偷偷加入评分流水线。Action 原始两个类被输出 other，严格匹配 58/60；Objects 为 49/49。参考标签均为公开元数据结构，不冒充人工金标。

`render_matrix_tables.py` 输出解析、生成器原视频、全矩阵变换、完整配对子集、Action 协议、模型可见性、Scene 分类的 CSV/Markdown/LaTeX；另给便于论文排版的紧凑表。Scene 三分类 macro-F1（含类混淆、非二分类准确率）为 test2 0.79337 [0.76210,0.81956]、补充普查 0.79412 [0.72706,0.85073]。现有 PDF 仅是原始视频/分类草稿，完整变换尚在运行，不能当最终交付。

ocean→sea 开发诊断全部 20 视频/16 帧以及 279 个唯一文本推理完成：Origin 原始 0.5625 → 0.03125，配对差 -0.53125；Rule 0.575 → 0.575；Model 0.959375 → 0.959375。只有一个来源家族，不给 CI。此诊断没有全量独立正确性标签，报告分数不变性，不把 0.959375 写成准确率。

UMT 实际 GPU parity 补充通过：四个生成器各选缓存判定的一正一负，共 8 视频，逐项与锁定上游 `human_action()` 入口重新执行结果一致；缓存/上游均值均为 0.5。选择覆盖两个分支，**不是随机质量估计**。上游源文件 blob、视频 SHA、权重与缓存 SHA 已绑定，报告 `output/upstream-parity/umt-eight-videos.json`，弥补此前单一负例 smoke 的局限。

同义补充普查的配对“正确且一致”提升：Model−Origin 0.65406 [0.57281,0.73375]，Model−Rule 0.29688 [0.16156,0.42500]，全部使用相同 10 个来源块重采样。test2 38 对仅 1 块，差值 0.26316 / 0，仍不报 CI。

`select_evidence_cases.py` 按已明确的机制选择字典序首例，仅作解释，不替换任何统计分母。`export_failure_frames.py` 在官方解码环境导出实际原帧、精确镜像、记录的遮挡或旧标注 PNG，核对视频/帧/变换身份。Action 预览从已核对 SHA 的 UMT 输入反归一化展示，学生原始输出与保守映射后的 other 分开记录；Scene 标签分歧明确允许 teacher 错误或歧义，不能把每个分歧都断言为学生错误。真实导出待验收；148 项测试与 CLI/diff check 已通过。

7 个实际帧案例已成功导出，视频/帧/UMT 输入/镜像身份核验通过。查看图像也发现“唯一检测框”不等于可靠视觉真值（例如生成苹果的茎被检测为 banana），因此这类例子只证明检测框层的公式问题，不能充当人工确认的方向正确率。另保存四个生成器各一个水平镜像候选，仍仅作解释。所有 4,520 原视频的字节 SHA 均不同，没有重复文件被当成独立原视频。

底座的五个完整权重分片现已与 Hugging Face 锁定 revision `b968826d9c46dd6066d109eabc6255188de91218` 的上游 LFS SHA256 逐项核对一致；本地目录没有下载元数据，故没有仅凭目录名/配置声称 revision 已验证。原始上游 JSON 与核验报告保存在 `output/audit-recovery/base-{upstream-file-metadata,revision-verification}.json`。

Action 同义 60 个独立表达中，Model 的 35 个最终输出 other，协议正确 25/60。原分数 0.865 → 0.37833，配对差 -0.48667 [-0.59750,-0.37583]；Rule 因使用预先声明的封闭同义词典，原/变换均为 0.88917。该规则是词典内对照，不是未见表达泛化证据。OOV 的 assembling furniture 实际只是一条独立输入，不能将重复在 1,200 视频上的同一个预测冒充 60 个独立拒识家族：新统计按唯一 prompt 计算协议准确率，OOV 不报 CI；视频配对分数仍按原始 60 家族重采样。混合已知/OOV 协议 Model 45/60，不能让 other 隐去已知类损失。

Objects 的背景对照分析新增同视频、同等级、两边均完整的配对差，避免不同有效子集直接比较；同时核验背景/目标 mask 等面积、遮挡外像素恒等、level 0 恒等，报告检测标签保留率。逐级单调性仅为完整梯度子集上的描述统计，不当作部分遮挡必然按比例掉分的语义真值。主表同时保留全分母与完整配对子集，未改 0.03 界限。

主评分入口进一步核查实际矩阵所需的缓存记录、所有变换 ID 和唯一文本预测是否存在，不仅相信外层退出码。输出绑定 scorer 源文件、Git commit/dirty diff、缓存/模型推理的支持清单 SHA；最终仍须在 Objects 缓存和标签齐全后运行。最新完整测试 150 passed。

最终 8 个 Objects 复核分片均 exit 0 后，第一次合并因请求 journal 含 8 条失败条目而触发 `KeyError: response`；标注与预测并未受损。汇总器改为分别计成功响应和失败请求，并绑定全部 journal SHA，失败账本不清零、不重新调用 API。恢复使用新的 `final-matrix-v1-retry1` job 目录与新代码 SHA，原失败日志保留，模型、标签与评分公式不变。

另澄清 test2 构造中名为 insufficient 的 60 条仅是“无官方 scene key”候选池：实际 caption 标签为 22 insufficient / 38 contradicted，不能把条件名当真值。共同二分类 432 项不包含此池；全 492 项三分类使用实际标签。候选池 43/60 标签一致不应称为无场景门控准确率。

合并恢复后，新 H100 checkout 缺少 `data/raw/k400/labels.json` 使评分在加载词表时停止。已复制本地冻结词表，SHA `97f3e3ee0e60d1b35ba47fe0fc2fce86f6124d25950af4c8df5cec88eee8f1a9` 与原矩阵 manifest 一致，恢复目录 `final-matrix-v1-retry2`。未改标签、模型、规则或词表内容，未重复 API 调用；前两轮失败日志均保留。
