# VBench 1.0 四维三方案确定性实验

版本说明：本文保存 **matrix-v1 的共同官方评分公式实验**。Spatial 已在后续 [repair-v2](deterministic/spatial-repair-v2/README.md) 实现有符号后端；Action 已完成[同义接口修复](deterministic/action-repair-v2/README.md)；Objects 已增加[相邻帧确认](deterministic/objects-repair-v2/README.md)。本文旧 Spatial/Action/Objects 分数用于历史对照和组件消融，不代表当前默认 Repair。Origin、Scene 保持相同。

四维当前版本的统一表格、结论和局限见[最新结果汇总](deterministic/current-summary/README.md)。

状态：**四维三方案矩阵已完成并验收**。29,660 项清单、88,980 条评分记录、81 个维度/方案/变换汇总；无未执行的冻结项。遮挡无效与未经人工核验的限制仍保留。

本轮检验的是**评分接口在明确变换下的行为**，不是生成视频质量提升。三方案共享原视频、采样帧、视觉模型、权重与评分公式：Origin 使用官方文本条件，Repair-rule 使用冻结规则，Repair-model 使用固定步数的 final adapter。结果并不支持“四维均获得改善”：Scene 在模型标注的 caption 证据上有收益；Spatial 的官方公式忽略方向符号，文本修复无法消除此问题；Action 模型的同义表达泛化明显失败；Objects 原始实体接口已一致，差异主要由检测器与遮挡有效性决定。

## 交付与阅读顺序

- [完整表格目录](deterministic/matrix-v1/README.md)：CSV、Markdown、LaTeX；A 解析、B 原视频及生成器、C 全矩阵、D 不可见端点、E Scene 分类、F 完整配对、G Action 协议、H Scene 同义、I 水平/垂直镜像、J 匹配背景对照、K 条件通过率。
- 本地完整附录 [tables.pdf](../output/deterministic/matrix-v1/tables/tables.pdf) 与紧凑主表 [paper.pdf](../output/deterministic/matrix-v1/tables/paper.pdf)。PDF 表中分数均在 0–1 范围，CI 是 95% 来源家族区间。
- [逐项评分与执行报告](../output/deterministic/matrix-v1/report.json)、[配对记录](../output/deterministic/matrix-v1/paired-rows.jsonl)、[遮挡对照](../output/deterministic/matrix-v1/object-controls.json)。原始记录不进入 Git。
- [实际帧与解析案例](../output/deterministic/matrix-v1/gallery-final/index.html)，以及 [资源与身份汇总](deterministic/matrix-v1/provenance.json)。案例只解释机制，不替代全量统计。

## 范围、分母和统计约定

冻结 [实验协议](../configs/experiments/deterministic-v1.json) 和 E0 test 划分，四个生成器为 CogVideo、LaVie、ModelScope、VideoCrafter。4,520 个原视频的文件 SHA 均不同。清单共 29,660 项，三方案产生 88,980 条配对评分记录。一个视频的 16 帧不会变成 16 个独立来源家族。

| 维度 | 原视频计划数 | 适用数 | 适用来源家族 | 主要变换 |
| --- | ---: | ---: | ---: | --- |
| Spatial | 1,300 | 980 | 49 | 框证据镜像、真实视频镜像、方向词交换、联合变换 |
| Objects | 980 | 980 | 49 | 目标、等面积背景、非目标遮挡，各 4 个等级 |
| Scene | 1,040 | 1,040 | 52 | 严格同义、场景条件替换；同义适用 200 视频/10 家族 |
| Action | 1,200 | 1,200 | 60 | 同类同义、新 K400 类、OOV、已知类与 OOV 混合 |

Spatial 的 16 个 inside-of 家族（320 视频）预先不在四方向协议/官方辅助条件域内，仍保留在计划分母和不适用清单。Scene 840 视频没有冻结词典内的严格同义，不拿宽泛近义补足。ocean→sea 在 E0 dev，仅单列诊断，没有移动测试划分。

视频配对差为变换后减原始分数；均值按视频计，bootstrap 按源 prompt 家族整块重采样 2,000 次，seed=20260919。跨配对 Scene 标签按提示词/证据两端的连通来源块重采样。少于两个块不报 CI。区间是描述性区间，未做多重比较校正。固定不变性容差与正基例下降界均为 0.03，未按结果调整。

缺失、非法输出保留并计零，另报覆盖率、弃权率和两端完整子集；**由缺失产生的下降不能解释为敏感性成功**。成功率不允许整条弃权获得不变性信用。Scene insufficient 是弃权；Action other 是评分域外，既不满分，也不能抹掉混合输出中已知类别的错误。适用范围排除与执行缺失分开统计。

## 原始视频与文本接口

| 维度 | Origin | Repair-rule | Repair-model |
| --- | ---: | ---: | ---: |
| Spatial（适用 980） | 0.29810 | 0.29810 | 0.28020 |
| Objects（980） | 0.28431 | 0.28431 | 0.28431 |
| Scene（1,040） | 0.36382 | 0.37386 | 0.68912 |
| Action（1,200） | 0.88917 | 0.88917 | 0.86500 |

这些是后端评分均值，不是人类正确率。各适用原视频缓存与文本执行覆盖率为 1；模型原始 Scene 弃权率 0.21695，Action 为 0.03333。按生成器的分数、相对 Origin 的配对差和 CI 见表 B。

模型原始严格接口：Spatial 46/49=0.93878，Objects 49/49=1，Action 58/60=0.96667；均为有效 JSON，解析有效不代表语义正确。参考为公开元数据结构，并非人工金标。Spatial 额外规范化后的语义 F1 为 1，但实际评分没有剥离模型输出的冠词，3 个家族的 `an oven`、`an apple` 等字符串不能匹配原生实体名。完整公开 79 实体词表只产生 ski→skis、scissor→scissors 两个唯一序列化逆映射，应用于 Repair 文本，未修改检测结果或 Origin。

Action 的 `aerobics` 在保守冻结映射下成为 other，`riding or walking with horse` 也被模型输出为 other；均如实记错，没有在看分后新增别名。Origin/规则的原始结构解析高分部分来自直接使用官方元数据，不应描述为学习泛化。

## Spatial：matrix-v1 未修复方向，v2 已完成后端修复

以下描述首轮共同公式实验。[后续 v2](deterministic/spatial-repair-v2/README.md) 将 Repair 改为有符号主体—客体几何，并保留以下 Origin 对照；新收益属于确定性后端修复，不是 adapter 重训收益。

官方 Spatial 对水平/垂直几何使用绝对差，忽略 left/right、above/below 的符号，并会从 A 或 B 的框池中组合两个同名目标框。三方案保持完全相同的官方公式。因此，确定性镜像框坐标后，三方案分数均保持不变（Δ=0，CI=[0,0]），不能满足有向关系翻转的要求。改成有符号公式将是另一项后端消融，不能算本轮 adapter 收益。

真实无损视频镜像并重跑 GRiT 后，Origin/规则总体 Δ=−0.00125 [−0.01366,0.01107]，模型 Δ=−0.00367 [−0.01544,0.00838]。水平 600 视频/30 家族中三方案均 Δ=+0.00560；垂直 380 视频/19 家族，Origin/规则约 −0.01207、模型约 −0.01830。水平/垂直完整 CI 见表 I。变化来自检测器对镜像的不等变性及接口输出差异，不是方向判据已正确翻转的证据。

同步修改视频与方向词后，Origin/规则仍 Δ=−0.00125，模型 Δ=−0.00206 [−0.02023,0.01698]。模型不能修复评分公式本身。检测框满足有向关系只是检测器代理条件；没有全量人工核验的方向准确率。实际帧案例还显示检测器可能把苹果茎误认成 banana，故“唯一检测框”也不是视觉真值。

## Objects：遮挡完整性与不可见端点

后续 v2 已用同名相邻框确认抑制孤立检测，275 个模型复核不可见端点中的正分视频 29→15；被过滤者是弃权，全部明确阴性的端点仍为 246。原始均分 0.28431→0.26173，原始可见帧检出率 0.63843→0.61858；持续误报与召回损失均未隐去。详见 [Objects v2 的取舍、消融与全量回归](deterministic/objects-repair-v2/README.md)。下文保留首轮结果。

遮挡等级 0、0.5、0.8、1.0 是框的**线性边长比例**，不是视频可见面积比例。只把记录的 mask 内像素置为 RGB 128，mask 外像素逐帧核对恒等；level 0 是恒等对照。背景对照平移同一 mask 保持面积，非目标对照要求不覆盖目标框。目标未定位、找不到空背景或非目标框相交均保留为无效帧。

11,760 项变换全部尝试：4,716 完整、7,044 部分有效；188,160 帧槽中 88,084 有效，49,572 目标未定位、24,650 等面积背景不可用、17,142 非目标未定位、8,712 非目标 mask 与目标相交。全部尝试完成不等于全部变换有效。完整配对及同视频、同等级、两侧完整的背景对照分别见表 F/J，不能直接比较不同有效子集的均值。

目标框覆盖不自动说明对象消失。独立于 GRiT/adapter 输出的图板复核使用同一视觉模型至少两遍、分歧第三遍仲裁；共识可能相关，全部 human_reviewed=false。不可见端点要求 16 帧目标均 not_visible；严格隔离移除还要求遮挡前两对象均可见、遮挡后另一对象仍可见。只有满足复核、两端完整、原始分数为正且下降至少 0.03 才统计条件敏感性。部分遮挡单调性仅作描述，不强制解释为线性语义规律。

最终复核保留 980 个端点：505 项完成模型标注，475 项几何不完整而保留未知。505 项均至少有两遍有效响应，其中 275 项目标在全部帧不可见，142 项满足严格隔离移除；它们分别只占全计划的 28.06% 和 14.49%，不能把该条件子集外推为全部视频。

三方案全部 12,740 个 Objects 视频/条件的分数、覆盖逐项相同，**没有文本 adapter 收益**。全计划重度目标遮挡均分 0.28431→0.01754，Δ=−0.26677 [−0.31001,−0.22500]，但变换帧覆盖率仅 0.64872，不能把这个全量下降全归为正确敏感性。505 个两端完整视频为 0.41423→0.02772，Δ=−0.38651 [−0.45683,−0.31838]。

275 个模型确认的不可见端点中，246/275=0.89455 [0.83036,0.94636] 全帧拒绝；29 个仍至少有一帧正分。逐帧拒绝率 0.96409 [0.93576,0.98564]。严格隔离移除且原始正分的 115 视频/40 家族中，114/115=0.99130 [0.97222,1.00000] 下降至少 0.03。全计划已核验且全帧拒绝的保守下界为 246/980=0.25102，未知端点没有算作成功。

同视频、两侧完整的目标减背景遮挡分数差：level 0.5 为 −0.11007 [−0.20417,−0.03634]（67 视频/29 家族），0.8 为 −0.35771 [−0.48692,−0.23983]（47/25），1.0 为 −0.40500 [−0.59615,−0.22656]（25/16）。重度背景对照只覆盖 25/980，选择性很强，不能外推全数据。对原先已检测到的另一对象，重度目标遮挡后标签保留率 0.83311，说明存在附带影响；等面积背景遮挡对应为 0.99531。两者是各自有效帧条件下的检测保留率，不是不同子集之间的因果效应。

四级完整目标梯度中，容差 0.03 内的描述性单调比例 0.78020 [0.72797,0.83366]（505 视频），不当作严格语义通过率。像素恒等、成功帧遮挡外不变、背景/目标等面积核验全部通过；几何无效帧仍保留。

## Scene：同义收益与正确性边界

200 视频/10 家族的同义主矩阵：Origin 0.30594→0.05406，Δ=−0.25188 [−0.44094,−0.04500]；规则 0.35813→0.35813；模型 0.69094→0.69469，Δ=+0.00375 [−0.01188,0.02188]。逐视频 |Δ|≤0.03 的通过率为 0.53、1、0.82。规则使用已声明的闭合严格同义词典，是词典内控制，不是未见表达泛化。

仅分数不变还可能是稳定地答错。对同一域全部 200 视频×16 captions 的 3,200 对补充证据标注，比较 supported vs not-supported 的**正确且一致率**：

| 方法 | 正确且一致率（95% CI） | 原始 caption 二分类准确率 | 原/同义原生标签一致率 |
| --- | --- | ---: | ---: |
| Origin | 0.26719 [0.19063,0.34813] | 0.57219 | 0.64188 |
| Repair-rule | 0.62438 [0.49219,0.75031] | 0.62438 | 1.00000 |
| Repair-model | 0.92125 [0.88438,0.95563] | 0.92906 | 0.96094 |

模型相对 Origin 的配对提升 +0.65406 [0.57281,0.73375]，相对规则 +0.29688 [0.16156,0.42500]。始终 supported=0.73375，始终 not-supported=0.26625，复制模型原始判定=0.92906；模型超过常数对照，但略低于自身复制对照，仍有不变性损失。3,200 条来自 10 个来源家族，不是 3,200 个独立家族。补充域预先存在于冻结矩阵，但标签补充在 initial test2 记分卡之后完成，**不是全新的盲测**。

恢复的 test2 共 492 观测、5 个连通来源块，432 个官方场景条件可定义的共同二分类项：Origin/规则 0.81713 [0.77888,0.89848]，模型 0.96296 [0.94200,0.98980]，配对提升 +0.14583 [0.07868,0.19250]。模型全 492 条三分类准确率 0.82927，macro-F1=0.79337 [0.76210,0.81956]；补充普查三分类 macro-F1=0.79412 [0.72706,0.85073]。test2 的 38 个同义对仅一个来源块，不报 CI；该小切片不能替代 10 家族补充普查。

test2 构造中名为 insufficient 的 60 条是**没有官方场景键的候选池**，不能全部当作已核实的无场景金标。文本 teacher 实际给 22 insufficient、38 contradicted，模型给 25/35；43/60 标签一致不能命名为“无场景门控准确率”。部分文本仍可能含隐含场所要求。主共同二分类不包括这 60 条，三分类附表保留实际标注与不确定性，未按学生结果改标。

视觉标注与 caption 证据标签分开保留。test2 中 83/492 的两者标签不同；这显示信息差异，不能全归因于 caption 幻觉。学生只见 caption，主分类结果是相对模型生成的 caption 标签的一致性，不是人工认可的视觉正确率。模型/teacher 分歧案例允许 teacher 错误和歧义。旧 0.319/0.799、0.672/0.807 口径不再用作主结果。

ocean→sea 单独 dev 诊断为 20 视频/1 家族：Origin 0.56250→0.03125（Δ=−0.53125），规则 0.57500→0.57500，模型 0.95938→0.95938。后两者视频不变性为 20/20；没有全量独立正确性标签，0.95938 不是准确率；单家族不报 CI。

## Action：matrix-v1 模型链路失败，v2 同义接口已修复

后续检查原始生成文本发现，35 个同义 other 中 31 个是非 other 动作短语被旧归一化器拒绝，4 个为模型直接输出 other。下表保留旧链路结果；[v2](deterministic/action-repair-v2/README.md) 复用既有词典，改进输出短语归一化并用当前输入的完整同义契约约束类别，达到 60/60 类别一致、1,200/1,200 视频分数一致。这是固定 v9 模型加确定性接口的系统收益，不是新 LoRA 泛化成绩。

| 条件 | Origin 原始→变换 | Repair-rule 原始→变换 | Repair-model 原始→变换 |
| --- | --- | --- | --- |
| 同类同义 | 0.88917→0 | 0.88917→0.88917 | 0.86500→0.37833 |
| 新 K400 类 | 0.88917→0.00167 | 0.88917→0.00167 | 0.86500→0.00167 |
| OOV | 0.88917→0 | 0.88917→0 | 0.86500→0 |
| 已知类 + OOV | 0.88917→0 | 0.88917→0.44458 | 0.86500→0.34125 |

模型同义 Δ=−0.48667 [−0.59750,−0.37583]。60 个独立同义表达仅 25 个目标集合正确，35 个成为 other；规则 60/60。把全部 other 当稳定输出会掩盖失败，本报告不给这种行为不变性成功。规则知道冻结的词典同义，不能据此声称学到了开放表达泛化。

OOV `assembling furniture` 只有 **1 个独立提示词**，虽然重复配到 1,200 视频。规则与模型均正确输出 other，但协议样本数是 1、无 CI，不能声称稳健开放集拒识。混合已知/OOV 有 60 个独立输入，模型集合正确率 45/60，规则 60/60；other 计零使混合总均分减半，已知类保留另报，不能用总分下降指控已知类一定丢失。

新 K400 类替换时文本目标确实改变，但新动作在视频中不存在尚无全量独立真值核验；Scene 场景替换也一样。因此类别替换表只报告条件控制分数，不把所有下降称为经人工确认的敏感性成功。

## 实际帧案例

11 个案例按预定机制取字典序首例，仅作解释。`a vase and scissors` 的 frame 5 中，花瓶被灰块覆盖，GRiT 仍输出 vase 和 scissors，合取仍为 1；`a dog and a horse` 显示框覆盖后仍有可辨认目标部位，不能称完全不可见。toothbrush/sink 控制例中，图像中的洗手池仍可辨认，但 GRiT 的变换后标签没有 sink，说明像素/语义隔离也不保证检测标签稳定。Spatial 水平/垂直、冠词接口，Action aerobics→other，Scene 三标签分歧和同义例同时保留。全部图片来自原始证据或精确变换，未用生成式图像补画；案例不用于重选阈值或修改标签。

## 训练、来源与身份

| Adapter | train/dev | 固定 final 步数 | Trainer 时间（秒） | train loss |
| --- | ---: | ---: | ---: | ---: |
| Spatial v8 | 2,679 / 312 | 600 | 980.1921 | 0.029682 |
| Scene v8 | 455 / 108 | 300 | 512.7962 | 0.109157 |
| Action v9 | 605 / 67 | 300 | 462.4345 | 0.124953 |
| Objects v6（复用） | 7,688 / 811 | 900 | 940.1288 | 0.098269 |

四者 base_frozen=true、adapter_changed=true；final 权重与最后对应 checkpoint 的 SHA 相同。没有按 dev/test 选择 checkpoint 或超参数。训练在已授权 RTX4090 GPU 5；H100 GPU 4 做视觉缓存。Trainer 时间不包含所有启动/下载开销；训练峰值显存没有记录，保留未知。共有 43,646,976 可训练参数，LoRA rank 16/alpha 32、all-linear、学习率 1e−4、cosine、batch 1×accumulation 4、max length 2,048，配置完整保存。

底座为 Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`；全部 5 个 safetensors 分片均与[该 revision 的上游 LFS 哈希](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218)一致。adapter、tokenizer、配置、数据、缓存、评分代码均有 SHA。RTX 历史训练 manifest 如实保留既有 `D scripts/score_deterministic.py`；Objects v6 还记录当时未跟踪的配置目录，实际配置已存，不冒称全部训练来自干净 checkout。

旧 Action v5 在 60 个测试原始 prompt 家族中有 46 个模板暴露。来源审计后整组移除 113 个与原始/派生测试模板有关的来源组（135 train、14 dev 行），固定原超参数训练 v9，主表只用 v9。来源隔离独立于模型得分。当前四维 train/dev 对原始及派生测试 prompt 的规范化精确重合、已记录测试视频/图像身份重合、Scene 配对两端重合均为零，train/dev group_id 无交集。

固定字面近重复筛查只标出 Action 的 5 个原始家族、2 个类别替换词，属于不同 K400 模板类别，例如 cutting/eating watermelon、shaking/shaving head。该筛查不是穷尽语义近重复；视觉身份审计只覆盖记录了身份的来源。未根据测试错误再训练或加别名。本测试原始 prompt 均为 2–12 个空格分词：Spatial 8–12、Objects 4–7、Scene 2–4、Action 4–8 词。这些结果不能外推为 201+ 词输入表现；本轮三方案矩阵未加入未微调底座、全四维交叉隔离性或人工金标研究，这些不在已完成主张内。

## 官方一致性与复现

上游固定 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。源码规则 parity：2,000 组几何、6,000 帧与边界用例通过；UMT 实际 GPU 入口复算 8 视频逐项一致，覆盖四个生成器各一正一负（分支覆盖，非随机质量估计）；GIF/MP4 水平/垂直无损像素 parity 4 例通过。Action 使用官方 top-5 sigmoid 置信度先四舍五入 4 位再比较 0.85；Objects 使用原生实体合取；Scene Origin 匹配大小写敏感的辅助场景键子串。

项目 Python 3.11.14 / uv 0.9.17，依赖锁不变。官方视觉栈继续使用 H100 单独 Python 3.10.20、torch 2.5.1+cu121 环境；不是声称两套 Python 完全相同。GIF 若原文件无时长元数据，镜像使用 10 fps 表示并保留官方 PIL 全部帧索引；MP4 保留时间戳，评分采样像素核验完全一致。执行缓存报告、权重/帧哈希与评分 supporting manifests 一起保留；API 模型别名不是供应商不可变权重版本，冻结标签可复算，不能保证未来 API 逐字再生。

复现评分入口为 `scripts/score_matrix.py`，缓存参数包含四个 `*-test.jsonl`、Spatial/Objects `*-transforms.jsonl`，预测必须使用 `spatial.jsonl`、`objects.jsonl`、`scene.jsonl` 和 **`action-v9.jsonl`**，可见性使用 `merged.jsonl`。`scripts/analyze_object_controls.py` 产生匹配对照。表格命令：

```bash
uv run --no-sync python scripts/render_matrix_tables.py \
  --matrix data/deterministic/matrix-v1/matrix.jsonl \
  --paired output/deterministic/matrix-v1/paired-rows.jsonl \
  --score-report output/deterministic/matrix-v1/report.json \
  --metadata output/upstream-parity/VBench_full_info.json \
  --predictions data/repair/matrix-v1/spatial.jsonl \
  --predictions data/repair/matrix-v1/objects.jsonl \
  --predictions data/repair/matrix-v1/scene.jsonl \
  --predictions data/repair/matrix-v1/action-v9.jsonl \
  --scene-card test2=output/deterministic/matrix-v1/scene-test2-scorecard.json \
  --scene-card synonym-census=output/deterministic/matrix-v1/scene-matrix-syn-scorecard.json \
  --object-controls output/deterministic/matrix-v1/object-controls.json \
  --out output/deterministic/matrix-v1/tables
# 在输出目录分别运行两遍，以稳定 longtable 宽度：
pdflatex -interaction=nonstopmode -halt-on-error tables.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

本轮五个独立用途账本共 5,829 次预记账请求、5,820 次成功响应，9,157,964 输入 tokens、1,752,819 输出 tokens；9 次失败已保留。其中 Objects 为 1,230 请求（1,222 成功、8 失败）。此前 test2 失败批次另有 1,571 请求；早期其他训练标注账本不混入上述本轮小计。实际金额未知。

资源汇总可用 `uv run --no-sync python scripts/summarize_matrix_provenance.py --out docs/deterministic/matrix-v1/provenance.json` 从冻结产物重建。资源账本见 provenance.json：按用途保留预记账请求、成功响应、输入/输出 tokens；历史 test2 失败批次的 1,571 请求另列，未清零。供应商没有返回实际金额，currency_cost=null，请求数不是货币支出。相邻 vbench-audit 的冻结数据、结果、划分与 runs 保持只读。
