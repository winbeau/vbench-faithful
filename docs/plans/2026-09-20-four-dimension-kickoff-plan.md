# 四个候选维度开工计划

日期：2026-09-20。状态：供实施的计划，尚未启动实现或模型实验。

**首要交付是四个能接入现有工作区的独立 metric 包。** 每包有自己的 `pyproject.toml`、`src/`、测试和 console script，由根 `uv workspace` 统一管理；四包共用现有输入、CLI、元数据、调度、输出和 provenance 协议。先验收这些工程接口，再实现算法和反事实实验。不能交付四套互不兼容的研究脚本。

工程实施者优先读 C 节的目录/统一接口合同与 H.2 的 M1 验收；B 节算法实验属于接口就绪后的阶段。

本次仅新增本文。后文涉及的包、配置、测试和工具是后续实施交付物，不表示已经存在，也不授权本次修改它们。保留工作树已有改动；不下载权重，不改 `../VBench`，不 commit/push，不写 `data/`、`results/`、`splits/`、`runs/`、`figures/`。后续新实验只写新的 `output/four_dimension_20260920/`，不覆盖已发布反事实数据或历史 scores。

依据按要求顺序阅读：`AGENTS.md`、[11 维范围决定](2026-09-15-dimension-scope-11d.md)、[候选源码审查](../paper/unaudited-dimensions-triage.md)、[既有实验协议](2026-09-14-experiment-plan-8d.md)、[数据构造协议](../counterfactual-dataset.md)、[汇总与 review 入口](../counterfactual-reports/CONSOLIDATED.md)、[六维语义适配器约束](../semantic-adapter-constraints.md)、[补充实验手册](../supplementary-experiments.md)、`configs/upstream.toml` 和上游源码；另核对了 README、工作区重构约定、`subject-consistency` 与 `scene` 的包结构及 core 实现。

## A. 目标、范围与可证伪命题

本轮范围仍是 11 维：在办 7 维 `dynamic_degree`、`motion_smoothness`、`subject_consistency`、`scene`、`human_action`、`spatial_relationship`、`multiple_objects`，新增候选 4 维 `background_consistency`、`temporal_style`、`object_class`、`color`。`overall_consistency` 包、入口、测试和 pin 保留为 legacy；本文没有删除安排。语义适配器按六维理解，本次先做确定性解析/归一化，不把 LoRA 训练设为开工前置条件。

| 维度 | 要检验的命题与预期方向 | 可证伪条件与结论边界 |
| --- | --- | --- |
| `background_consistency` | 命名≠实现：整帧表示受到前景变化影响；同一背景异常的位置会影响首帧锚点评分。期望背景异常降低分数、等量前景变化基本不影响背景分数、异常平移基本不改处罚 | 在独立确认的背景/前景编辑及等面积位置实验上，若 Official 已满足上述关系，不能声称这些机制造成了实测违约。源码无背景分解是结构事实，不等于所有视频都失败 |
| `temporal_style` | 命名≠实现及截断：整条 prompt 的内容匹配不能单独证明风格匹配；相反风格若得到相同 32-token 输入，估计器无法区分它们。期望真实风格高于经人工确认的矛盾风格，等义子句换位不改风格评分 | 若实际 tokens 未丢失风格、或者 Official 对矛盾风格已有区分，不宣称该样本受截断伤害。十种风格可能共存，不能假定生成 prompt 就是视频真值 |
| `object_class` | 严格字符串匹配：同一视觉证据仅改变类别叫法就可能改变 Official 命中。期望同义/类别词单复数/大小写改写不改变类别存在性；真实目标高于明确不存在的异类 | 若真实 GRiT 输出与全部审核过的别名均无实际评分差异，则只能报告规则脆弱性，不能报告真实标签漂移率。不能把视觉漏检当成词法错误 |
| `color` | 严格字符串匹配、实例错位及条件分母：颜色证据可能绑定到错误实例，子串产生假阳性，未匹配样本退出分母。期望保持实例身份的重排不变、仅交换目标与干扰物颜色降低原目标得分、降低目标可见帧比例降低全帧支持率 | 若真实输出没有第二实例、绑定已一致，或白名单/丢弃未造成实测偏移，分别报告零事件及样本上界。修复工程缺陷并不保证自然偏好一致性提高 |

### A.1 源码身份与行号复核

**以下行号已于 2026-09-20 复核**，均针对 `Vchitect/VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。本次实际运行 `verify_upstream()`，remote、SHA、clean 状态和 12 个源文件哈希全部通过；这不是模型 parity。

| 维度 | Official 入口 | 已复核的主证据 |
| --- | --- | --- |
| Background | `vbench/background_consistency.py:68` `compute_background_consistency` | `:39` 全帧读取；`:42` 整帧 CLIP；`:49`、`:50`、`:51` 为非负 prev/first cosine 均值；`:64` 按转场总数聚合。读取默认见 `vbench/utils.py:108`、`:154`、`:155`，公式对应 `vbench/subject_consistency.py:58`、`:59`、`:60` |
| Temporal | `vbench/temporal_style.py:67` `compute_temporal_style` | `:48` 整条 prompt；`:55` middle 采 8 帧；`:59` 至 `:61` 视频/文本 cosine；`vbench/third_party/ViCLIP/viclip.py:25`、`:152` 限制 32 tokens；`vbench/third_party/ViCLIP/viclip_text.py:122`、`:150`、`:152`、`:153` 截尾并补 EOT |
| Object | `vbench/object_class.py:72` `compute_object_class` | `:32` 集合化类别；`:33`、`:34` 异常变空集；`:40` 精确成员；`:50` 从 aux 读 object；`:52` middle 采 16 帧；`:68` 全帧成功率 |
| Color | `vbench/color.py:93` `compute_color` | `:36`、`:37` 每个 caption 绑定 `[2][0]`；`:46` 类别精确相等；`:47` 至 `:55` 白名单分母与颜色子串；`:65` 至 `:67` prompt 字符串手术；`:69` 采 16 帧；`:81` 至 `:89` 条件率与视频丢弃 |

四个 pin 的 SHA-256：

```text
background_consistency  7def92057819df0dad01c857eecf3b39b8b555c8cdfe881a672cc2b19dc226b4
temporal_style          9f9e950efca4673721d0a07ea83137be9ebdc9aef2364f412300ba979beef57a
object_class            8d4c4a7feee52503e3d27f8aba0ed467cbc356ed75da2598367b572237e05db2
color                   03baffc12508fd7ce6c70a227d6794bbc7152c24ba2d2c447a816974466db3d7
```

补充定位：GRiT tuple 把完整类名列表重复放进每个实例，见 `vbench/third_party/grit_src/image_dense_captions.py:57` 至 `:65`；阈值 0.5 与 beam=1 见该文件 `:88`、`:80`。两路 head 从相同 proposals 计算 boxes，见 `vbench/third_party/grit_src/grit/modeling/meta_arch/grit.py:35` 至 `:40` 和 `vbench/third_party/grit_src/grit/modeling/roi_heads/grit_roi_heads.py:237` 至 `:257`；自由文本解码在后者 `:299` 至 `:301`。实例 scores 在 tuple 转换时已经未返回，不能仅撤销 `set` 就声称恢复了置信度。

对提示词摘要/triage 的更正或收窄：

1. Temporal 与 Overall 的文本 diff 除名称、dimension 串外，还差 `vbench/temporal_style.py:54` 的一行注释；替换名称后的 AST 相同。应写“执行逻辑相同”，不是字节逐行相同。
2. 集合精确成员**区分大小写**；triage 中“不区分大小写”的字面表述有误。
3. 32 长度包含 SOT/EOT；截断时最多保留 30 个内容 tokens。31 个英文词的最长 prompt 必须被截断，但不能由词数推出整个风格子句不可见；必须区分部分截断、完全不可见与 token 碰撞。
4. Color 的视频保留条件比“检出对象”更窄：还要有 caption 命中白名单颜色子串。当前 85 条官方 prompt 的字符串解析得到 10 种正确对象名；`pizza box → pizz box` 是扩展输入反例，不是官方 suite 已出现的错误。
5. `[2][0]` 是返回列表的第 0 个实例；不能未经运行验证就把它等同于文本置信度重加权后的最终最高分实例。报告同时保存返回顺序和实际 confidence rank。

### A.2 引用时必须声明的共享关系

- Background 与 Scene：`evaluate.sh:7`、`:10`、`dimension_to_folder.json` 映射到同一 `scene/`；本次元数据统计验证 86 条 prompt 集合完全相同。Background 没有自身 aux，Scene 有。共享的是视频/prompt 来源，**不是评分公式或人类标注**。涉及 scene 视频编辑的论文结论须同时注明 Background 输入受到扰动；不能把旧 `environment_coverage` 的 Repair 优势迁移过来。
- Temporal 与 Overall：共享同一估计器，使用不同 prompt suite。整 prompt cosine、截断和缓存诊断也影响对 legacy Overall 机制的引用；Temporal 上测出的违约率、风格判别性能不能冒充 Overall suite 实测结果。只读源码比较不启动第五维评分。
- Object/Color 的归一化规范影响六维适配器中 GRiT 标签契约的描述；不自动改变其余 metric 的代码或重报其旧结果。

## B. 每维契约、反事实族与 repair

### B.0 所有族共用的规则

“base”指独立源视频/源 prompt 组；同源视频、改写、派生片段及复用 donor 按关联组件分组，不能跨 dev/test。来自 E0 的输入沿用既有 prompt split；没有既有 split 的候选先按稳定 hash 划分。先定资格和预期关系，再看任何 metric 分数。下列数量是**目标预算，不是现有可用数量**；不足时交付缺额和原因，不重复采样凑 base、不在看过 test 分数后放宽资格。

- `ordered`：预注册有向比较；不要求无依据的两个负例互相排序。逐 base 计算全部合法 pairs，另报严格顺序率和 Spearman（有合法 ladder 时）。
- `same-rank`：语义上同级，报告 score range、相对 range、MAE 与 tie CPA；不会因为可调 margin 给出 1.0 就宣布成立。
- `invariance`：字节、query 语义或冻结证据只发生无关改写。缓存/索引级不变性要求数值完全相同；真实模型的数值 parity 容差与语义 same-rank 的容差分开，见 F 节；两者均同时报告原始差值。
- 编辑族必须重编码 level 0，保持时长、采样位置、分辨率和编码参数一致；记录解码后的帧/timestamp hash。不用生成模型重新生成基础视频，只从现有官方视频做确定性派生。缺视频则阻断对应 GPU 实验。

### B.1 `background_consistency`

| 族 | 变换与预期关系 | base 数；来源/是否生成 | 资格与判据 |
| --- | --- | --- | --- |
| `background_temporal_relocation` | 同一固定背景区域、同一异常贴片、相同强度/帧数分别放 start/middle/end；`clean > 每个 corrupt` 为敏感性半边，三个 corrupt 为 same-rank 半边 | 5 dev + 20 test；每 base 4 clips；用官方 `scene/`，需派生编码 | 无切镜、背景相对稳定；前景 median-box 经人工审核，背景编辑区域与前景轨迹不相交；逐位置保存面积、强度、异常帧数、转场数；分别报告两半及位置 range |
| `background_severity` | 固定 middle、固定区域/时长，背景异常 alpha 为 0/.25/.5/.75/1；期望分数 ordered 下降 | 同一 25 bases × 5 levels；与上族共享 clean、middle-full | alpha 仅是施工参数，人工须确认背景异常强度顺序；不根据 CLIP 分数挑 donor；用于防止敏感性只在极强破坏上饱和 |
| `foreground_background_selectivity` | 同等面积/强度/时长的 foreground-only 与 background-only 编辑；`clean ≈ foreground`（same-rank），`clean > background`（ordered） | 同一 25 bases × 3 levels；仅 foreground 版本额外编码；三族去重后约 200 clips | foreground 和 background 各有独立有效区域；不得误把前景编辑泄漏到背景。检测器只需稳定定位一个前景，不要求两个目标；语义区域正确性由人工确认 |

median-box 复用的是 subject 的施工约束，不导入它的 metric 模块，也不继承旧得分。每个 base 对前景框取中位数并冻结；若移动主体超出固定框，不能声称剩余区域是纯背景，须人工修订构造或记录 ineligible。评分侧自动 mask 与施工用 oracle mask 分开存放、分开报告，避免同一掩码既定义真值又制造必然通过的分数。

自动 box 资格初值为：在上游 0.5 检测门槛下，至少 80% 解码帧原生检出目标前景；补帧/最近邻继承不能增加这个资格比例。人工 mask 队列不受检测器资格过滤，单独计数；二者都要求剩余背景区域非空、位置族的编辑区域在所有位置保持同面积。

Repair 拆分：

- `vbench/background_consistency.py:42` 的整帧输入 → 在 metric 内构造前景抑制图，前景填充固定 RGB 常量、背景保持原像素，再使用同一 CLIP ViT-B/32 与相同 resize/crop。保留 raw/processed mask、来源、面积、前景漏出率、原图与抑制后 feature。没有合格 mask 时不回退成整帧并冒称背景分数。
- `:49` 至 `:51` → 去 first anchor 的 `masked_prev_mean`；另实现 `masked_prev_worst = min_t max(0, cos(z[t-1], z[t]))`，检验全局均值稀释和端点边界。初始推荐把 worst 作为候选 Full、mean 作为消融，**最终选择只在 dev 冻结**。worst 对单帧异常敏感，必须保留失败结果。
- 端点异常与中间异常有 1/2 次转场差异，prev-only mean 不自动满足 same-rank。保留端点主契约，同时用 5 个 dev bases 做有 clean guard 的 early/middle/late 诊断；不在 test 上删除端点以制造胜利。如所有候选都失败，报告 repair 未解决该契约。
- 消融矩阵：整帧+原公式、抑制前景+原公式、整帧+prev、抑制前景+prev mean/worst。公式与抑制策略属于本 metric；mask provider、CLIP 加载/特征读取属于共享模型适配，不在 core 里放 cosine 聚合。

离线证据：86 prompt/目录耦合、公式结构可由 M0 判定。前景泄漏、位置效应与 repair 性能必须走 M3 的 H100 测量。所需模型为 CLIP，GRiT 仅用于可选自动前景资格/box；无权重可用人工冻结 mask、合成 feature 做合约测试，但不能产生真实 CLIP 结论。

### B.2 `temporal_style`

已复核 suite 是 10 个 base 内容 × 10 个风格子句，共 100 prompts，无 `auxiliary_info.temporal_style`；最长三类 prompt 的词数为 31/30/29。**独立内容组只有 10 个，不是 100 个。** 默认 3 个内容组 dev、7 个 test；每内容组每风格选 1 个可用官方视频，共目标 100 source clips。增加生成器只能增加重复观测，不能把内容组数变大。

| 族 | 变换与预期关系 | base 数；来源/是否生成 | 资格与判据 |
| --- | --- | --- | --- |
| `style_clause_swap` | 每个视频固定 pixels/base 内容，评估同 base 的 10 个风格子句；只对人工确认的“支持风格 > 矛盾风格”记 ordered pair | 3 dev + 7 test 内容组；100 视频 × 10 queries；官方 `temporal_style/`，不生成新视频 | 不用 GRiT；人工查看真实运动，标注 supported/contradicted/compatible/unclear；pan 与 smooth 可共存，不把其强行当负例；报有效关系数及 10 类相似度矩阵 |
| `style_clause_position` | 同一 style clause 置于内容前/后，语义不变；Repair style 分量 invariance | 同一 10 内容组、100 视频，每视频 2 queries；无新视频 | 从 suite 的末尾子句表精确切分；若自由文本不能无歧义拆分则 unsupported，不能用测试视频反推 style；保存 token 序列与位置 gap |
| `style_truncation_collision` | 保持同一长内容前缀，只替换尾部两个互相矛盾的风格；同时提供 style-only 对照；支持风格应高于矛盾风格 | 同一 10 内容组、100 视频，每视频 2 个长 query，style-only 复用首族；无新视频 | 长前缀为重复原内容的受控文本挑战，不冒充自然 prompt；tokens 完全相同即可证明不可区分。真实语义方向只在人工确认支持/矛盾的视频上评分，其余保留 unclear |

约 1,300 个去重后的 `(video, query)` 请求/后端，只有约 100 个视频文件。精确数由 manifest 生成器输出。

Repair：`vbench/temporal_style.py:48` → `content_text` 与 `style_text` 分离；借鉴 `vbench/appearance_style.py:50` 的 aux 读取口径，但**不改 Official 输入**。style text 包含特殊 token 后必须 ≤32；超长拒绝或使用 dev 已冻结的等义短写，原串及改写均保留。`:59` 至 `:61` → 同一 ViCLIP 视频 feature 分别计算 style cosine 与 content cosine；公开主分数先用 style cosine，content 单独报，不加权混成“风格更好”。

同 base 的 10 个 style-only logits 必须一起输出，附目标 rank、相对差值及固定温度 softmax（初始 `temperature=0.01`，仅诊断，不当校准概率）。没有互斥标注时不将 softmax top-1 当准确率。消融为 full-prompt、style-front、style-only、style-only 相对判别。保留 full/kept token IDs、截断 token 数、风格保留比例、视频采样帧、各文本/视频特征和内容/风格分数。

切分/结构化字段属于适配器职责；读哪个 query、相对判别与标量定义属于 metric 评分职责。需要 ViCLIP 及其 BPE，不需要 GRiT、CLIP ViT-B/32 或 MUSIQ。M0 可验证 AST、suite、词数及模型无关 token 碰撞；实际分数/排序必须 M4。机制引用涉及 legacy Overall，实测率仅适用于 Temporal suite。

### B.3 `object_class`

| 族 | 变换与预期关系 | base 数；来源/是否生成 | 资格与判据 |
| --- | --- | --- | --- |
| `object_label_invariance` | 同一视频改写 aux 目标类别：原标签、大小写、类别词单复数；审核过的同义词另附。类别存在性 invariance | 5 dev + 20 test，不同源 prompt；每 base 至少 3 queries；官方 `object_class/`，不生成视频 | 主队列由人工确认目标存在，不以 Official/Repair 分数筛选；检测条件子集需 GRiT 在 ≥25% 的 16 个采样帧定位目标。类别复数不引入“两个对象”的计数命题 |
| `object_target_contrast` | 原目标、外观相近但不存在的类别、无关且不存在的类别；原目标分别高于两个负例；负例之间无顺序 | 同一 25 bases × 3 queries，原目标复用；总计约 125 请求、25 视频 | 人工确认目标/负例真值；不要求同一帧定位两个目标。若视频真实含某个负例，预先换负例或标 invalid，不能看分数后改标签 |
| `object_unsupported_status`（合约 fixture） | 对不能无歧义表达的类别、计数/身份要求做等义改写；预期 unsupported 状态 invariance，`score=null` | 79 个官方标签的解析检查 + 20 个扩展反例；无视频，不计 GPU CPA | 未观测到的 GRiT 类名不等于模型不支持；只对明确超出解析/任务能力的 query abstain，保留 raw query 与原因 |

Repair：`vbench/object_class.py:32` 的 tuple/set 消费 → 从原始 `Instances` 保存所有 boxes、`det_obj`、description、两路 confidence（确实可取到的字段）、原顺序和阈值信息；必须验证观察 hook 不改 Official 返回值。`:40` → 使用版本化确定性同义表、大小写/空格与类别词单复数归一。上下位关系不当同义词，不能把未检测到的目标改成检测器已输出的类别。

主分数仍为固定全部采样帧上的目标支持比例，成功推理但没有目标命中记 0；confidence 先作诊断，置信度加权另作命名消融，不默认更换主分数。类别抽取和标签归一属于六维适配器职责；实例证据保留属于模型适配；命中谓词及帧聚合属于 metric。将 `exact_miss_alias_hit` 与 `no_detection`、`wrong_class`、`runtime_failure` 分开，真实词法漂移率只在人工核对后命名。

M0 对 79 个标签和人工构造的预测集合可证明目标串改写改变规则输出；没有实际 GRiT trace 时不能推算发生率。M5 使用 GRiT ObjectDet，阈值保持上游 0.5；其他三种模型均不需要。没有权重时仅跑 parser、固定 trace 与合约测试。

### B.4 `color`

| 族 | 变换与预期关系 | base 数；来源/是否生成 | 资格与判据 |
| --- | --- | --- | --- |
| `color_label_invariance` | 相同视频/语义，用对象大小写、去掉不改变含义的冠词、审核过的对象别名/gray–grey 改写；invariance | 主队列 5 dev + 20 test；至少 3 queries/base；官方 `color/`，无需新视频 | 原目标颜色由人工确认；同义词仅在适用子集报告数量，不凑 25 个 gray 样本；不能把 crimson/navy 无条件归为 red/blue |
| `color_visibility_denominator` | 目标原本颜色正确；使其可见帧比例为 100/75/50/25/0%，其余帧确定性遮挡目标；全帧分数 ordered 下降，0% 不得消失 | 同一 25 bases × 5 clips；需派生编码 | 单目标可定位即可；施工 mask 需逐帧可用且人工确认不改其他对象；检测失败帧保留。Official 条件率可能不变或无返回，分别记录 |
| `color_instance_swap` | 两个不同类别、不同颜色的实例 A/B；只交换实例颜色，固定原目标 query；original > swap | 另 3 dev + 12 test 配对 bases × 2 clips；先找现有官方视频，不足时可用两个现有官方片段确定性并置，需单列 composite 层 | **必须验证 GRiT 能在同一帧定位 A/B**：原始同帧检出率 ≥25%，原有 bbox 对应人工正确率 ≥80%；要求两路 head 的 boxes/order 可核验。人工确认原目标及交换真值 |
| `color_instance_permutation`（证据重放） | 将完整 `(box,class,caption,score)` 实例记录一起置换，预期正确算法完全 invariance；额外加入 red/colored、red/hundred 等词边界反例 | 复用上项 15 bases 的真实 traces，各 2 permutations；另 12 色 × 6 类文本 fixture；不生成视频 | synthetic fixture 不需检测器；真实重放需先成功捕获两实例。原规则的 trace replay 与真正调用 `compute_color` 的 Official 数值分栏，不能混称 |

主队列 125 clips + 双实例队列 30 clips，约 155 视频、205 Official/Repair 请求；证据重放另计。两队列若共享原输入，按源 prompt 聚类；组合视频的两个 donor 全部留在同一 split。

Repair 必须分开消融，避免把四种变化合为一个无法解释的差值：

1. `vbench/color.py:37` `[2][0] → [2][idx]`；先 assert caption/class/box 长度和顺序。若两路顺序不一致，用原始 boxes 做一对一 IoU 绑定（dev 初值 ≥0.9，歧义 abstain），不得重新取最高分实例代替。**这是评分输入绑定错误的修复，不是 LLM 解析能力。** 共享模型包负责返回忠实实例记录，metric 负责选目标与消费正确关联。
2. `:46`、`:50` → 对象别名归一 + 有词边界、否定/歧义处理的颜色支持谓词；颜色归一属于适配器，谓词真值和实例聚合属于 metric。`colored`、`hundred` 不能支持 red；caption 没提颜色是 evidence insufficient，不能当视觉真值。
3. `:47` 至 `:55`、`:81` 至 `:89` → 主分母固定全部采样帧；成功推理却未发现目标/颜色证据的帧支持值为 0，整条视频保留。条件率 `P(color evidence | target and usable caption)` 只作诊断，并明确分母。
4. `:65` 至 `:67` → Audit 使用显式 `object`、`color` 字段或经验证的 prompt 编译；Official 原样传 raw prompt + color aux，保留原规则对反事实 prompt 的失败。

依次比较 Official、binding-only、binding+lexical、binding+lexical+all-frame。保存逐实例原/归一类名、caption、颜色 token spans、box/head 对应、置信度、每帧命中与分母标志，以及 Official 缺失的视频列表。

需要 GRiT DenseCap 与 ObjectDet 两路，使用同一 checkpoint；不能直接把 ObjectDet 的缓存当作 Color 的 DenseCap 输出。CLIP、ViCLIP、MUSIQ 均非必需。M0 fixture 可确定索引/子串/解析缺陷；真实错位占比、丢弃率和分数偏移必须 M6。缺权重只完成 trace fixture 与合约验证。

## C. 实现拆分：独立包、同一接口，先工程后算法

### C.1 四个 uv 子项目

| 发行包/CLI | 独立项目文件 | Python 源码目录 | 独立测试目录 |
| --- | --- | --- | --- |
| `background-consistency` | `metrics/background-consistency/pyproject.toml` | `metrics/background-consistency/src/background_consistency/` | `metrics/background-consistency/tests/` |
| `temporal-style` | `metrics/temporal-style/pyproject.toml` | `metrics/temporal-style/src/temporal_style/` | `metrics/temporal-style/tests/` |
| `object-class` | `metrics/object-class/pyproject.toml` | `metrics/object-class/src/object_class/` | `metrics/object-class/tests/` |
| `color` | `metrics/color/pyproject.toml` | `metrics/color/src/color/` | `metrics/color/tests/` |

每包都交付如下结构；不可只有根脚本，没有可独立构建/测试的包：

```text
metrics/<kebab>/
  pyproject.toml
  IMPLEMENTATION_REPORT.md
  src/<snake>/
    __init__.py
    cli.py
    metric.py
    models.py
    diagnostics.py
    backends/
      __init__.py
      vbench.py
      audit.py
      repair.py
  tests/
    test_cli_smoke.py
    test_counterfactual_contracts.py
    test_official_adapter.py
    test_repair.py
    test_runtime_parity.py
```

- 子项目 `[project]` 独立声明 name/version/Python 范围、实际直接依赖；`[project.scripts]` 指向 `<snake>.cli:main`；setuptools 按 `src` 寻包。基础依赖尽量保留纯算法/接口需要的集合，模型依赖进入各自 `[project.optional-dependencies].models`，不能依靠另一个 metric 的间接依赖。
- 根 `pyproject.toml` 保持 `members = ["metrics/*", "packages/*"]`，新增四包的根 dependencies、`[tool.uv.sources]` workspace 映射和 models extra 聚合。各包依赖 `audit-core`，实际使用共用模型时直接依赖 `audit-models`。**一份根 `uv.lock`** 管理所有子项目，不各建相互漂移的 lock，不改 `.python-version` 的 3.11.14。
- `uv run --package <kebab> <kebab> --help`、`uv build --package <kebab>`、该包自己的 pytest 均须成立。从 metric 子目录执行 uv 也应找到相同 workspace。独立包意为依赖显式、可单独构建/安装；共用包尚未发布时，独立环境须显式安装本地 `audit-core`/`audit-models` wheel，不声称任意 PyPI 环境可直接获得它们。
- 不能 `import subject_consistency` 来复用背景公式，也不能 `import object_class` 来实现 Color；全仓跨 metric 导入检查覆盖 importlib 动态导入与发行包依赖。现有 Subject 未提交实现只作为结构参考，不移动、不覆盖。

### C.2 必须复用的公共输入输出架构

| 公共能力 | 唯一实现位置/现有接口 | 新四包必须遵循的行为 |
| --- | --- | --- |
| CLI 参数 | `audit-core/cli.py:build_parser` | `--vbench/--audit/--both` 必选互斥；`--video/--video-dir` 必选互斥；共同支持 `--metadata`、`--gpu`、`--output`、`--seed`。维度参数仅追加，不另造 `--mode official` 绕过基础协议 |
| 输入枚举 | `audit-core/inputs.py:enumerate_videos` | 单视频接受已有有效路径；目录只读直接子文件 `video_<至少三位数字>.mp4`、按数字排序；空目录/非法名称/重复数字索引在加载模型前失败，不四包分别 glob |
| 元数据 | 同文件 `find_metadata/load_metadata` | 显式 `--metadata` 优先，否则查输入旁 `metadata.json`；沿用 `{"videos": [...]}`、`video` 键，保留未知扩展字段；缺失/重复映射先报错。每维只验证自己的语义字段 |
| GPU 与分片 | `audit-core/devices.py`、`coordinator.py` | 使用共同解析/校验和 spawn 生命周期；每隔离进程一张可见卡、逻辑 `cuda:0`；both 两套后端分别完整运行并释放模型，不把结果复制到另一后端 |
| 输出路径 | `audit-core/paths.py:output_base` | 默认仓库根 `output/<kebab>/<vbench\|audit>/<run-id>/`；显式 `--output DIR` 替换 output 基目录；不能随 cwd 改变默认位置 |
| 输出结构 | `audit-core/schemas.py`、`outputs.py:write_results` | 同一 `VideoResult`/`RunSummary`；同一套 `results.json`、`results.csv`、`summary.json`、`run.json`、`run.log`；both 共用 run-id；再次运行不覆盖 |
| 来源与版本 | `audit-core/upstream.py` | 校验 remote/SHA/clean/hash/import origin；标准 provenance 记录模型/输入/代码 hash，不能每包定义一套版本字段 |

**不新造第二套用户入口或结果 schema。** `backends/audit.py` 可做诊断和后端编排，`backends/repair.py` 放候选修复；两者在 CLI 中均属于 `--audit`，以 `variant` 区分 `diagnostic`、`repair` 和消融。未实现的分支必须 `not_implemented`、`score=null`、非零退出，不能通过相同分数假装 `--both` 已就绪。

四包共用同一内部 batch 约定，实施时在 core 给它明确类型/合约：`evaluate_batch(backend, videos, metadata, device, config) -> list[VideoResult]`。metric 自己提供算法、模型 factory 和聚合函数，core 只负责调用/调度/序列化。研究脚本调用这个约定或相同 CLI；不绕开它直接重写每维输入处理。

元数据示意，表示共同外壳而非要求四维都读取所有字段：

```json
{
  "videos": [{
    "video": "video_000.mp4",
    "video_uid": "source-or-derived-uid",
    "query_uid": "query-uid",
    "base_id": "base-uid",
    "prompt": "a red car",
    "dimension_metadata": {
      "object_class": {"object": "car"},
      "color": {"object": "car", "color": "red"}
    }
  }]
}
```

| 维度 | Official 最小语义 | Audit 语义扩展（字段置于该维命名空间） |
| --- | --- | --- |
| Background | 只需 video，prompt 不参与评分 | `foreground_masks` 或可审计的 box/mask manifest；mask 为 oracle/自动来源必须显式标识 |
| Temporal | 完整 `prompt` | `content_text`、`style_text`、固定 `style_bank`/`base_content_id`；能由冻结 suite 无歧义编译，否则缺失/不支持 |
| Object | `object` 转成 `auxiliary_info.object_class.object` | raw/canonical object、解析 supported 状态及原因；不根据检测结果改 query |
| Color | raw `prompt` + `auxiliary_info.color.color` | 显式 object/color、支持状态、别名表版本；不把修复后的 object 反灌 Official |

输出的 `video/status/score/error` 沿用 core；`backend` 固定 `vbench` 或 `audit`，`variant`、`formula_version`、`video_uid`、`query_uid`、`base_id`、`prompt` 作为统一扩展列。当前 `VideoResult.metric` 是内部扩展容器，writer 会展开到 JSON 行顶层；不另输出一个不兼容嵌套格式。大数组放本次 output 的 evidence 文件，行内保留路径和 hash；CSV 保留标量，JSON 保留 diagnostics 引用。

`summary.json` 保留现有字段并以兼容的可选字段补 `status_counts`、`denominator_kind` 和计数；不要让下游猜 `valid_samples` 是视频数、帧数还是转场数。Official Background 用转场加权，Object 用帧计数，Temporal 用视频/query 均值，Color 用上游实际保留的视频条件均值；merge 使用逐项充分统计量，不能平均 shard 均值。修复聚合另记 formula version。

### C.3 Official、共享模型与审计工具的边界

1. **Official 新四包必须调用锁定上游的 `compute_*`。** `backends/vbench.py` 将共同 metadata 转成 output 临时 JSON，传入本地权重配置，保存原始返回和规范化结果。一个 shard 可一批调用以减少模型重载；单个坏输入导致批失败时，记录失败再按显式恢复策略隔离，不能在后台悄悄重试并抹去失败。不得从现有 Subject 包复制公式就称为官方后端；现有包的优化路径也不能替代这项要求。
2. 直接构造 submodule 参数：Background `[absolute_clip_path, False]`；Temporal `{"pretrain": absolute_viclip_path}`；Object/Color `{"model_weight": absolute_grit_path}`。不调用会自动补文件的 `init_submodules()`，不改上游文件、默认值或源码 hash。
3. `packages/audit-models/src/vbench_audit_models/` 按需新增 `clip.py`、`viclip.py`、`grit.py`、mask/语义编译接口。只负责显式本地模型加载、帧/特征/实例忠实输出与可审计缓存。GRiT ObjectDet/DenseCap、RGB 预处理和不同 resize 路径分别建缓存 key；不能仅凭同 checkpoint 合并结果。
4. `packages/audit-core/` 只补共同 schema/状态、路径、provenance、稳定 ID、输入视图和分片恢复缺口；所有余弦聚合、类别命中、颜色分母与 ranking 公式留在对应 metric。`configs/four_dimension/` 只放可审阅 TOML/JSON 规范，不藏 Python 评分逻辑。
5. 后续交付一个薄研究驱动 `scripts/four_dimension.py`，用于 B 节数据构造、调用四包、合并与报告；**不是第二套评分实现**。现有 `scripts/counterfactual/score.py` 的维度白名单只有旧七维，`run_dimension.py` 有取首行 family 的逻辑，不能直接假定它们已支持四个新维度/多族。新驱动使用公共结果 schema，与旧报告格式作显式转换。
6. 驱动的拟定命令为 `offline`、`inventory`、`prepare`、`validate`、`parity`、`score`、`merge`、`report`、`freeze`。公共参数为 `--root`、`--upstream`、`--dimension`（仅新四维，`offline/inventory` 及报告集成模式可省略）、`--protocol`、`--models`；具体命令见 H。这些是**后续研究工具的命令合同，目前不存在**：M2 交付 offline/inventory，M3–M6 逐维补足数据/模型/报告分支；它们不作为 M1 四包统一接口验收的前置条件。

### C.4 接口阶段的完成定义

四包都应在无 CUDA/无权重环境完成 help、参数错误、mock batch、元数据、默认输出、both 分离、跨目录调用、异常和四 shard 合并测试；独立构建与直接依赖检查通过。此时可以交付“包与统一接口就绪”，不能交付“真实评分完成”。无需等全部反事实或模型实验结束才能评审工程接入。

## D. 数据、元数据与模型清单

### D.1 来源、独立性与存储

- 优先只读 H100 既有 `/root/wenbiao_zhao/datasets/vbench-1.0-human-preference/`，实际目录/视频/标注由 inventory 逐项确认；不能由 prompt 数推断视频已齐。Background 显式映射 `scene/`；其余查 `temporal_style/`、`object_class/`、`color/`，缺失记录 `missing_input`。当前七维发布集不是这四维已有数据的证据。
- 元数据以锁定 `vbench/VBench_full_info.json` 为 suite 来源：Background 86、Temporal 100、Object 79、Color 85。视频按 source annotation/archive、generator、路径和内容 SHA-256 关联；不靠文件名猜 query，也不继承另一维人类标签。
- 统一 CLI 的编号目录通过**新的 output 输入视图**映射到原视频，使用只读来源的 symlink 或明确的派生文件，不重命名源输入。同一视频的多个 query 建独立 view/metadata；保留原 `video_uid` 和独立 `query_uid`，禁止 basename 碰撞。
- GIF 等输入若不能由对应上游直接解码，保留 native unsupported/failed；如另做确定性 MP4 转码，只写新的 output，记录原容器/timestamp/派生 hash，标为转码输入上的对照，不能称原容器 parity。
- 新 manifest/split/派生片段分别放 `$FD_RUN/manifests/`、`$FD_RUN/protocol/`、`$FD_RUN/clips/`；detector traces、features、masks 放 `$FD_RUN/evidence/`。施工与评分的不同配置各有 fingerprint，不能把旧缓存当成新计算。
- 最低人工工作：所有入选 base 的资格与方向先独立审核；Temporal 的 100 视频 style 状态由两人标注、分歧仲裁。另每维抽 20 对、每个主族至少 5 对，三人盲审，共 80 对/240 judgments；机械字节/记录重排控制由确定性检查验证。家庭有效性目标为 ≥80% 多数票支持预期、invalid/unclear <10%；未达标只在 dev 重设计，test 失败保留并撤回该族命题。
- 自然偏好兼容性为后续条件项：只有确实存在该维独立人类标注时才计算 pair accuracy；没有就填 `--`。不借 Scene 的标签评 Background，不借 Overall 的标签评 Temporal。首轮预算不包含大规模新偏好标注或全量自然集评分。

### D.2 权重/运行时核验与无权重降级

| 资产 | 使用维度 | H100 路径与已知状态 | 缺失时的工作范围 |
| --- | --- | --- | --- |
| CLIP ViT-B/32 | Background Official/Repair | `/root/.cache/clip/ViT-B-32.pt` 曾用于 Scene Repair；本次未登录复核，M2 记录存在性/hash，再由 Background parity 验证 loader/preprocess | 只做 feature fixture、mask/输入输出测试；不以其它 CLIP checkpoint 顶替 |
| GRiT | Object、Color；Background 自动 mask 资格可选 | `/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth` 有既有运行记录；M2 仍须复核文件和 Detectron2 运行时 | 固定 Instances/trace mock、parser/词边界测试；自动检测和实测漂移率 blocked |
| ViCLIP | Temporal | `/root/.cache/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth` 是上游预期位置；**是否已存在尚未证实**，Overall legacy 包存在不等于权重就绪 | AST/prompt 统计、mock features；BPE 存在时可单独 CPU tokenize；不产出真实相似度 |
| ViCLIP BPE | Temporal 文本审计/模型 | `/root/.cache/vbench/ViCLIP/bpe_simple_vocab_16e6.txt.gz`；M2 核验 | 缺失时保留词数下界与源码截断机制，精确 token 数/碰撞统计标未验证；不下载，不默用 VBench 2 的文件替代 |
| BERT tokenizer 资产 | GRiT | 上游 `vbench/third_party/grit_src/grit/modeling/roi_heads/grit_roi_heads.py:51` 调用 `bert-base-uncased`；须核对既有本地 HF cache | 不允许为 tokenizer 联网补文件；GRiT 模型测试 blocked |
| MUSIQ | 无 | 不属于本轮四维的必需模型，是否已有不影响开工 | 不增加画质评分/权重任务；编辑副作用用原像素、mask 检查与人工验证 |

RAFT 既有目录和 Scene Tag2Text 可记录 inventory，但不进入四维必需计算；SAM/其他分割模型也不作为默认新依赖。开发机无 VBench 权重、无可用 CUDA；本机只承担文档、包结构、纯算法/合约与无模型离线证据。

**导入前阻断自动获取文件。** `vbench/utils.py:244` 起多条初始化分支会补权重；ViCLIP `vbench/third_party/ViCLIP/simple_tokenizer.py:10` 至 `:16` 会补 BPE，且 `:67` 的默认参数在导入时求值。必须先检查预期缓存位置再 import；设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，并在合约测试中禁止 network/subprocess 获取路径。仅设置 HF offline 不能阻止上游 wget，因而文件预检不可省略。所有模型使用显式已存在路径，缺失分支不尝试模型构造。

## E. H100 计算、分片与恢复

### E.1 机器与运行身份

H100 工作目录 `/root/wenbiao_zhao/vbench-audit`，解释器 `/root/wenbiao_zhao/venvs/vbench/bin/python`，上游 `/root/wenbiao_zhao/VBench`。正式前核对代码与本机待实施版本的 source manifest；由于禁止 commit，`git HEAD` 不足以唯一标识工作树，记录 HEAD、dirty、diff hash、所有参与源码/配置（含 untracked）hash、`uv.lock` hash。不要通过 commit 构造版本号。

默认物理卡 **1,2,3,4**；5 仅为故障替换/资源不足时经检查后的备用，6 不可用。以 UUID/PCI bus ID 核对物理映射，每 worker 在导入 torch 前设置唯一 `CUDA_VISIBLE_DEVICES`，运行时 assert 只见 1 张卡，传 `--gpu 0` / `cuda:0`。不沿用旧 H200 的 4–7、旧手册的 0–3 或“6 卡”描述；不终止既有进程。

### E.2 首轮预算（资源预留，非测速结果）

按 2–4 秒短视频、完整 B 节队列，Official 与 Repair 加小规模机制消融估算；帧数/时长必须 inventory 实测，长视频不能硬塞本估算。所有表内机时均为待 pilot 校准的范围。

| 顺序 | 目标视频/请求 | 卡数 | 预留 GPU 卡时 | 四卡墙钟（含装载/校验余量） | 主要开销 |
| --- | --- | --- | --- | --- | --- |
| Background | 约 200 clips/200 queries | 4 | 4–8 GPU·h | 1–2 h | 全帧 CLIP、mask 资格、原/抑制图消融 |
| Temporal | 约 100 clips/1,300 queries | 4 | 8–16 GPU·h | 2–4 h | ViCLIP；Official 的多 query 会重复视频编码，不能按只有 100 次编码估价 |
| Object | 约 25 clips/125 queries | 4 | 8–16 GPU·h | 2–4 h | GRiT 16 帧、目标改写、原始实例证据 |
| Color | 约 155 clips/205 queries，另 trace replay | 4 | 16–32 GPU·h | 4–8 h | 双 head、实例资格、可见度/颜色编辑与分阶段消融 |
| 合计 | 不含缺失资产处置和新增自然集 | 每次最多 4 | **36–72 GPU·h** | **9–18 h** | 可再预留约 20% 恢复余量，墙钟约 11–22 h |

单维先取 4–8 个 dev 视频（Temporal 覆盖长/短文本，Color 含零/一/多实例）实测装载、每帧/每视频 p50/p95、显存和 CPU 解码时间。预计墙钟用各 shard 的 `load + Σvideo_time` 最大值加合并耗时计算，不简单用单卡时间除四。超出预留两倍时先报告规模/时长原因；优先减少消融级数，不减少独立 test bases 来追求显著性。

### E.3 单维串行闸门与输出

- 顺序固定为 **Background → Temporal → Object → Color**。每维完成 dev、冻结协议、parity、test、结果去重/覆盖检查、按族报告和 review 清单后才开下一维 GPU 测量；不能四维各占一卡混跑。每维内部四个隔离 shard 并行。
- 某维缺模型/视频导致 blocked 时，只能形成明确的 blocked 交付和缺额记录；不得记录为“完成评分”。默认不跳过闸门启动下一维，是否改顺序列为末尾决策项。
- 同维 Official/Repair/消融按后端或 variant 分阶段运行；Official 真调用 `compute_*`。Repair 可复用已验证的相同输入 feature，cache key 必须包含视频/frame/preprocess/checkpoint/head/query 相关 hash；Official 重放数字独立标 `rule_replay`，不代替直接上游调用。
- 原始共同 CLI 输出放 `$FD_RUN/cli/<dimension>/<variant>/shard-<k>/<kebab>/<backend>/<run-id>/`；规范化 scores/evidence 分别在 `$FD_RUN/scores/<dimension>/`、`$FD_RUN/evidence/<dimension>/`。每次重跑产生新 attempt，保留旧日志。
- 在固定 `(dimension, backend, variant, query_uid)` 层内按 `video_uid` 去重；不同 query 的同视频不得互相覆盖。跨卡恢复用完整任务键匹配 frozen expected set，允许合并相同 fingerprint 的重复成功项，但内容冲突立即报错；不能用“最后一条覆盖”隐藏冲突。
- 合并要求 `expected = succeeded + failed + unsupported + missing_input + invalid` 的任务集合严格覆盖，无未知/重复键；coverage=100% 指有状态记录完整，不等于 score coverage=100%。失败保留，恢复仅运行缺项/显式指定失败项。
- 每维输出 `timing.json`：视频组定义、base/clip/query/采样帧数、每视频媒体时长、逐 worker 起止/模型装载/推理时间、四卡整体起止、GPU UUID/显存、代码/上游/模型 hash、命令、输出路径；未知媒体时长记 null 并说明，不能用帧数冒充秒数。

## F. 验收、失败分母与报告口径

### F.1 分层验收，不能用 mock 代替模型 parity

| 层级 | 本机/H100 | 必须验证 | 通过后可声称 |
| --- | --- | --- | --- |
| 包与统一接口 | 本机 | 四包直接依赖/独立构建，根 lock，12 个入口 help（7 在办+4 新增+1 legacy），共同 metadata/输出、退出码、both、默认路径和分片故障测试 | 工程接入兼容 |
| 无模型算法/契约 | 本机 | 固定 features/Instances 上的 expected relations；别名/词边界、实例置换、全帧分母、所有失败/不支持分支；未实现后端不能伪成功 | 对声明 fixture/合约成立 |
| Official 接线 | 本机 mock + H100 | spy 确认调用目标 `compute_*`、参数/JSON/源文件 origin 正确；原始返回与规范化返回保持分数和缺失语义 | Official 适配器接线正确；mock 仍不证明数值 parity |
| 真模型 parity | H100，每维 4–8 dev 视频 | 新 wrapper 与独立进程直接调用同一 `compute_*`；同 decode/frames/metadata/checkpoint；逐视频、充分统计量、聚合、状态；一 shard 与四 shard 对齐 | 对此 revision/权重/样本的上游调用 parity |
| 冻结 E0 parity | H100，存在适用基线时 | 找到确实属于该维的 E0 源码/权重/输入身份并逐项比较 | 只有通过才称复现 E0；没有该维基线则记不适用/未验证 |
| 完整反事实与报告 | H100+本机 | Frozen manifest 全覆盖，按族/半边统计及 review；并非要求 Repair 获胜 | 被实际数据支持的局部机制/契约结论 |

初始浮点容差为 `atol=1e-5, rtol=1e-4`，固定 trace 的整数计数、目标串、token IDs、顺序和状态须精确一致。dev pilot 若发现数值非确定性，先报告差异原因、重复运行范围并冻结新容差，不能事后放宽来“通过”。自然集人类对齐是另一层，不能由调用 parity 推出。

语义 same-rank 使用另一个预注册阈值：建议初值 `tau_abs=0.02`（对应各自原始支持率/cosine 单位），至少 90% 有效 bases 的位置/改写组最大 gap ≤该值才称“达到本计划容差”；同时必须报告 margin-free gap 分布、敏感性半边及覆盖率。该阈值须在 dev review 冻结，不能对 test 调大。Repair 的不变性改进以配对 range 缩减的 CI 为证据，不以这个通过比例单独定胜负。

CLI 共同退出行为：参数/输入错误为 2；逐视频失败/不支持/部分完成非零并保存记录；依赖缺失写 blocked/null，both 仍尝试具备条件的另一后端；中断保存已得结果及 remaining/interrupted，退出 130。测试禁止自动 CPU 评分回退、隐式获取权重和网络调用。

### F.2 四维一致的分母和状态策略

**未检出与运行失败不是同一个状态。** 全帧主分母不等于把所有程序异常写成 0：

| 情况 | 分数/证据 | 样本如何保留 |
| --- | --- | --- |
| 推理成功、目标未检出/无匹配颜色证据 | Object/Color 该帧支持值 0，分母仍为全部计划采样帧 T；这是代理证据 0，不宣称对象真实不存在 | `no_detection`、`label_mismatch`、`color_insufficient` 分开计数；整段全零视频仍有记录 |
| 部分帧检测/解码异常 | Repair 主分数 null、failed/partial；可另报固定 T 下支持率上下界 `[observed_hits/T, (observed_hits+unknown_frames)/T]`，不能只平均成功帧 | 原始帧状态、异常数、上下界和缺失都保留。异常率分母为所有计划帧 |
| query 明确超出支持范围 | unsupported、score=null | 进入总请求/不支持率分母；不强制映射最近标签，不算 0 分失败样本 |
| Background mask 不可用/背景为空，Temporal 无法确定 style clause | 对相应 Audit variant abstain/null | Official 仍按自身前置条件尝试；报告该后端 coverage 与原因，不拿整帧/整 prompt 结果冒充修复 |
| Official Object 内部吞异常、Official Color 无返回 | 完整保留 `compute_*` 的 native 返回；wrapper 补齐输入 ledger，标 `native_error_as_empty` 或 `native_omitted`；无法观测内部异常时明确“不透明” | native 分数与用于共同支持比较的有效性标志分列。Color 整批零分母异常保存原异常，不能篡改为官方零分 |
| 缺视频/invalid 反事实 | missing_input/invalid、score=null | 留在 inventory/manifest 和招募/有效率分母；不以重复视频或测后重标替换 |

每个报告同时列 requested、eligible、valid、succeeded、unsupported、model_failed、missing_input、native_omitted 数及各分母；Object/Color 另报帧数、检测器资格通过率、不支持标签率、词法未命中率、视觉漏检率（需人工真值）、条件分母退出率。禁止删除困难样本抬分。

CPA 只在有预定有效关系且双方成功的共同支持 pairs 上比较，同时报告各后端绝对覆盖和共同支持率。额外给 `正确且成功 pairs / 全部预定有效 pairs` 作为覆盖感知成功率下界，明确它**不是**把缺失 score 置零后的 CPA。必须同时给全队列结果与 detector-conditioned 子集，不能仅报告通过 GRiT 资格的子集效果。

### F.3 统计与允许的论文表述

- 按 `dimension × family × split × contract_half × backend × variant` 输出；ordered、same-rank/invariance、实例重放、词法 fixture 不合为一项。不得单独引用 pooled CPA，也不以四维平均掩盖失败族。
- ordered 同时报 zero-margin CPA、dev 冻结 margin 的 CPA、signed gap、合法 ladder 的严格顺序率/Spearman、tie 率；按 base 先平均。paired Δ 使用同一个 base/prompt cluster 的 2,000 次 bootstrap，seed=2026；dev/test CI 分表，不能把全量 CI 标 test。
- invariance 主报绝对 range、相对 range（均值接近 0 时标不稳定）、MAE、必要时归一化对称误差和固定容差内比例；CPA 是附属。Temporal cosine 可负，不直接套 `range/mean`；报 absolute gap 与 dev 固定尺度归一化。缓存记录置换应精确等于，测试通过不充当视觉能力证据。
- Temporal 按 10 个内容组聚类；test 只有 7 组，须展示逐组/逐风格矩阵和 leave-one-group-out 方向稳定性，CI 只作探索性描述，不能把 1,300 queries 当独立样本。Original prompt 的风格名不是自动真值。
- Background 的敏感性和位置不变性两半必须一起读；引用 [Subject review](../counterfactual-reports/subject_consistency.review.md) 和 [P1.5](../counterfactual-reports/P1_NATURAL_AND_CONTROL_RUNS.md) 的限制：median-box 减少施工混淆，但不能证明所有视觉混淆消失；旧 CPA 的 margin/样本集依赖不能迁移成新维已成功。
- Scene、Human Action、Spatial 的既有族设计问题须连同相应 review 读；不把其 pooled CPA 当本计划选族依据。MUSIQ/画质分数不替代编辑有效性审查。
- Repair 改进仅在具体有效族、具体契约半边、相同模型证据及 coverage 条件下陈述；若 paired Δ 的 95% CI 不支持方向，写“未建立改进”。即使契约成立，没有自然偏好非劣证据也不称为 Official 的更好替代。
- 若具备自然标注，另报 zero-margin/tie-aware pair accuracy、Kendall tau-b、coverage；模型级 Pearson `n=4` 仅描述性。替代性主张的非劣 margin 必须在 dev 冻结，推荐先讨论 0.03，不把它当现成通过标准。

`docs/counterfactual-reports/CONSOLIDATED.md` 的 Raw result 仍是正式总表唯一来源。本次不改它。后续四维结果完成 review 后才通过显式 opt-in 集成到原表及 `SUMMARY.md/table2.csv/table2.json/README.md`，保留原脚注；新四维的 staging 报告不能自称另一份权威总表。冻结的 `figures/` 不纳入本计划写入步骤。

## G. 风险与决策边界

| 风险 | 防止误判的执行约束 | 失败后交付 |
| --- | --- | --- |
| 包独立但接口分叉 | 共用 parser/metadata/writer/coordinator，参数化跨包测试，研究驱动复用 batch 合同 | 阻断算法接入，先修工程接口 |
| 直接复制旧包优化路径绕过 `compute_*` | 对实际 Official 调用加 spy/独立进程 parity，不把公式 fixture 当 Official | 标官方接线/数值未验证 |
| detector-conditioned 选择偏置 | 人工先定主队列，资格率/全队列与条件子集分栏，保存全部拒绝原因 | 样本不足/检测器不支持该施工，而非 repair 效果 |
| 施工与评分共用 mask 导致恒等式 | oracle-mask 和自动 mask 两轨，报告区域泄漏；跨位置 equal-area 和转场数诊断 | 合约诊断成立但不宣称端到端背景理解 |
| 风格重叠、内容组仅十个 | 人工 partial-order、内容/风格分开、按内容组统计 | 探索性结果；不宣称普遍风格识别改进 |
| 只修类别名而 caption/视觉证据错误 | 实例 trace + 独立人工错误归因，简单规则先行 | 语义适配器能力边界，不启动未经确定的 LoRA |
| Color heads 数量/顺序假设失效 | 原始 box/head 记录、严格长度检查、一对一绑定歧义 abstain | 报 binding unsupported，而非继续错误归类 |
| 缓存/导入自动联网或错误源 | 导入前本地文件预检，offline 标志与调用检查，固定来源 hash | blocked + 无权重合约结果，无自动补文件 |
| 脏工作树/远端环境与本机不同 | 文件级 source manifest、dirty diff hash、uv/torch/CUDA/扩展构建指纹 | 明示环境偏差；不以 HEAD 相同声称运行版本相同 |
| 旧 pytest 根导入故障 | 暂用 `PYTHONPATH=.`；核对当前工作树已有的 pythonpath 改动与测试 | 保留历史基线，不把旧故障归因于新增包 |

本机 2026-09-15 的历史基线是 `PYTHONPATH=. uv run --no-sync --group test pytest tests` 的 117 passed；不用 `PYTHONPATH` 时四个 `tests/test_counterfactual_*.py` 收集失败。本文未重跑该测试，也不把历史数字称为当前验证结果。**交付前只读复查发现工作树的 `pyproject.toml:60` 已出现 `pythonpath = ["."]`，这是规划期间工作区其他改动，本任务没有写入它，也未验证其测试结果。** 后续不要重复添加；核对是否保留并验收即可。

## H. 里程碑、命令与顺序

### H.0 命令可用性说明

M0 的内联命令使用现有 Python/文件，今天即可运行且不需要权重。M1 的 build/help/pytest 是**对应文件实现后**的验收命令；它们不会自动生成四个包。`scripts.four_dimension` 是 M2 起逐步交付的后续工具，不是当前仓库已经支持的接口。H.3–H.5 的命令必须在该工具、对应 metric 后端及人工审查文件就绪后执行；缺失即明确阻断，不能略过检查。

后续实施者先在本机设置（本文编写阶段不执行生成命令）：

```bash
cd /home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit
export FD_REPO="$PWD"
export FD_RUN="$FD_REPO/output/four_dimension_20260920"
export FD_UPSTREAM="$(realpath ../VBench)"
export FD_PY="$FD_REPO/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VBENCH_AUDIT_UPSTREAM="$FD_UPSTREAM"
export PYTHONPATH="$FD_REPO/packages/audit-core/src:$FD_REPO/packages/audit-models/src:$FD_REPO"
```

### H.1 M0：零成本离线证据优先（本机，0 GPU·h）

输入：锁定上游、12 pins、原始 suite。本次已完成只读 pin/行号/元数据复核；下面命令将其生成可留档的机器证据。预计 CPU 1–5 min，人工核对 0.5–1 h。

```bash
mkdir -p "$FD_RUN/offline"
"$FD_PY" -B - <<'PY' > "$FD_RUN/offline/source-evidence.json"
import ast, dataclasses, json, os
from collections import Counter
from pathlib import Path
from vbench_audit_core.upstream import verify_upstream

up = Path(os.environ['FD_UPSTREAM'])
state = verify_upstream(up)
rows = json.loads((up / 'vbench/VBench_full_info.json').read_text())
dims = ('background_consistency', 'scene', 'temporal_style', 'object_class', 'color')
suite = {d: [r for r in rows if d in r['dimension']] for d in dims}
prompts = {d: {r['prompt_en'] for r in suite[d]} for d in dims}
parts = [r['prompt_en'].rsplit(', ', 1) for r in suite['temporal_style']]
left = (up / 'vbench/temporal_style.py').read_text().replace('temporal_style', 'overall_consistency')
right = (up / 'vbench/overall_consistency.py').read_text()
def checker(filename):
    source = up / 'vbench' / filename
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_generate')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['check_generate']
oc, color = checker('object_class.py'), checker('color.py')
classes = ['car', 'bicycle']
caps = [('a red car', [0, 0, 10, 10], classes), ('a blue bicycle', [20, 0, 30, 10], classes)]
legacy = [[(c[0], c[2][0]) for c in caps]]
indexed = [[(c[0], c[2][i]) for i, c in enumerate(caps)]]
result = {
    'upstream': dataclasses.asdict(state),
    'counts': {d: len(suite[d]) for d in dims},
    'background_scene_prompt_equal': prompts['scene'] == prompts['background_consistency'],
    'temporal_normalized_AST_equal': ast.dump(ast.parse(left)) == ast.dump(ast.parse(right)),
    'temporal_base_count': len({p[0] for p in parts}),
    'temporal_styles': dict(Counter(p[1] for p in parts)),
    'temporal_aux_count': sum('temporal_style' in r.get('auxiliary_info', {}) for r in suite['temporal_style']),
    'temporal_max_words': max(len(r['prompt_en'].split()) for r in suite['temporal_style']),
    'object_rule_fixture_couch_sofa': [oc(t, [{'sofa'}]) for t in ('couch', 'sofa')],
    'color_red_substring_fixture': color('red', 'car', [[('a colored car', 'car')]]),
    'color_blue_car_legacy_indexed': [color('blue', 'car', p) for p in (legacy, indexed)],
    'color_blue_bicycle_legacy_indexed': [color('blue', 'bicycle', p) for p in (legacy, indexed)],
    'scope': 'source and synthetic rule witnesses; no model measurements'
}
assert result['counts'] == dict(zip(dims, (86, 86, 100, 79, 85)))
assert result['background_scene_prompt_equal'] and result['temporal_normalized_AST_equal']
assert result['temporal_base_count'] == 10 and result['temporal_aux_count'] == 0
assert result['object_rule_fixture_couch_sofa'] == [0, 1]
assert result['color_red_substring_fixture'] == (1, 1)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
```

验收：12 pin 全通过、86 同源/100=10×10/79/85 等统计匹配；同义目标在冻结 fixture 上为 `[0,1]`；Color 原规则把 `colored` 当 red，索引更正会消除“blue car”错绑并保留 blue bicycle。**这些是源码规则反例，既不是检测器预测也不是 repair 实测收益。**

精确 token 审计无需 GPU/ViCLIP 权重，但需要已存在的匹配 BPE 和依赖。推荐在 H100 的既有 venv 执行下面命令；若 BPE 缺失，输出 blocked 文件并结束，不 import tokenizer：

```bash
"$FD_PY" -B - <<'PY' > "$FD_RUN/offline/temporal-tokens.json"
import hashlib, json, os, sys
from pathlib import Path
up = Path(os.environ['FD_UPSTREAM'])
cache = Path(os.environ.get('VBENCH_CACHE_DIR', str(Path.home() / '.cache/vbench')))
bpe = cache / 'ViCLIP/bpe_simple_vocab_16e6.txt.gz'
if not bpe.is_file():
    print(json.dumps({'status': 'blocked', 'reason': 'local BPE missing', 'path': str(bpe)}))
    raise SystemExit(2)
sys.path.insert(0, str(up))
from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
tok = SimpleTokenizer(str(bpe))
sot, eot = tok.encoder['<|startoftext|>'], tok.encoder['<|endoftext|>']
def tokens(text):
    full = [sot] + tok.encode(text) + [eot]
    kept = full[:] if len(full) <= 32 else full[:31] + [eot]
    return full, kept
rows = json.loads((up / 'vbench/VBench_full_info.json').read_text())
records = []
for row in rows:
    if 'temporal_style' not in row['dimension']:
        continue
    prompt = row['prompt_en']
    content, style = prompt.rsplit(', ', 1)
    full, kept = tokens(prompt)
    sf, sk = tokens(style)
    a = tokens((content + ' ') * 40 + ', pan left')[1]
    b = tokens((content + ' ') * 40 + ', pan right')[1]
    records.append({'prompt': prompt, 'style': style, 'full_tokens': full,
                    'kept_tokens': kept, 'truncated': len(full) > 32,
                    'style_only_length': len(sf), 'style_only_truncated': len(sf) > 32,
                    'long_prefix_collision': a == b})
print(json.dumps({'status': 'succeeded', 'bpe_sha256': hashlib.sha256(bpe.read_bytes()).hexdigest(),
                  'records': records}, ensure_ascii=False, indent=2))
PY
```

产物 `$FD_RUN/offline/temporal-tokens.json`，CPU 1–3 min；长前缀的相反风格 token 碰撞是估计器不可辨的充分证据。自然 suite 的丢失 spans/碰撞分组由后续 `offline` 报告读取 full/kept IDs 生成；只统计 truncated 条数不能称为“全部风格不可见”。该命令不计算视频分数；若依赖缺失同样标未验证。

### H.2 M1：先完成四包与公共接口（本机）

输入：C 节合同、现有 core、独立子项目模板、当前工作树。实施四个骨架、根 workspace/lock 注册、共同参数化合约测试和模型配置读取合同；使用注入 mock 验证共用 batch/输出，真实 scorer 后续逐维接通。**M1 可独立评审交付，不等待研究驱动、标注、反事实数据和权重。** 预计实施 1–2 人日；CPU 验收约 0.5–1 h；0 GPU·h。

```bash
# 实施文件并由 uv 更新锁之后执行；不手写 uv.lock。
uv lock
uv lock --check
uv sync --locked --group test
uv pip install --python .venv --index https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
PYTHONPATH=. uv run --no-sync --group test pytest tests metrics
for metric in background-consistency temporal-style object-class color; do
  uv run --no-sync --package "$metric" "$metric" --help
  uv build --package "$metric" --out-dir "$FD_RUN/dist"
done
for metric in dynamic-degree motion-smoothness subject-consistency scene human-action spatial-relationship multiple-objects overall-consistency; do
  uv run --no-sync --package "$metric" "$metric" --help
done
uv build --package audit-core --out-dir "$FD_RUN/dist"
uv build --package audit-models --out-dir "$FD_RUN/dist"
for metric in background-consistency temporal-style object-class color; do
  uv venv --python 3.11.14 "$FD_RUN/isolated/$metric"
  uv pip install --python "$FD_RUN/isolated/$metric/bin/python" \
    "$FD_RUN"/dist/audit_core-*.whl "$FD_RUN"/dist/audit_models-*.whl \
    "$FD_RUN/dist/${metric//-/_}"-*.whl
  "$FD_RUN/isolated/$metric/bin/$metric" --help
done
git diff --check
```

CPU torch overlay 沿用仓库 README 的版本；这是测试依赖，不是模型权重。overlay 之后不再 sync models extra。需要隔离验证时为单包建立新的测试环境，安装本地 core/models 与该 metric 的 wheels；检查无需安装其余三个 metric 即可 import/help/跑该包纯测试。

产物：C.1 四包结构、root 注册/lock、`tests/test_four_dimension_cli_contracts.py`、`tests/test_four_dimension_runner_contracts.py`、`tests/test_four_dimension_workspace.py`、`configs/four_dimension/models.h100.toml` 配置示例、`$FD_RUN/dist/`。验收以接口真实兼容为准；未实现算法明确 null，不要求四个 dummy 分数通过。

### H.3 M2：H100 资产核验与运行时门禁

输入：M1 的可审阅实现/模型配置、H100 既有 checkout/venv/视频/缓存。先在本机实现 `scripts/four_dimension.py` 的 offline/inventory 命令并准备 `configs/four_dimension/protocol.toml`，预计约 0.5–1 人日；再用文件 manifest 核对 H100 代码版本，不同步/覆盖上游或冻结输入。资产核验预计 CPU 15–30 min、设备探测 <0.1 GPU·h；人工缺额核对 0.5 h。

在 H100 已就绪的 audit 工作区执行：

```bash
cd /root/wenbiao_zhao/vbench-audit
export FD_REPO="$PWD"
export FD_RUN="$FD_REPO/output/four_dimension_20260920"
export FD_UPSTREAM=/root/wenbiao_zhao/VBench
export FD_PY=/root/wenbiao_zhao/venvs/vbench/bin/python
export FD_DATA=/root/wenbiao_zhao/datasets/vbench-1.0-human-preference
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VBENCH_CACHE_DIR=/root/.cache/vbench
export VBENCH_AUDIT_UPSTREAM="$FD_UPSTREAM"
export PYTHONPATH="$FD_REPO/packages/audit-core/src:$FD_REPO/packages/audit-models/src:$FD_REPO"
mkdir -p "$FD_RUN/offline" "$FD_RUN/inventory"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.free --format=csv
uv pip freeze --python "$FD_PY" > "$FD_RUN/inventory/runtime-before.txt"
# 只注册本次本地包，不替换既有 CUDA/GRiT 依赖。
uv pip install --python "$FD_PY" --no-deps \
  -e packages/audit-core -e packages/audit-models \
  -e metrics/background-consistency -e metrics/temporal-style \
  -e metrics/object-class -e metrics/color
uv pip check --python "$FD_PY"
fd() {
  "$FD_PY" -B -m scripts.four_dimension "$@" \
    --root "$FD_RUN" --upstream "$FD_UPSTREAM" \
    --protocol "$FD_REPO/configs/four_dimension/protocol.toml" \
    --models "$FD_REPO/configs/four_dimension/models.h100.toml"
}
fd inventory --dataset-root "$FD_DATA" --gpus 1,2,3,4
fd offline
```

`inventory` 在模型 import 前核验文件，对应维缺资产时记录 blocked；验证 GPU UUID、各进程 `device_count=1`、不运行真实模型。执行 H.1 token 块时继承此处变量。若 `uv pip check` 暴露依赖缺口，先报告明确的 CUDA/依赖偏差，不粗暴重装 H100 环境。

产物 `$FD_RUN/inventory/{assets,source-videos,source-code,gpus,environment}.json`、`$FD_RUN/offline/<dimension>.json`。四份离线报告分别列：Background 的 suite 耦合；Temporal 的同估计器、子句交换/token 截断；Object 的 79 目标标签改写 fixture；Color 的实例索引、词边界及固定分母 fixture。验收要求每项都有 `verified/blocked/not_applicable`，没有推理则模型统计字段为 null。

### H.4 M3–M6：逐维接算法、冻结契约、运行并合并

每维都先在本机实现该包的 Official/audit/repair、受影响的共享模型适配与对应纯测试，再在 H100 执行该维。**四包接口统一先行，四维真实模型评分严格串行。** 每维若只需 metric 内算法变更，运行该包的纯算法/合约测试；若动共同接口/依赖，重新执行 M1 全套检查。

本机逐维验收命令（将包名分别取四个 C.1 名称）：

```bash
export FD_METRIC=background-consistency
PYTHONPATH=. uv run --no-sync --group test pytest "metrics/$FD_METRIC/tests" \
  tests/test_four_dimension_cli_contracts.py tests/test_four_dimension_runner_contracts.py
uv run --no-sync --package "$FD_METRIC" "$FD_METRIC" --help
```

研究驱动的资格/构造阶段必须先输出候选 ledger、dev/test 分组、逐 base 参数及待人工标注清单，再接受真实的人工标注文件。`prepare --split dev` 冻结两组 source IDs，只构造 dev 视频；不得在看到 dev/test 得分后重新随机挑 test。`--resume` 仅复用 hash 完全一致的构造。`validate` 接收审查文件并严格校验重放、泄漏和预期关系；空白标注、缺失审查不可当通过。

在 H100 继承 H.3 的 `fd` 后定义通用命令。以下函数只封装**同一个驱动接口**；没有各维私有 worker 输入/输出格式：

```bash
set -euo pipefail
fd_dev() {
  local dimension="$1"
  fd prepare --dimension "$dimension" --split dev --dataset-root "$FD_DATA" \
    --gpus 1,2,3,4 --resume
  fd validate --dimension "$dimension" --split dev --replay \
    --validity "$FD_RUN/validity/$dimension.dev.jsonl"
  fd parity --dimension "$dimension" --split dev --limit 8 \
    --gpus 1,2,3,4 --direct-compute --compare-shards 1,4
  fd score --dimension "$dimension" --split dev --variants configured \
    --gpus 1,2,3,4 --resume
  fd merge --dimension "$dimension" --split dev --strict
  fd report --dimension "$dimension" --split dev --iterations 2000 --seed 2026
}
fd_test_and_close() {
  local dimension="$1"
  fd freeze --dimension "$dimension" \
    --review "$FD_RUN/reviews/$dimension.dev.json"
  fd prepare --dimension "$dimension" --split test --dataset-root "$FD_DATA" \
    --frozen "$FD_RUN/protocol/$dimension/frozen.json" --gpus 1,2,3,4 --resume
  fd validate --dimension "$dimension" --split test --replay \
    --validity "$FD_RUN/validity/$dimension.test.jsonl"
  fd score --dimension "$dimension" --split test --variants configured \
    --frozen "$FD_RUN/protocol/$dimension/frozen.json" --gpus 1,2,3,4 --resume
  fd merge --dimension "$dimension" --split test --strict
  fd report --dimension "$dimension" --split test --iterations 2000 --seed 2026
  fd validate --dimension "$dimension" --phase close \
    --review "$FD_RUN/reviews/$dimension.test.json"
}
```

`validity/*.jsonl` 与 `reviews/*.json` 由具名审查者填写，驱动只生成待填清单，不伪造人工答案；命令遇缺失就停在该步，可待文件补齐后从该步继续。`freeze` 记录规则表、变换、mask/检测资格、margin、主分数、消融、人工协议和所有代码/config hash。test 后禁止再调这些参数。

`score` 收集 metric CLI 的非零退出与逐项状态；研究命令成功只表示完整任务 ledger 已落盘，不表示所有模型成功。明确记录的 unsupported/失败可进入覆盖率报告，worker 无返回/指纹冲突则禁止 merge。`validate --phase close` 要求 expected set 完整、统计可重算、review 给出允许/撤回的命题，不要求 Repair 获胜。

| 里程碑/执行者 | 输入与具体命令（前一步 close 后才执行下一维） | 预期产物 | 验收/预计机时 |
| --- | --- | --- | --- |
| **M3 Background**；本机实施，H100 测量 | M1/M2 + 官方 Scene 视频 + 合格区域；先 `fd_dev background_consistency`，dev review 后 `fd_test_and_close background_consistency` | `manifests/background_consistency/`、`evidence/background_consistency/`、`scores/background_consistency/`、`reports/background_consistency/`、`protocol/background_consistency/frozen.json` | 25 base 目标及缺额；3 族、敏感/不变两半、端点转场诊断、oracle/自动 mask 分栏；4–8 GPU·h，四卡 1–2 h，CPU 构造 0.5–1 h；实施约 1–2 人日 |
| **M4 Temporal**；本机实施，H100 测量 | M3 close + 100 prompt/目标视频/BPE/ViCLIP + style 人工标注；`fd_dev temporal_style` → review → `fd_test_and_close temporal_style` | 同上以 `temporal_style` 命名；附 `tokens.jsonl`、`style_matrix.csv`、`content_style_scores.csv` | 10 内容组不泄漏；自然/长前缀挑战分栏；style-only 无截断，矛盾/共存风格分开；8–16 GPU·h，四卡 2–4 h，CPU 0.25–0.5 h；实施约 1–2 人日 |
| **M5 Object**；本机实施，H100 测量 | M4 close + Object 视频/79 标签/GRiT/别名审核；`fd_dev object_class` → review → `fd_test_and_close object_class` | 同上以 `object_class` 命名；附 `label_drift.csv`、逐实例 `instances.jsonl`、支持目录快照 | 25 base 目标；原始实例不丢失；词法/视觉/运行失败分开，全帧分母；8–16 GPU·h，四卡 2–4 h，CPU 0.25–0.5 h；实施约 1 人日 |
| **M6 Color**；本机实施，H100 测量 | M5 close + Color 视频/双实例资格/同一 GRiT checkpoint；`fd_dev color` → review → `fd_test_and_close color` | 同上以 `color` 命名；附 `binding_diagnostics.csv`、`denominator_audit.csv`、逐实例/逐帧 trace | 25 主 bases + 15 双实例目标及缺额；binding/lexical/denominator 消融分开；丢弃视频全部可追踪；16–32 GPU·h，四卡 4–8 h，CPU 编码 1–2 h；实施约 1–2 人日 |

上表路径均相对 `$FD_RUN`。人工标注/审查约另需 1–3 人日，不能折成 GPU 时间；候选实现合计约 6–10 人日，顺序闸门和资产缺失可能延长日历时间。

接口的正式 smoke 可在 `prepare` 生成共同输入视图后直接运行四包，无需学习研究驱动的内部数据格式：

```bash
CUDA_VISIBLE_DEVICES=1 "$FD_PY" -B -m background_consistency.cli --both \
  --video-dir "$FD_RUN/views/background_consistency/smoke" --gpu 0 \
  --metadata "$FD_RUN/views/background_consistency/smoke/metadata.json" \
  --model-config "$FD_REPO/configs/four_dimension/models.h100.toml" \
  --output "$FD_RUN/acceptance"
CUDA_VISIBLE_DEVICES=1 "$FD_PY" -B -m temporal_style.cli --both \
  --video-dir "$FD_RUN/views/temporal_style/smoke" --gpu 0 \
  --metadata "$FD_RUN/views/temporal_style/smoke/metadata.json" \
  --model-config "$FD_REPO/configs/four_dimension/models.h100.toml" \
  --output "$FD_RUN/acceptance"
CUDA_VISIBLE_DEVICES=1 "$FD_PY" -B -m object_class.cli --both \
  --video-dir "$FD_RUN/views/object_class/smoke" --gpu 0 \
  --metadata "$FD_RUN/views/object_class/smoke/metadata.json" \
  --model-config "$FD_REPO/configs/four_dimension/models.h100.toml" \
  --output "$FD_RUN/acceptance"
CUDA_VISIBLE_DEVICES=1 "$FD_PY" -B -m color.cli --both \
  --video-dir "$FD_RUN/views/color/smoke" --gpu 0 \
  --metadata "$FD_RUN/views/color/smoke/metadata.json" \
  --model-config "$FD_REPO/configs/four_dimension/models.h100.toml" \
  --output "$FD_RUN/acceptance"
```

上面四条分别安排在该维 milestone 内，不作为跨维抢跑脚本。`--model-config` 是四包统一追加的同名可选参数，读取配置中本维需要的资产；必须在 M1 参数合同中实现。H100 真 smoke 不以 mock 覆盖，无权重时只由本机合约测试注入 mock。

### H.5 M7：报告复算与可引用交付（本机 CPU/H100 CPU）

输入：四维已 close 的 manifest、scores、provenance、validity、review。预计 CPU 0.5–1 h、人工核对 0.5–1 人日，0 GPU·h。

```bash
for dimension in background_consistency temporal_style object_class color; do
  fd merge --dimension "$dimension" --split test --strict
  fd report --dimension "$dimension" --split test --iterations 2000 --seed 2026
  fd validate --dimension "$dimension" --phase close \
    --review "$FD_RUN/reviews/$dimension.test.json"
done
```

产物为 `$FD_RUN/reports/<dimension>/{report.md,statistics.json,statistics.csv,coverage.csv,timing.json,review.md}` 和 `$FD_RUN/reports/claim-ledger.json`。每个 claim 指向族、半边、split、有效 n、CI、覆盖率、代码/模型身份与 review，未知项写 `--`/null。报告不能要求读者从某个 pooled CPA 自行推断风险。

后续正式文档集成另作为明确开关：拟定 `fd report --integrate-reviewed --review-index "$FD_RUN/reviews/index.json"` 只接受四维审查索引中明确准许的行，将已审报告放入 `docs/counterfactual-reports/<dimension>.md` 与 `<dimension>.review.md`，更新唯一 `CONSOLIDATED.md` 及同步派生表/README，保留历史脚注。此模式必须先生成 diff 供审阅，再由正式实施任务授权应用；当前规划任务不执行、不触碰历史表，更不写 `figures/`。

## I. 交付物清单与生成/验证命令

| 交付物路径 | 生成或验收方式 | 完成阶段 |
| --- | --- | --- |
| `docs/plans/2026-09-20-four-dimension-kickoff-plan.md` | 人工规划；检查章节、链接、命令、边界及逐维覆盖 | 本次唯一新增交付 |
| `metrics/{background-consistency,temporal-style,object-class,color}/pyproject.toml` 与各 `src/<snake>/`、`tests/` | 人工实施；M1 的 `uv build --package ...`、`uv run --package ... --help`、逐包 pytest 验收 | M1 骨架，M3–M6 算法 |
| 根 `pyproject.toml`、`uv.lock`；共享 core/models 的必要兼容改动 | 声明四个 workspace 项后 `uv lock`；`uv lock --check`、sync、跨包/独立构建测试 | M1，后续接口改动再验收 |
| `tests/test_four_dimension_{cli_contracts,runner_contracts,workspace}.py` | 人工实现合同；`PYTHONPATH=. uv run --no-sync --group test pytest tests metrics` | M1 |
| `configs/four_dimension/{protocol,models.h100}.toml` | 人工编写可审阅配置；`fd inventory`/`fd freeze` 校验；不放权重/公式 | M1/M2 |
| `scripts/four_dimension.py` | 人工实现薄驱动；`"$FD_PY" -m scripts.four_dimension --help` 与 fixture 测试；调用包的公共架构 | M2 至 M6，不阻塞 M1 |
| `$FD_RUN/offline/{source-evidence,temporal-tokens}.json`、`<dimension>.json` | H.1 内联命令、`fd offline` | M0/M2 |
| `$FD_RUN/inventory/` | `fd inventory --dataset-root "$FD_DATA" --gpus 1,2,3,4` | M2 |
| `$FD_RUN/manifests/<dimension>/`、`clips/`、`views/`、`evidence/` | `fd prepare --dimension ... --split dev/test ...`；`fd validate --replay ...` | M3–M6 |
| `$FD_RUN/validity/`、`reviews/`、`protocol/<dimension>/frozen.json` | 具名人工审核填表；`fd freeze --dimension ... --review ...` 验签名/字段/hash 并冻结 | M3–M6 |
| `$FD_RUN/cli/`、`scores/`、`parity/<dimension>/` | metric 共同 CLI、`fd parity`、`fd score`、`fd merge --strict` | M3–M6 |
| `$FD_RUN/reports/<dimension>/` 与 `claim-ledger.json` | `fd report ... --iterations 2000 --seed 2026` | M7 |
| 各 `metrics/<kebab>/IMPLEMENTATION_REPORT.md`；后续 `docs/cli.md`、`docs/upstream-mapping.md`、`docs/semantic-adapter-constraints.md` 的必要补充 | 人工据真实运行记录更新；记录接口已测/真实模型未测/已测范围，不复制虚构通过数 | M1、每维 close |
| 正式独立报告与唯一总表的已审新增行 | 后续显式 `fd report --integrate-reviewed --review-index ...`，先输出文档 diff；不改冻结输入或 figures | M7 后条件交付，本次不执行 |

### I.1 提交计划前自检

本次实际验证：12 pin 与上游 clean 校验；文档内 M0 无模型源码/规则反例在 stdout 成功运行；shell 命令块通过 `bash -n`，内联 Python/JSON 语法和文档相对链接检查通过。未执行后续 build/安装/模型/实验命令，未产出 GPU 数值。

- [x] 四维均有命题/可证伪条件、反事实族、离线与 GPU 两级证据、repair 职责、模型/降级、共享引用边界和失败/不支持率口径。
- [x] 四包各自 `pyproject.toml` + src/tests；由同一 uv workspace/lock 管理，不能互相 import；共用评测输入输出架构及接口合约优先于算法。
- [x] 每族列出关系、目标 base 数、来源/派生需求、资格及判据；纯 fixture 和 GPU 结果不混报。
- [x] 无下载权重、改上游、改冻结目录、删除 legacy、commit/push 的执行步骤。
- [x] 本机只承诺纯算法/合约/离线证据；H100 模型资产、实际影响与 parity 未验证部分逐项标明。
- [x] 里程碑列出输入、命令前置条件、输出、判据、执行机器和预计机时；待实现命令没有冒充当前能力。
- [x] 不单引 pooled CPA；Background 两半、Temporal 内容组/风格关系、Object/Color 失败率与各自 review 必须伴随结论引用。

### 需要用户拍板的问题（不阻塞本次计划交付）

1. **首轮计算预算**：推荐预留 36–72 GPU·h，四卡约 9–18 h，另留约 20% 恢复余量；先以各维 pilot 修正估算，不把该预算用于 M1 包接入验收。
2. **GPU 顺序及缺资产处理**：推荐 Background → Temporal → Object → Color；某维 blocked 先交缺额记录，若要调整次序再明确决定，不自动跨维启动。
3. **Background 的 Full 候选**：推荐 dev 同时评估 `masked_prev_mean`/`masked_prev_worst`，固定 mask 条件，选择后冻结；保留 oracle/自动 mask 的结论边界。
4. **Temporal 的有限样本结论**：推荐接受 10 内容组的探索性审计与 partial-order，暂不增加新生成视频；没有独立自然标注时不作替代性主张。
5. **Color 双实例队列不足时**：推荐先报告真实视频缺额，允许另列确定性并置队列作机制补充；不将 composite 结果混称自然视频分布效果。
6. **语义适配器是否启动训练**：推荐首轮只做确定性词表/规范化及最小 JSON；Object/Color 的 LoRA 纳入六维方案另行定模型/数据/预算，不作为四包接口前置条件。
7. **pytest 根导入修复**：交付前工作树已存在 `pythonpath = ["."]`；推荐保留并在后续回归验证，不重复添加。本计划命令继续显式 `PYTHONPATH=.`，本次不改该文件。
8. **首轮验收是否要求自然偏好非劣**：推荐首轮验收工程兼容、真实调用 parity、反事实及失败报告；自然标签 inventory 齐备后再定独立扩展预算和非劣 margin，不承诺“Repair 更好”。
