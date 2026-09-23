# VBench 各维度的原始评分、机制分析与 Repair 方法

本文档给出本文实验所覆盖维度的论文风格方法说明。除特别说明外，`Origin` 指 VBench 原始评分路径，`Repair` 指当前仓库中用于正式核验的最新路径。Repair 不必然使分数上升；它的目标是让分数对目标语义中应当敏感的因素敏感、对不应影响目标的因素保持稳定。所有数字均以对应实验报告和实际输出为准，未运行或尚未完成的项目明确标注为 `NOT RUN` 或“未完成”。

本文当前范围包括 Dynamic Degree、Motion Smoothness、Subject Consistency、Scene、Human Action、Spatial Relationship、Multiple Objects、Background Consistency、Object Class、Color Consistency 和 Temporal Style。Overall Consistency 属于旧的总体指标整理，不作为本稿的独立实验维度。

## 1. Dynamic Degree

### Original Scoring

本章节暂空。

### Mechanism Analysis

本稿暂不引用旧版本 Dynamic Degree 的实现或结论；当前默认实现、问题路径及其与正式实验的对应关系尚未完成最终核验。

### Repair Strategy

本章节暂空。当前状态为 `NOT RUN`，不将历史聚合消融或旧版实现写作当前 Repair。

### Experimental Evidence

正式反事实结果、候选数、评分数、置信区间和人类偏好实验均待补齐。

## 2. Motion Smoothness

### Original Scoring

本章节暂空。

### Mechanism Analysis

本稿暂不引用旧版本 Motion Smoothness 的实现或结论；当前默认实现、问题路径及其与正式实验的对应关系尚未完成最终核验。

### Repair Strategy

本章节暂空。当前状态为 `NOT RUN`，不将历史 temporal aggregation 消融或旧版实现写作当前 Repair。

### Experimental Evidence

正式反事实结果、候选数、评分数、置信区间和人类偏好实验均待补齐。

## 3. Subject Consistency

### Original Scoring

VBench 的 Subject Consistency 使用 DINO ViT-B/16 提取整帧视觉特征。对视频解码得到的全部帧应用官方 `dino_transform(224)`，对特征作 L2 归一化，然后同时计算相邻帧相似度与首帧锚定相似度，并以相等权重合成为视频分数：

\[
S_{\mathrm{origin}}=
\frac{1}{2}\operatorname{mean}_{t>0}\cos(f_t,f_{t-1})+
\frac{1}{2}\operatorname{mean}_{t>0}\cos(f_t,f_0).
\]

其中首帧 \(f_0\) 在第二项中被固定为参考。数据集级汇总沿用官方实现的帧/视频聚合逻辑；单 GPU 和多 GPU 汇总时需要注意 transition-weighted 与 video-average 的差异。

其数据流为：

```text
视频 → 全部解码帧 → whole-frame DINO ViT-B/16 →
相邻帧余弦相似度 + 首帧锚定余弦相似度 → Subject score
```

### Mechanism Analysis

该评分并未显式隔离主体。因而背景纹理、相机运动、主体尺度和主体位置变化都可能改变 whole-frame DINO 表征，即使主体本身保持不变，分数也可能下降。相反，背景相似也可能掩盖主体变化。首帧锚定项还使结果依赖于首帧内容和首帧中的主体状态，而不是只依赖主体在时间上的稳定性。

### Repair Strategy

当前正式 Repair 是 Subject Stability v5。构造端使用 SegFormer-B0 与 GrabCut 生成主体保留、背景高斯模糊的反事实视频；构造端的 mask 不直接作为评分证据。评分端对 clean 与 counterfactual 两个实际视频分别运行独立的 Mask R-CNN 与 MobileSAM subject localizer，不读取构造 mask，也不复用 clean 视频的检测框或特征。对每一帧，将主体区域保留、主体外区域填为常数值 128，并按带 10% margin 的正方形主体区域裁剪到 224；随后提取 DINO patch-level 表征并进行主体区域聚合。正式稳定性汇总采用 all-pairs subject evidence，缺失主体证据按主实验规则排除。

其数据流为：

```text
视频 → 独立 Mask R-CNN / MobileSAM 定位主体 → 主体隔离与正方形裁剪 →
DINO patch 表征 → all-pairs subject evidence → Subject Repair score
```

因此，当前 Repair 的核心不是单独替换一个 temporal aggregation 公式，而是将视觉证据从整帧迁移到每个视频各自检测得到的主体区域。仅使用 all-pairs 聚合而不隔离主体的路径属于旧版消融，不是当前 v5 主方法。

### Experimental Evidence

Subject Stability v5 共构造 720 个候选视频（72 个 prompt family、10 个候选，4 个生成器各 180 个）。其中 241 条通过构造并完成主评分，479 条被构造质量检查拒收，未发生运行时失败；已知 truck 排除后，主分析计划为 240 条。补充的 single-frame 评分在 241 条中完成 204 条后停止，属于部分输出，不能称为全量完成。

在开头四分之一背景模糊反事实上，Origin 从 `0.936417` 降至 `0.816382`，平均绝对变化为 `0.120035`，95% CI 为 `[0.106987, 0.134431]`；Repair v5 从 `0.951413` 降至 `0.941724`，平均绝对变化为 `0.010596`，95% CI 为 `[0.008069, 0.013583]`。对应主体干预的 Repair 平均降分为 `0.099319`，95% CI 为 `[0.093839, 0.104371]`。240 条主分析中有 180 条满足 `≤0.01` 的单条稳定性阈值，但严格门槛整体未通过，联合门槛仅有 70/240 条通过。因此可以支持“Repair 显著减弱背景干扰，同时仍对主体变化保持响应”，不能宣称已经满足严格的全量不变性验收。

## 4. Scene

### Original Scoring

Scene 维度旨在判断生成视频是否呈现了 prompt 所描述的场景。VBench 首先使用 Tag2Text 为采样帧生成 caption，随后提取 prompt 中与场景相关的关键词，并采用严格的词面字符串匹配进行判断：只有当目标 scene words 均直接出现在预测 caption 中时，该帧才被判定为匹配成功。进而计算成功匹配的帧比例得到 Scene score。当前实现保持 16 个中点采样帧和同一份 Tag2Text caption cache。

其数据流为：

```text
视频 → 16 个采样帧 → Tag2Text caption → prompt 场景词提取 →
严格 substring / token 匹配 → 帧级二值结果 → 16 帧平均
```

### Mechanism Analysis

这种基于字符串匹配的评分方式，将“场景语义是否一致”简化为“指定词是否出现”。因此，语义等价但词面不同的描述可能得到不同分数；同时，词面碰撞、否定表达和 caption 中偶然出现的目标词也可能产生错误支持。该问题发生在文本目标与视觉 caption 的接口，而不是帧采样本身，因此不能通过改变采样帧来解释或修复。

### Repair Strategy

我们保留原始的 16 帧采样和 Tag2Text 视觉证据，仅重新设计文本目标与视觉描述之间的匹配方式。具体而言，我们使用独立训练的 Qwen3-8B Scene Verifier，同时输入原始 prompt \(p\) 与对应帧的 caption \(c_t\)，并输出三类语义判断：

\[
\{\text{supported},\ \text{contradicted},\ \text{insufficient}\}.
\]

其中，`supported` 表示当前 caption 在语义上支持目标场景，`contradicted` 表示视觉描述与目标场景冲突，`insufficient` 表示当前 caption 无法提供足够的场景证据。评分时，仅将 `supported` 映射为 1，其余两类记为 0。batch 推理按 `(prompt, caption)` 去重，但同一视频的 16 条 caption 仍分别参与帧级评分。

其数据流为：

```text
视频 → 16 个采样帧 → Tag2Text caption →
Qwen3-8B Scene Verifier(prompt, caption) →
supported=1 / 其余=0 → 16 帧平均
```

当前正式 Repair 是语义 verifier；封闭 synonym dictionary 仅作为独立的 deterministic rule control，不等同于 Repair-model。当前路径没有 SAM、ROI、segmentation、OpenCLIP 或四象限聚合。

### Experimental Evidence

当前矩阵包含 1040/1040 个完成条目、52 个 family。主要 synonym 实验包含 200 个视频、10 个 family，并对所有方案复用同一份 caption cache。Origin 从 base 到 synonym 的均值为 `0.30594 → 0.05406`，变化为 `−0.251875`，95% CI 为 `[−0.4409375, −0.0450000]`；Repair-model 为 `0.6909375 → 0.6946875`，变化为 `+0.003750`，95% CI 为 `[−0.011875, +0.021875]`。这支持“Repair 对同义替换基本保持稳定，而 Origin 对词面替换敏感”。

在 3200 个 caption-pair 证据上，Origin 的 correct-and-consistent 比例为 `0.26719`，rule control 为 `0.62438`，Repair-model 为 `0.92125`，95% CI 为 `[0.88438, 0.95563]`；模型相对 Origin 的配对增益为 `+0.65406`，95% CI 为 `[0.57281, 0.73375]`。需要限定的是，这些证据依赖 Tag2Text caption，且 verifier 的判断不是独立人工视觉真值；ocean→sea 仅为 dev-only 的 20 个视频/1 个 family，不能替代完整测试集结论。

## 5. Human Action

### Original Scoring

Human Action 使用官方 action target 与 UMT ViT-L/16 Kinetics-400 top-5 预测 cache。对每个视频，UMT 输出类别及其 sigmoid 分数，分数按官方实现保留四位小数；当目标 action 与 top-5 类别完全匹配且满足阈值 `0.85` 时，帧/视频级结果记为通过，最后进行视频级和数据集级平均。当前 target 来自冻结的 metadata 与 prompt compiler，而不是只根据视频文件名推断。

其数据流为：

```text
prompt → action target → UMT ViT-L/16 top-5 cache →
阈值与 exact-match 判定 → 视频级平均 → Action score
```

### Mechanism Analysis

原始路径把自然语言 action target 直接与 Kinetics-400 的固定类别词表对接。因而，同义词、复合表达和 paraphrase 可能在视觉证据完全不变的情况下无法命中 canonical class；该问题主要位于 target-to-label interface，而不一定表示视频没有呈现对应动作。

### Repair Strategy

当前 Repair v2 是固定的 Action v9 模型与 deterministic text contract 的组合，而不是重新训练一个依赖视频分数的后处理器。prompt 先经过现有 Qwen3-8B Action v9 输出 JSON，再通过声明的 60 个 action synonym family 与 Kinetics-400 canonicalization 词典规范化。该过程保留已知 action 和 `other`，拒绝歧义、否定、观看/意图等上下文，最后仍使用同一份 UMT top-5 cache 与原始评分规则。

其数据流为：

```text
prompt → Qwen3-8B Action v9 JSON →
repair-v2 text contract / 60-family canonicalization →
同一 UMT top-5 backend → Action Repair score
```

Repair v2 不读取视频分数、视觉标签、pair prediction 或 reference target，也不修改 UMT 的视觉证据。Qwen3-8B Action v9 本身为固定版本（605/67 train/dev、300 steps）；当前改善来自模型输出与 Kinetics canonical label 接口的确定性规范化，不能归因于 LoRA 单独改变了视觉评分。

### Experimental Evidence

当前 synonym 主实验包含 60 个 family、4 个生成器、每个 family 5 个 seed，共 1200 个视频，并复用同一份 UMT cache。Origin 从 `0.88917 → 0`，变化为 `−0.88917`，95% CI 为 `[−0.92833, −0.84333]`，60/60 个 family 均未保持；当前 v9+interface v2 Repair 为 `0.88917 → 0.88917`，变化为 `0`，95% CI 为 `[0, 0]`，60/60 family、1200/1200 视频保持不变。这支持“原始 target-label 接口对同义词敏感，而 Repair 在视觉证据固定时保持分数不变”。

旧的 raw v9 model chain 结果为 `0.865 → 0.378`（25/60），属于已被当前 deterministic interface v2 supersede 的消融结果，不应作为当前主方法结论。new-class 实验的 Repair 均值为 `0.00167`，known+OOV 混合场景为 `0.44458`；当前没有自然视频的人类动作真值，因此不把这些数解释为完整语义准确率。

## 6. Spatial Relationship

### Original Scoring

Spatial Relationship 的原始路径使用 prompt 中的关系目标、16 个中点采样帧以及 GRiT 检测框和标签。原始几何判定使用绝对位移，例如 \(|\Delta x|\) 或 \(|\Delta y|\)，并采用主轴严格占优和重叠权重；因此它只编码“相隔多远”，没有保留 left/right 或 above/below 的方向符号。原始 pair logic 还可能在同标签实例之间组合候选。

其数据流为：

```text
prompt → spatial target → 16 帧 GRiT boxes/labels →
unsigned |Δx|/|Δy| geometry → 帧级关系判定 → Spatial score
```

### Mechanism Analysis

无符号几何将一个有向关系压缩为无向距离关系。于是，镜像前后的 `A left of B` 与 `A right of B` 可能得到相同分数，即目标关系已经反转但评测没有响应。另一方面，实体名称规范化和多实例配对也会影响目标是否能进入几何判定；这些接口问题与方向符号丢失是两个不同机制。

### Repair Strategy

当前 Repair v2 使用微调的 Qwen3-8B Spatial parser，将 prompt 解析为有序的 `relationships[]` 三元组 `(subject, relation, object)`，并通过已有 entity vocabulary 的唯一反向映射规范化 `a/an/the` 等冠词。随后仍使用同一批 16 帧 GRiT boxes，不改变视觉检测器；Repair backend 改用带符号的几何量，其中 x 向右、y 向下，left/right 和 above/below 分别比较中心点的符号方向，同时保留官方主轴严格占优、IoU=0.1 重叠权重和多实例歧义规则。subject/object 标签匹配不同实例，并在有效有序 pair 中取最大值。

其数据流为：

```text
prompt → Qwen3-8B Spatial LoRA v8 → 有序关系三元组与实体规范化 →
同一 16 帧 GRiT boxes → signed spatial geometry → Spatial Repair score
```

`Repair-rule` 与 `Repair-model` 在当前实验中共享这一 signed backend，因此最终分数相同；Qwen 的作用主要是结构化和规范化 prompt，方向敏感性来自 Repair geometry。当前默认 `spatial_backend=repair-v2`，`legacy-official` 仅用于复现旧路径。

### Experimental Evidence

在原始方向被支持的 106 个视频/38 个 family 的 deterministic box-mirror 实验中，Origin 为 `0.5597 → 0.5597`，没有响应预期的方向翻转；Repair-rule/model 为 `0.4442 → 0`，变化为 `−0.4442`，95% CI 为 `[−0.5253, −0.3492]`，106/106 条符合预期。reverse-direction 子集 98 条中，Repair 从 `0 → 0.3724`，而 Origin 为 `0.5187 → 0.5187`。

在全部 980 个适用视频上，Origin 为 `0.29810 → 0.29810`；Repair 在 box mirror 上为 `0.07584 → 0.06057`，变化为 `−0.01527`，95% CI 为 `[−0.02911, −0.00080]`。真实视频镜像上 Repair 为 `0.07584 → 0.05310`，同步方向词的控制为 `0.06940`。106 条真实原始方向视频中，101/106 条 Repair 下降至少 `0.03`；同步方向词后仅 40/106 条恢复 `±0.03`。当前仍缺少独立人工方向真值，GRiT 检测错误和多实例歧义是主要限制。

## 7. Multiple Objects

### Original Scoring

Multiple Objects 的原始评分并不是在评分阶段重新理解完整 prompt，而是直接读取预先写入 metadata 的 `auxiliary_info["object"]` 字符串。官方协议要求这个字段提供两个目标，并用字面分隔符 `and` 将其拆成两个槽位：

```python
key_a, key_b = object_info.split(" and ")
```

随后，GRiT 对 16 个中点采样帧逐帧输出 object label set；只有当 `key_a` 和 `key_b` 都以精确字符串形式出现在同一帧的 label set 中时，该帧才记为 1，否则记为 0。视频分数是成功帧数除以 16，数据集级结果再按成功帧数与总帧数汇总。换言之，Origin 的“多个对象”实际是“metadata 中 `and` 两侧的两个目标字符串的同帧硬合取”，而不是对 prompt 中所有名词进行开放式对象识别。

其数据流为：

```text
prompt → 预生成 auxiliary_info["object"] → literal " and " split → 两个目标槽位 →
16 帧 GRiT label sets → key_a ∧ key_b 的同帧 hard conjunction → Objects score
```

例如，若 prompt 为 `a boy running and shouting`，而上游目标构造错误地把动作短语写入 object metadata，则 `running` 或 `shouting` 可能被放进目标槽位；若 metadata 写成 `boy running and shouting`，字面切分得到的两个目标甚至会是 `boy running` 与 `shouting`。Origin 不判断这些词是对象、动作还是属性，也不回到原 prompt 做语义纠正，而是把它们原样交给 GRiT label matching。由于 GRiT 通常输出 `boy` 或 `person` 等对象标签，而不会输出 `running`、`shouting` 这类动作词，错误目标只要有一个无法命中，就会把对应帧的 conjunction 直接置为 0。

### Mechanism Analysis

该路径包含两层错配风险。第一层是目标构造/metadata 的错配：literal `and` split 只产生两个字符串槽位，无法区分名词、动作和属性；一旦动作或复合短语进入槽位，后续视觉检测即使正确，也会因为目标本身错误而失败。第二层是检测证据的不稳定：仅使用当前帧的 label presence 会把瞬时漏检、短时遮挡和真正不存在混为一谈，单帧检测幻觉也可能直接计为成功。最后的 hard AND 会放大两类误差——两个目标中只要有一个没有精确命中，整帧就记为 0——而原始路径没有利用相邻帧的实例连续性。

### Repair Strategy

当前 Repair v2 先使用微调的 Qwen3-8B Objects parser 将 prompt 解析为不限于两个实体的 `entities[]`，并作实体规范化；随后仍使用同一 GRiT 标签和框，但加入严格的 temporal confirmation。当前帧中的每个实体检测必须在前一帧或后一帧存在同名框，且 IoU≥0.5；不跨越缺失帧、不在首尾回绕，也不借用相邻帧检测来制造当前帧不存在的对象。帧状态分为 `supported`、`absent`、`unconfirmed` 和 `missing`；只有 `supported` 记为 1，其余记为 0，并分别记录 abstention/missing。

其数据流为：

```text
prompt → Qwen3-8B Objects v6 → entity set →
当前帧 GRiT boxes + 相邻帧 IoU confirmation →
supported / unconfirmed / missing → 16 帧平均
```

Repair-rule 与 Repair-model 在当前实验使用相同的 temporal backend，因而结果一致；`legacy-official` 仅保留用于复现原始分数。该 Repair 是保守的时序确认，不声称能够解决所有 GRiT hallucination。

### Experimental Evidence

在 275 条模型复核的 all-frame invisible-target endpoint 视频上，正分视频数从 29 降至 15，下降 `48.3%`；正帧数从 `158/4400` 降至 `118/4400`，family endpoint rate 的变化为 `−5.0909` 个百分点，95% CI 为 `[−8.3916, −2.2436]`。全部 980 个视频的 Origin 均值为 `0.28431`，Repair v2 为 `0.26173`，变化为 `−0.02258`，95% CI 为 `[−0.02659, −0.01907]`。

在 heavy target occlusion 子集上，Origin 为 `0.28431 → 0.01754`，Repair 为 `0.26173 → 0.01040`，Repair coverage 为 `0.64872`。可见 Repair 能滤除一部分短时不可确认的正例，但也会损失部分真实可见检测（可见检测计数 `3120/4887 → 3023/4887`）。因此当前证据支持“更保守、减少瞬时检测支持”，不支持“对所有目标都无损提高准确率”；尚无完整人工视觉真值。

## 8. Background Consistency

### Original Scoring

Background Consistency 沿用官方 CLIP ViT-B/32 的逐帧全图 cosine 相似度，并使用 VBench 的 global/transition aggregation。原始评分没有建立单独的背景区域，输入仍是完整视频帧，因此前景主体的变化也会直接改变 background score。

其数据流为：

```text
视频 → 全帧 CLIP ViT-B/32 → 全图 cosine → 官方 temporal aggregation → Background score
```

### Mechanism Analysis

该路径将“背景是否稳定”与“整幅图像是否相似”混为一谈。主体移动、形变、替换或遮挡都可能造成全图表征变化，即使背景没有变化；同时，背景相似也可能掩盖前景变化。因此，背景反事实和主体反事实都可能引起同一个全图分数变化。

### Repair Strategy

当前默认 Repair 为 `patch_frame_calibrated`。反事实构造端使用 SegFormer-B0 ADE20K foreground union 与 GrabCut 生成主体保留、背景模糊的视频；这些构造 mask 不进入评分器。评分端对 clean 与 counterfactual 视频分别使用独立的 CocoSubjectBoxDetector 与 MobileSAM 定位前景，再取同一 CLIP 的背景互补 patch tokens，以相同精度计算 all-pairs cosine，并使用固定分母和 calibrated gain=1.75 汇总。新的 104-class construction/localizer 候选仍处于暂停状态，未提升为默认方法。

其数据流为：

```text
视频 → 独立 foreground box / MobileSAM → 背景互补 patch tokens →
CLIP patch cosine + fixed denominator + calibration → Background Repair score
```

### Experimental Evidence

冻结 holdout 包含 1040 个候选，其中 188 条接受、852 条拒收，0 条运行时失败，共 1504 个版本。全图 Origin 在 clean→background blur 上为 `0.946845 → 0.934706`，MAE 为 `0.016240`；Repair 为 `0.962999 → 0.959592`，MAE 为 `0.006733`。自然偏好从 `52.12%` 提升至 `61.22%`，变化为 `+9.10` 个百分点，95% CI 为 `[+6.03, +12.18]`；background response retained 为 `88.80%`，95% CI 为 `[84.13, 94.42]`。

视觉审计发现 8 个 mask error；188 条中有 29 条 Repair full absolute change 超过 `0.01`，最大为 `0.184038`。因此当前结果支持“背景专用证据降低前景变化的干扰”，但 mask 质量和独立人工语义 IoU 仍是限制。

## 9. Object Class

### Original Scoring

Object Class 的官方路径使用 GRiT 检测到的 label set，对目标 object string 做 exact membership 判断。原始下游主要保留标签是否存在，未统一大小写、单复数或别名；confidence、box 和完整实例级证据不进入最终判定，并存在 dropped/null 行和条件分母。

其数据流为：

```text
prompt → object target → GRiT label set → exact string membership →
帧级/视频级聚合（含原始 dropped/null 行为）
```

### Mechanism Analysis

原始路径把“视觉上未检测到”和“检测到了但词形不同”混为一谈；同时，丢弃行或条件分母会使不同候选集之间的分数不易比较。仅保留 label presence 也无法区分 alias hit、other class、no detection 和 runtime failure。

### Repair Strategy

当前 Repair 默认使用 deterministic label vocabulary/alias compiler，并保留 GRiT 的完整实例、框、confidence 与帧顺序。每个目标显式记录 `exact_hit`、`alias_hit`、`no_detection`、`other_class` 和 `runtime_failure`；错误或 null 不自动记为 0，若主聚合不完整则返回 null，subset mean 仅作为诊断。Qwen3-8B Object LoRA 是可选的 prompt semantic compiler，将原始 prompt 转成冻结 JSON；它不替代 GRiT，也不直接修改视觉分数。

其数据流为：

```text
prompt → deterministic alias compiler（可选 Qwen3-8B Object LoRA） →
canonical target → 完整 GRiT instances/boxes/confidence →
状态分类与全帧分母 → Object Class Repair score
```

### Experimental Evidence

计划为 5 个 dev、20 个 test；实际接受 2 个 dev、14 个 test，拒收 9 个，共 63 个 request，63/63 双输出完成，0 条运行时失败。Test uppercase 反事实中，Origin 为 `1 → 0`，Repair 为 `1 → 1`，中位绝对变化为 `1`，95% CI 为 `[1, 1]`；alias control 的 deterministic Repair 保持不变，absent proxy 为 `0 → 0`。自然 Object 实验包含 1580 个视频、2370 个 pair、79 个 prompt，test 为 1410 个；deterministic/LoRA Repair 均为 `0.3851`，base 为 `0.2638`，差值为 `−0.1213`，95% CI 为 `[−0.1752, −0.0716]`。当前没有独立人工语义真值，Qwen compiler 不能被解释为视觉检测准确率提升。

## 10. Color Consistency

### Original Scoring

Color 与 Multiple Objects 一样依赖固定的文本字段和 GRiT 字符串标签，但 Origin 的解析操作并不是把 `and` 拆成两个对象。官方实现读取 `auxiliary_info["color"]` 作为 `color_key`，再对完整 prompt 做固定字符串处理：删除 `a `、`an ` 以及颜色字符串，剩余文本整体作为 `object_key`：

```python
color_key = info["auxiliary_info"]["color"]
object_key = info["prompt"].replace("a ", "").replace("an ", "")
object_key = object_key.replace(color_key, "").strip()
```

因此，`and` 在 Color Origin 中不是一个语义解析器；如果 prompt 中包含多个并列短语，它会留在剩余的 `object_key` 中，而不是像 Multiple Objects 那样产生两个独立目标。随后 GRiT 对 16 个采样帧生成 dense caption 与对象标签。对每个检测片段，官方实现使用返回对象列表中的第一个类名（`[2][0]`）作为该 caption 的对象标签；只有当它与 `object_key` 精确相等时，才进入目标对象计数。目标对象被计入后，再用 12 色白名单判断该 caption 是否提供颜色证据，并用 `color_key in caption` 的子串匹配判断目标颜色。最终分数是“有颜色证据的帧数 / 有目标对象证据的帧数”；如果整条视频没有目标对象证据，则该视频被上游丢弃。

其数据流为：

```text
prompt + auxiliary_info["color"] → 删除 a/an/颜色词的字符串处理 → object_key, color_key →
16 帧 GRiT dense captions + object labels → object_key 精确匹配 →
12 色白名单的对象存在门控 → color_key 子串匹配 → 条件帧平均 → Color score
```

例如，若 prompt 为 `a red car and a blue bicycle`，而 `color_key=red`，Origin 的字符串处理可能得到 `car and blue bicycle` 这一整个 `object_key`，而不是得到 `car` 与 `bicycle` 两个对象。因此，即使 GRiT 分别检测到 car 和 bicycle，也未必有任何标签能精确匹配这个复合字符串。这里与 Multiple Objects 的共同点是固定字符串接口会放大目标构造错误；不同点是 Multiple Objects 明确用 `split(" and ")` 产生两个目标，而 Color 会把 `and` 保留在单个对象查询中。

### Mechanism Analysis

Color Origin 的问题来自多个串联的词法和实例错配。首先，prompt 到 `object_key` 的删除式解析无法理解并列对象、修饰范围或颜色所属对象，且把 `and` 留在查询中；其次，`object_key` 与 GRiT 标签使用精确相等，词形差异会被当成未检出。再次，每个 caption 都使用对象列表中的 `[2][0]`，导致目标对象不是该位置实例时无法进入分母。最后，12 色白名单决定“对象是否存在”的条件分母，而目标颜色使用无词边界的 substring：例如 `red` 可能命中 `colored` 或 `hundred`。因此，未检测到对象、目标颜色错误、字符串未对齐、检测失败和上游丢弃并不具有相同含义，却可能被压缩为一个条件均值或被直接排除。`crimson`、`navy`、`maroon` 等具体颜色也不应未经声明就折叠成 `red` 或 `blue`。

### Repair Strategy

当前 Repair 不继承 Origin 的删除式 `and`/颜色字符串处理，而是用 Qwen3-8B Color semantic compiler 输出绑定在同一条记录中的 `{"object": ..., "color": ...}`；无法无歧义解析时返回 null 或 abstain。视觉侧保留完整 GRiT instances、boxes、confidence 和检测顺序，通过唯一 IoU≥0.9 的 object-instance binding 绑定对象与颜色。若无法唯一绑定则 abstain。颜色判断使用 word boundary、否定和 object scope 规则，并显式声明 aliases；聚合采用完整帧分母，区分 exact/alias hit、no detection、wrong color、ambiguity 与 runtime failure。Qwen3-8B Color LoRA 只作为 prompt semantic compiler，不是视觉 scorer，也不直接改变 GRiT 证据。

其数据流为：

```text
prompt → object/color semantic compiler（可选 Qwen3-8B Color LoRA） →
目标对象与颜色 → GRiT instances → unique IoU binding →
scope/negation/word-boundary 判定 → 全帧聚合 → Color Repair score
```

### Experimental Evidence

计划为 5 个 dev、20 个 test；实际接受 2 个 dev、5 个 test，拒收 18 个，共 42 个 request。官方输出仅有 `33/42` 条有效行，9 条被丢弃；Repair 为 `42/42` 完整。5 个 test base 的五级可见性反事实中，Origin→Repair 的 median drop 为 `1.0`，95% CI 为 `[0.625, 1]`，5/5 单调；2 个 dev base 中仅 1/2 单调，dev gate 未通过。官方 synonym control 没有有效 paired row；自然 Color 实验和 0/200 人工复核均为 `NOT RUN`。因此当前结果只支持修复分母、绑定和可追溯状态，不支持完整颜色语义准确率结论。

## 11. Temporal Style

### Original Scoring

当前仓库中的 temporal-style package 仍是 M1 scaffold。`vbench` 和 `audit` 返回 `status=not_implemented`、`score=null`；没有可用于当前实验的 ViCLIP/BPE 权重或 checkpoint，也没有 numeric parity。因此本维度目前不能把某个历史 temporal-style 数值写成 Origin score。

### Mechanism Analysis

当前没有已验证的数值评分，因而也没有足够证据将某一种 temporal-style 偏差归因于原始指标。离线 prompt token audit（100 个 prompt，token 数 min/median/max 为 `6/13/37`，10/100 超过 30）只说明文本长度分布，不构成视频风格实验。

### Repair Strategy

当前没有实现 Repair；不引入 ViCLIP、BPE 或旧的未完成接口作为正式方法。相关候选数、评分数、置信区间和人类偏好均标记为 `NOT RUN`。

其当前状态为：

```text
视频 → temporal-style backend（未实现） → score=null
```

### Experimental Evidence

本维度所有正式反事实、开发集、后续验证和人类偏好实验均未完成。后续若实现，应首先固定官方 backend、权重、采样协议和 aggregation，再单独建立 Repair 对照。

## 汇总表

下表用“数据流”概括当前方法；括号中的两个数字统一表示 `base → counterfactual`。分数变化的好坏取决于反事实：对于应保持不变的干预，变化越接近 0 越好；对于应改变语义的干预，分数应按预期方向变化。

| 维度 | Origin | Repair | 敏感性 / 不变性实验与当前结论 |
|---|---|---|---|
| Dynamic Degree | 当前方法与正式协议待确认 | 暂空，`NOT RUN` | 暂无证据；不引用历史方案 |
| Motion Smoothness | 当前方法与正式协议待确认 | 暂空，`NOT RUN` | 暂无证据；不引用历史 temporal aggregation |
| Subject Consistency | 全帧 DINO；相邻帧 + 首帧锚定 | 独立 Mask R-CNN/MobileSAM 主体定位 → 主体裁剪 → DINO patch → all-pairs | 背景模糊：`0.936417 → 0.816382`，MAE `0.120035`；Repair `0.951413 → 0.941724`，MAE `0.010596`。背景不变性改善，但严格全量 gate 未通过；主体干预仍平均降分 `0.099319` |
| Scene | Tag2Text caption + 场景词严格字符串匹配 | Qwen3-8B Scene Verifier(prompt, caption) → supported/contradicted/insufficient | 同义替换：Origin `0.30594 → 0.05406`；Repair `0.6909375 → 0.6946875`。Repair 抑制词面敏感性；caption 误差和独立视觉真值仍有限 |
| Human Action | UMT top-5 + threshold `0.85` + exact Kinetics match | Qwen3-8B Action v9 → repair-v2 canonicalization → 同一 UMT backend | 同义替换：Origin `0.88917 → 0`；Repair `0.88917 → 0.88917`。接口不变性得到支持；无自然人工动作真值 |
| Spatial Relationship | GRiT + unsigned `|Δx|/|Δy|` geometry | Qwen3-8B Spatial triples → 同一 GRiT boxes → signed geometry | 方向镜像：Origin `0.5597 → 0.5597`；Repair `0.4442 → 0`。Repair 对有向关系敏感；人工方向 gold 缺失 |
| Multiple Objects | `auxiliary_info["object"]` → literal `and` split 成两个目标 → GRiT label sets + 同帧 hard conjunction | Qwen3-8B entities → GRiT boxes + 相邻帧 IoU≥0.5 confirmation | 全矩阵 Origin `0.28431`，Repair `0.26173`；Repair 更保守、减少瞬时支持，但有可见检测损失 |
| Background Consistency | 全帧 CLIP cosine + 官方聚合 | 独立 CocoSubjectBoxDetector/MobileSAM → 背景 patch CLIP → calibrated all-pairs | 背景模糊：Origin `0.946845 → 0.934706`，Repair `0.962999 → 0.959592`；背景不变性改善，但 mask error 仍存在 |
| Object Class | GRiT exact label membership，含 dropped/null 行为 | alias/词表规范化 + 完整 GRiT 实例状态；Qwen Object LoRA 仅作 compiler | uppercase：Origin `1 → 0`，Repair `1 → 1`；支持词形不变性，但自然语义真值缺失 |
| Color Consistency | `auxiliary_info["color"]` + prompt 删除式 object_key（`and` 不拆分）→ GRiT `[2][0]` 标签 + 12 色门控 + 颜色子串 | Qwen3-8B 绑定 object/color → IoU≥0.9 唯一绑定 + scope/negation/word-boundary + 完整分母 | 5 个 test base 的可见性干预中 Repair 5/5 单调；dev gate 未通过，自然实验 `NOT RUN` |
| Temporal Style | 当前 scaffold，`score=null` | 未实现 | 全部正式实验 `NOT RUN` |

## 证据与复现入口

- Subject 的正式交接报告：`docs/counterfactual-reports/subject_stability_20260920.md`；对应已推送提交为 `6ae1ef5`。
- 本地 Background、Object Class、Color 和 Temporal Style 的实现报告与配置位于 `metrics/` 下对应维度目录；本稿中的数值以冻结 holdout 或对应报告为准。
- Scene、Human Action、Spatial Relationship 和 Multiple Objects 的当前 v2/v8/v9 实现、脚本和确定性报告位于相邻的 `vbench-prompts-compile` 工作区；其 working tree 中已有用户未提交的最新 v2 改动，不能用相邻仓库的旧 HEAD 文档替代当前源码。
- 所有实验均应同时记录：模型/代码版本、prompt 或 split 来源、候选与实际评分数量、拒收和失败、输出路径、聚合规则及原始 JSON/CSV 证据。缺少任一项时，本文只报告为限制或 `NOT RUN`，不补猜测数字。
