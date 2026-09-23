# 各维度 Origin / Repair 方法与反事实实验（初稿）

本文件是供确认的独立初稿。**2026-09-23仅补齐Dynamic Degree当前aligned-v1与450组冻结结果**，同步论文、AGENTS.md和README.md；Motion Smoothness仍暂空。其余维度保留此前初稿快照，本轮未重新核验，不能把旧解析器/旧结果当作当前论文表值。较新论文风格说明见 [VBENCH_DIMENSION_METHODS_PAPER_STYLE.md](VBENCH_DIMENSION_METHODS_PAPER_STYLE.md)。本次不清理旧代码，不替换公开默认，不提交其他任务的改动。

## 统一阅读规则

- 实验表中的 Repair 单元格尽量写成 “base → 反事实”；若一个单元格是多档 profile，会按档位顺序列出。
- CPA、平均绝对变化、准确率和 natural preference 是不同统计量，不互相替代；区间沿用对应原始报告的 paired/bootstrap 95% CI。
- 当前方法的负结果、失败、拒收和 NOT RUN 都保留；原始历史报告不在正文重复展开。
- Subject 只按最新 v5 主体定位/隔离 Repair 撰写；历史方案不在本稿重复。
- 本初稿不移动、删除或重写 data/、results/、splits/、runs/，也不把模型权重和大体积输出加入 Git。

统一索引入口：[EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)。原始实现、报告和输出路径在各章节保留。

---

## 1. Dynamic Degree（2026-09-23当前aligned-v1）

### Origin实现与问题

```text
官方视频 → 约8 fps采样 → 相邻帧RAFT → top-5%光流幅值均值
  → 分辨率缩放阈值 + 超阈值帧对计数 → 视频二值判分 → 动态视频比例
```

原版将足够大的局部位移作为动态证据，因此纹理快速往返抖动也可能得到高分，却没有新增连贯对象/相机运动。对应问题是Nuisance Entanglement，本实验检验无关抖动下的**不变性/稳定性**，而非让评分对真实运动也不敏感。

### Repair数据流

```text
原生16帧/8 fps → 全帧384² → 冻结V-JEPA 2.1 ViT-B
  → 4608×768时空tokens → 投影 + 注意力池化 + MLP（51,393参数）
  → sigmoid连续相对运动分数
```

210训练源/60开发验证源，联合人类偏好、原片/抖动一致性、静止低分锚点、平移/连贯往返排序。最后从既有锚点头继续固定300步，只用TRAIN原片Origin总体均分提供弱尺度监督；不用450评估视频训练、不复制逐视频0/1、不后处理调分。实际源码：[冻结编码器](../packages/audit-models/src/vbench_audit_models/vjepa.py)、[小头](../metrics/dynamic-degree/src/dynamic_degree/learned_probe.py)、[锚点损失](../metrics/dynamic-degree/src/dynamic_degree/anchored_probe.py)、[均分损失](../metrics/dynamic-degree/src/dynamic_degree/aligned_probe.py)。

### 反事实构造、结果与相关控制

450组官方VBench 1.0原始MP4，每源两固定种子8px局部纹理坐标往返抖动，原始16帧/8 fps/2秒不变。不加RGB噪声、亮度闪烁或替换成静态片；另有450编码控制，共1800/1800评分、零失败。历史1200 qualified/600 rejected标记输入全部保留；先前30个GIF仍未评分。Origin沿用同批锁定结果，Repair重新推理；450编码控制和全部1800特征哈希验证一致。

| 实验 | Origin（base → CF） | 当前Repair（base → CF） | 结论 |
|---|---|---|---|
| 450源、900条8px干预 | 0.680000 → 0.857778 | 0.629251 → 0.623035 | 批量虚增抑制；MAE分别0.177778/0.021731 |

Repair有符号Δ=−0.006216，95% prompt聚类CI [−0.009879,−0.002862]；MAE区间[0.018178,0.025484]。先每源平均两CF再源等权，不以总体均分差替代MAE。人类171有序对原片/两CF各144正确，但逐对会变化，原片相对Origin改善CI跨零。

冻结后单图静止/静止加抖动/8px平移/32px平移诊断得分0.007435/0.013868/0.351916/0.554858，单独报告，不充当450主实验或严格单因素消融。训练阶段DEV60自然偏好17/23未达≥19/23，严格尺度门槛仍失败。450组仍有28/900下降>0.1，最大下降0.377932；尚不能称逐片不变、物理强度标定或全新独立留出。旧模型对照保留在报告中，不混入本节当前Repair。

权威入口：[当前450报告](counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-aligned450)、[冻结协议](../configs/dynamic-static-jitter/vjepa-aligned450-v1.json)、[原始统计](../output/dynamic-static-jitter/vjepa-aligned450-v1/analysis/summary.json)。论文位置：方法§2.2.2及Table 1第六行；[专项说明与版本](../../overleaf/docs/dynamic-degree-stability.md)。公开默认不变。

---

## 2. Motion Smoothness（本版暂空）

本版按要求暂不整理 Motion Smoothness 的 Origin、问题、Repair、反事实实验和消融，待方法确认后补写。

---

## 3. Subject Consistency

### Origin 实现方法

Origin 使用锁定 VBench 的 load_video 读取全部解码帧；每帧经过官方 dino_transform(224)，由 DINO ViT-B/16 得到 L2-normalized whole-frame feature。相邻帧余弦相似度负值截为 0，视频分数由相邻项与固定首帧锚定项等权组成：

Origin = 0.5 × L（相邻帧平均）+ 0.5 × A（首帧到所有帧平均）。

数据集聚合还保留 VBench 的 single-GPU transition-weighted 与 multi-GPU video-average 差异。源码和当前包说明见 [Subject 实现报告](../metrics/subject-consistency/IMPLEMENTATION_REPORT.md)。

数据流为：

~~~text
视频 -> 全部解码帧 -> DINO ViT-B/16 whole-frame feature
     -> adjacent cosine + first-frame anchor
     -> Origin score
~~~

### Origin 问题

第一，固定首帧是 privileged anchor，导致时间位置依赖，不能对整段轨迹对称地计算一致性。第二，whole-frame DINO 同时读取主体、背景、构图和尺度，背景变化也可能被当成主体不一致。

### Repair 方法

最新 v5 稳定性实验使用主体定位/隔离 Repair。反事实构造仍用 SegFormer-B0 + GrabCut 生成“主体保留、背景糊化”的版本；评分阶段使用独立的 Mask R-CNN + MobileSAM，在每个实际 clean/反事实 clip 上逐帧重新检测和分割，并读取官方 subject_en。评分不读取构造掩码，不借用 clean 版本的框或特征，也不通过 DINO score 选择路径。

v5 的表示为 encode-time subject isolation：主体外像素 RGB fill=128，10% margin 的 square crop 到 224，DINO patch pooling，再做 all-pairs subject evidence aggregation；主实验的 missing policy 是 exclude。源码中仍有 NpzSubjectMaskProvider 和 SAM3 adapter 等通用接口，但 720 主稳定性结果的实际 localizer 是独立 Mask R-CNN + MobileSAM。

### 实验设置（反事实构造）

最新 v5 稳定性实验：72 prompts × 10 candidates = 720 candidates，4 generators × 180。241 条构造并完成主评分，479 条构造拒收；排除预定主分析中的 known truck 后，主分析分母为 240 bases。主条件为开头四分之一背景 Gaussian blur，主体像素保持；middle、end、full 是对照。主评分 versions 2,169 条全部完成、无 runtime failure。单帧 supplement 在 204/241 条时停止，completed=false，不能写成全量完成。

### 实验表：最新 v5 主稳定性实验

| 条件 | Origin | Repair v5 hybrid |
| --- | ---: | ---: |
| opening-quarter background blur | 0.936417 → 0.816382；MAE 0.120035，CI [0.106987, 0.134431] | 0.951413 → 0.941724；MAE 0.010596，CI [0.008069, 0.013583] |
| subject intervention（repair diagnostic） | — | 平均降分 0.099319，CI [0.093839, 0.104371] |

主结果中 180/240 条 repair ≤0.01，worst=0.149715；联合 background gate 为 74/240，同时满足 subject response 的为 70/240。用户接受“0.01 左右”的实用均值，但原严格 aggregate ≤0.01 门槛仍未通过。

### Repair 相关实验与消融

- v5 development 34/34 完成：Origin MAE 0.131364，Repair 0.009979，repair prompt-cluster CI [0.006280, 0.014117]，联合成功 10/34；严格 ≤0.01 的统计门槛未通过。
- normalized followup 的 60 个 main bases：start Origin/Repair=0.080759/0.006353，middle=0.037052/0.005491，end=0.028937/0.003222，full=0.028567/0.009856；这是机制 followup，不替代 720 主实验。
- dose 2x/4x 同一 60 bases：start Origin/Repair=0.088412/0.006389 和 0.088890/0.006455；联合成功为 9/60、7/60。
- 两候选 quality2：pre-encode crop 将 background absolute change 从 0.014485 降到 0.005910，6/6 partial subject interventions 降分；n=2，不能外推。
- 1440 natural extension 中 isolated-crop candidate 为 48.76%，paired CI [−13.57, −5.74] pp，失败；后续 CLS candidate 57.05% vs Official 58.45%，CI [−3.88, +1.09] pp，未优于 Origin。

### 证据与边界

最新交接：[subject_stability_20260920.md](counterfactual-reports/subject_stability_20260920.md)；实现：[metrics/subject-consistency/IMPLEMENTATION_REPORT.md](../metrics/subject-consistency/IMPLEMENTATION_REPORT.md)；方法说明：[subject_consistency.md](paper/subject_consistency.md)。当前可支持的是 v5 在主背景干预上的平均稳定性改善；严格逐条门槛、单帧补充和自然集全面优越性仍有限。

---

## 4. Scene

### Origin 实现方法

当前正式四维矩阵中的 Origin 仍是 VBench 官方的 caption lexical path。数据流为：

~~~text
原视频
  -> 固定 16 个 midpoint frames
  -> Tag2Text 逐帧 caption cache
  -> 官方 scene key + caption substring match
  -> 每帧 0/1
  -> 16 帧均值 / video mean
~~~

也就是说，Origin 的目标来自官方 auxiliary scene key，视觉证据来自 Tag2Text caption；它不是一个从 prompt 直接理解场景的生成模型。当前矩阵原始分数均值为 0.36382（1,040 个适用视频）。

### Origin 问题

核心问题是 lexical presence：同义改写、词形变化和 caption 没有写出目标场景时，字符串命中会直接失败；反过来，substring collision 也可能误报。这个分数只表示 caption 是否含有官方字符串，不表示场景覆盖面积，也不等同于人工视觉正确率。200 个同义反事实中，Origin 从 0.30594 降到 0.05406，paired delta 为 −0.25188，95% CI [−0.44094, −0.04500]。

### Repair 方法

针对上述文本目标与视觉描述的错配，Repair 保留同一份 Tag2Text 视觉描述，不更换帧采样、caption 模型或视觉输入；它只把原来的字符串命中替换为语义证据判别。正式矩阵使用独立的 Qwen3-8B `SceneVerifier`，不是四象限、OpenCLIP、ROI/segmentation 或视觉区域聚合。Qwen3-0.6B 只做过消融/冒烟，不作为正式结果。

实际实现是 `scripts/predict.py --task scene` 调用 `src/vbench_prompts_compile/inference.py` 中的 `SceneVerifier`：每个请求同时输入**原始完整 prompt**和对应的**单帧 Tag2Text caption**，模型通过 Qwen chat template 生成一个标签；严格解析只接受 `supported`、`contradicted`、`insufficient` 三者之一。批量入口按 `(prompt, caption)` 去重并批量推理，但同一视频的 16 个 caption 仍逐帧独立保留。`score_matrix.py` 再将 `supported` 映射为 1，将 `contradicted` 和 `insufficient` 映射为 0，同时记录 label coverage 与 insufficient/abstention；因此 Repair 的视频分数仍是 16 个帧级证据判断的均值。

~~~text
视频
  -> 与 Origin 完全相同的 16 帧 midpoint sampling
  -> Tag2Text 逐帧 caption（共享、冻结的视觉证据）
  -> (原始 prompt, 当前帧 caption)
  -> 独立 Qwen3-8B SceneVerifier
  -> supported / contradicted / insufficient
  -> supported=1；其余=0；另记 coverage / insufficient
  -> 16 帧平均
  -> Repair-model video score
~~~

Scene model 与 Spatial / Human Action / Multiple Objects 的共享三 LoRA 路径不同：它是独立模型，输入同时包含 prompt 和 caption；后三者是 prompt-only 的任务适配器。当前正式底座为 Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`，Scene v8 使用 455/108 train/dev、固定 300 steps。注意：`Repair-rule` 只是声明的封闭同义词典控制，它仍调用同一官方 substring 评分，不等于 `Repair-model` 的语义 verifier。Repair-model 原始矩阵均值为 0.68912。

### 实验设置（反事实构造）

最终四维矩阵冻结 1,040 个 Scene 计划项，1,040 个适用并完成；适用来源为 52 个 source families，主要变换为严格同义和 scene-condition replacement。主同义矩阵是 200 个视频、10 个 source families；同一视频、同一 16 帧 caption cache 同时进入 Origin、Repair-rule 和 Repair-model，避免把视觉缓存差异误认为方法差异。

补充证据核验包含 3,200 个 caption pairs（200 videos × 16 captions，10 个 families）；恢复的 test2 有 492 observations、5 个 connected source blocks。ocean→sea 是 20 videos、1 个 family 的 dev-only 诊断，不能当独立 test。矩阵级执行完成，没有把旧四象限 coverage fixture 作为当前 Scene 结果。

### 实验表（每格为 base → 反事实）

| 条件 | Origin | Repair-model（Qwen3-8B） |
| --- | ---: | ---: |
| 原始矩阵均值（1,040） | 0.36382 | 0.68912 |
| 严格同义主矩阵（200 videos / 10 families） | 0.30594 → 0.05406；Δ −0.25188，CI [−0.44094, −0.04500] | 0.69094 → 0.69469；Δ +0.00375，CI [−0.01188, +0.02188] |

### Repair 相关实验与消融

- Repair-rule 是冻结的 deterministic parser，使用声明过的严格同义词典；在同义主矩阵中为 0.35813 → 0.35813，逐视频不变率为 1.00。这是词典内控制，不是开放表达泛化。
- 对 3,200 个 caption pairs 做“正确且一致”补充核验：Origin 为 0.26719，95% CI [0.19063, 0.34813]；Repair-rule 为 0.62438，CI [0.49219, 0.75031]；Repair-model 为 0.92125，CI [0.88438, 0.95563]。Repair-model 相对 Origin 的 paired 提升为 +0.65406，CI [+0.57281, +0.73375]。
- 恢复 test2 的 432 个官方定义域内共同二分类项：Origin/Repair-rule 为 0.81713，CI [0.77888, 0.89848]；Repair-model 为 0.96296，CI [0.94200, 0.98980]，paired 提升 +0.14583，CI [+0.07868, +0.19250]。全 492 条三分类中，Repair-model accuracy=0.82927、macro-F1=0.79337，CI [0.76210, 0.81956]。
- 20-video ocean→sea dev 诊断中，Origin 0.56250 → 0.03125，Repair-rule 0.57500 → 0.57500，Repair-model 0.95938 → 0.95938；后两者 20/20 不变，但只有 1 个 family，没有独立正确性金标。
- 83/492 个 test2 observation 的视觉标签与 caption-evidence 标签不同；60 个 insufficient 候选没有官方 scene key，不能当成 60 个已确认的“无场景”金标。Repair-model 的改善主要证明 caption evidence interface 的一致性，不是完整的人类视觉真值准确率；其 correct-and-consistent 还略低于简单复制原判定的 0.92906 baseline。

### 证据与边界

最新方法与矩阵证据在 H100 项目 /root/wenbiao_zhao/vbench-prompts-compile-git：docs/deterministic-experiments-report.md、docs/EXECUTION-2026-09-20.md、scripts/predict.py、scripts/score_matrix.py，代码提交 ea9d500，最终评分代码提交 4d2a78d。本仓库的旧 metrics/scene/IMPLEMENTATION_REPORT.md 和旧四象限报告不再作为当前 Scene 方法依据；待初稿确认后再处理其历史标记和 README/AGENTS 指向。

---

## 5. Human Action

### Origin 实现方法

当前正式矩阵的 Origin 是官方 action target 编译加 UMT 闭集分类。数据流为：

~~~text
原始 prompt / 官方 human_action metadata
  -> official action target
  -> 同一视频的 UMT ViT-L/16 Kinetics-400 top-5 cache
  -> sigmoid score 四舍五入 4 位并以 0.85 threshold 判定
  -> target 与 top-5 label 精确匹配
  -> 每视频 bool / dataset mean
~~~

缓存阶段同时保留 official filename label，并用 8 个视频做 UMT upstream parity；但最终四维 matrix 的 target 由冻结 metadata/prompt compiler 产生。原始矩阵中 1,200 个适用视频的 Origin 均值为 0.88917。

### Origin 问题

Origin 把自然语言 action target 直接压成 K400 label，再做 hard top-5 match；表达改写后，UMT 视觉证据不变而文本 target 可能无法映射到同一个 K400 类。0.85 threshold 和 bool mean 还会丢失连续 action evidence。Origin 在同类同义反事实中从 0.88917 直接降到 0，表现为文本接口的 contract failure，而不是视频运动本身变差。

### Repair 方法

当前 Repair 是共享冻结 Qwen3-8B backbone 上的 Human Action LoRA v9。Spatial、Human Action、Multiple Objects 各自硬路由到独立 LoRA；Action 不自动调用其他 adapter，也不由末端 LLM 做视频匹配。数据流为：

~~~text
prompt（只给文本）
  -> AdapterRouter: task=action
  -> 共享冻结 Qwen3-8B + action LoRA v9
  -> JSON actions[]
  -> K400 vocabulary canonicalization；未知动作保留为 other
  -> 同一份 UMT top-5 cache
  -> 每个已知 action 调 official action_score；other 计 0
  -> actions 平均为 video score，再汇总到 matrix
~~~

因此 Repair 的职责是把 prompt 解析成结构化 K400 action set；UMT 仍是冻结视觉后端，LoRA 没有把 UMT 改成连续动作模型，也没有自行修正 UMT 误分类。Action v9 训练使用 605/67 train/dev、固定 300 steps；旧 Action v5 的泄漏来源审计后移除 113 个相关 source groups，再以固定超参重训。

### 实验设置（反事实构造）

最终矩阵计划 1,200 项、1,200 项适用，60 个 source families，所有 action 变换复用同一 UMT top-5 visual cache。反事实包括：同类同义、新 K400 类替换、OOV、已知类 + OOV。模型文本接口的严格有效性为 58/60 个独立 prompt families；其余输出/映射失败保留在账本中，不清零。

同义实验有 60 个独立表达；OOV 的 assembling furniture 只有 1 个独立 prompt，虽然被配到 1,200 个视频，因此不报虚假的大样本 CI。新 K400 类替换只改变文本目标，视频中是否真的没有该动作没有全量独立人工真值，故只称 protocol control。

### 实验表（每格为 base → 反事实）

| 条件 | Origin | Repair-model（Qwen3-8B + action LoRA v9） |
| --- | ---: | ---: |
| 同类同义（60 independent prompts） | 0.88917 → 0 | 0.86500 → 0.37833；Δ −0.48667，CI [−0.59750, −0.37583] |
| 新 K400 类 | 0.88917 → 0.00167 | 0.86500 → 0.00167 |
| OOV（1 unique prompt） | 0.88917 → 0 | 0.86500 → 0 |
| 已知类 + OOV（60 prompts） | 0.88917 → 0 | 0.86500 → 0.34125 |

### Repair 相关实验与消融

- Repair-rule deterministic parser 是闭合词典消融：同类同义为 0.88917 → 0.88917，60/60 个独立表达解析为预期 K400 target；它证明规则字典可以保持词典内不变，不证明开放表达泛化。
- Repair-model 同义表达只有 25/60 个 target set 正确，35/60 输出 other；把 other 当“不变”会掩盖语义失败，主结果不这样计。
- known + OOV 混合中，Repair-rule 的总分为 0.44458，Repair-model 为 0.34125；模型集合正确率为 45/60，已知 action 的保留情况另行记录，不能把 other 的计零简单解释为已知动作丢失。
- 新 K400 class 的变换均值、OOV 的 1 个 unique prompt 和 UMT parity 是协议/后端核验，不是 human action semantic accuracy。当前没有独立人类金标、自然集 Repair preference 或连续 target-probability calibration，均不作已完成结论。

### 证据与边界

最新方法与矩阵证据在 H100 项目 /root/wenbiao_zhao/vbench-prompts-compile-git：docs/deterministic-experiments-report.md、docs/EXECUTION-2026-09-20.md、scripts/predict.py、scripts/score_matrix.py，主 Action 权重为 v9。旧 metrics/human-action/IMPLEMENTATION_REPORT.md 中的 explicit target_action、UMT temporal-window scalar 路径不再作为当前 Repair 依据；待初稿确认后再处理历史方法标记。

---

## 6. Spatial Relationship

### Origin 实现方法

当前正式矩阵的 Origin 数据流为：

~~~text
prompt + official spatial metadata
  -> official ordered target
视频 -> 16 个 midpoint frames -> GRiT object boxes / labels cache
  -> 官方 spatial_scores / get_position_score
  -> 同名框池化、无序 box pair、abs(dx)/abs(dy)
  -> 16 帧均值 / video mean
~~~

GRiT cache 与文本接口分离；Origin 的最终关系判据来自锁定 VBench upstream。当前 980 个适用视频的 Origin 均值为 0.29810。

### Origin 问题

这是源码级的 direction/role contract 问题：官方几何使用绝对差，忽略 left/right、above/below 的符号，并可能从 A/B 的同名框池组合 pair。确定性镜像框坐标后，分数可以完全不变，所以 Origin 不能保证有序关系 R(subject, object) 真正翻转。

### Repair 方法

当前 Repair 是共享冻结 Qwen3-8B backbone 上的 Spatial LoRA v8，不是本地旧的 signed geometry / ordered-role scorer。数据流为：

~~~text
prompt（只给文本）
  -> AdapterRouter: task=spatial
  -> 共享冻结 Qwen3-8B + spatial LoRA v8
  -> JSON relationships[]: subject / relation / object
  -> entity / relation canonicalization
  -> 映射到官方关系短语
视频 -> 同一份 16 帧 GRiT boxes cache
  -> 官方 unsigned spatial_scores
  -> 每关系、每帧、每视频聚合
~~~

Repair-model 只替换“prompt 如何变成关系 triple”，不替换检测器、框配对、几何公式或最终聚合。关系输出要求保留有序 subject/object，但当前后端仍用官方 unsigned geometry；因此 Qwen LoRA 不能单独修复评分公式本身。Spatial v8 训练为 2,679/312 train/dev、600 steps；严格接口有效率为 46/49 个独立 families。

### 实验设置（反事实构造）

最终矩阵计划 1,300 个 Spatial 项，980 个适用原始项，49 个 source families；变换包括 evidence mirror、real video mirror、relation-word swap 和 joint transform。变换共尝试 2,600 项，其中 1,960 个适用且完成；不适用项保留在计划和 exclusion ledger。所有 scheme 共享同一份视觉 cache，镜像只改变证据或按协议同步改变关系词。

### 实验表（每格为 base → 反事实）

| 反事实 | Origin | Repair-model（Qwen3-8B + spatial LoRA v8） |
| --- | ---: | ---: |
| 原始矩阵均值（980 applicable） | 0.29810 | 0.28020 |
| 确定性 box mirror（官方几何） | base → base；Δ=0，CI [0, 0] | base → base；Δ=0，CI [0, 0] |
| real video mirror（模型 scheme） | Δ −0.00125，CI [−0.01366, +0.01107] | Δ −0.00367，CI [−0.01544, +0.00838] |
| video mirror + relation-word swap | Δ −0.00125 | Δ −0.00206，CI [−0.02023, +0.01698] |

确定性 mirror 的 base 与反事实严格相同，不应从 0.29810/0.28020 的全矩阵均值外推一个 subset 的绝对反事实均值。真实视频 mirror 的小幅变化来自 GRiT 对镜像的检测不等变性，不代表方向判据正确翻转。

### Repair 相关实验与消融

- Repair-rule 是 deterministic relation parser；在原始矩阵中 Origin/Repair-rule 均为 0.29810，Repair-model 为 0.28020。Repair-model 的 JSON 解析有效率 46/49；规范化后的语义 F1 不能等同于后端得分，3 个 families 仍有冠词/原生实体名不匹配。
- 三个 scheme 在确定性 mirrored boxes 上全部 Δ=0；这直接说明文本 adapter 没有改变官方 unsigned geometry。若要修复方向符号，需要另做 signed backend ablation，不能把它冒充本轮 LoRA 收益。
- real video mirror 的水平子集为 600 videos / 30 families，三方案均 Δ=+0.00560；垂直子集为 380 videos / 19 families，Origin/Rule 约 −0.01207、Model 约 −0.01830。这些是检测器不等变性信号，不是有向语义准确率。
- 没有全量人工 direction ground truth；实际案例还发现 GRiT 可能把苹果茎识别为 banana。不能以“唯一框已检测”替代视觉真值。

### 证据与边界

最新方法与矩阵证据在 H100 项目 /root/wenbiao_zhao/vbench-prompts-compile-git：docs/deterministic-experiments-report.md、docs/EXECUTION-2026-09-20.md、scripts/predict.py、scripts/score_matrix.py。旧 metrics/spatial-relationship/IMPLEMENTATION_REPORT.md 中的 signed ordered geometry、detector-aware gate fixture 和 0.0122/0.0694 表格不再作为当前 Repair 依据；待初稿确认后再处理历史方法标记。

---

## 7. Multiple Objects

### Origin 实现方法

当前正式矩阵的 Origin 数据流为：

~~~text
prompt + official multiple_objects target
  -> official entity list / object string
视频 -> 16 个 midpoint frames -> GRiT frame label cache
  -> 每帧检查所有 target entities 是否同时出现
  -> hard conjunction（全对象合取）
  -> 16 帧均值 / video mean
~~~

Origin 使用锁定 GRiT label set；box 和 confidence 不进入最终实体合取。980 个适用原始视频的 Origin 均值为 0.28431。

### Origin 问题

主要问题不是文本 target 是否能被 LLM 解析，而是后端 hard conjunction 与 GRiT 检测/遮挡共同决定结果：任一目标在一帧未进入 label set，该帧就与“两个目标都不可见”同样记 0；partial visibility、目标置信度和漏检原因不进入最终分数。当前 Objects 矩阵也显示，原始实体接口本身已足够一致，主要风险在视觉检测与遮挡有效性。

### Repair 方法

当前 Repair 是共享冻结 Qwen3-8B backbone 上的 Multiple Objects LoRA v6。它是语义接口 Repair，不是另一个 detector，也不使用旧的 ROI threshold=0、confidence SoftMin 或四象限 aggregation。数据流为：

~~~text
prompt（只给文本）
  -> AdapterRouter: task=objects
  -> 共享冻结 Qwen3-8B + objects LoRA v6
  -> JSON entities[]（不限制为两个对象）
  -> article / modifier / alias canonicalization
  -> 对齐冻结 GRiT native labels
视频 -> 同一份 16 帧 GRiT frame labels cache
  -> 每帧 all(entities in labels) hard conjunction
  -> 16 帧均值 / video score
~~~

因此 Repair-model 只替换“prompt 如何变成实体列表”；计数、检测去重、逐帧合取和最终聚合都由确定性代码完成。它可以表达超过两个实体，但官方后端只对可对齐的 native labels 计分。Objects v6 使用 7,688/811 train/dev、固定 900 steps；49/49 个独立 matrix families 的 JSON 接口有效。

### 实验设置（反事实构造）

最终矩阵保留 980 个 Objects endpoints，围绕 target、equal-area background、non-target occlusion 做四级遮挡/可见性变换。11,760 项变换全部尝试，4,716 项完整、7,044 项 partial-valid；188,160 个 frame slots 中，88,084 有效，49,572 个 target 未定位，24,650 个等面积 background 不可用，17,142 个 non-target 未定位，8,712 个 non-target mask 相交。缺失不删除，也不把 partial-valid 当作 complete。

独立模型复核保留 505 项完成、475 项几何不完整；其中 275 项被模型复核为全帧 target 不可见，142 项满足严格 isolation removal。全部 human_reviewed=false，不能称人工金标。

### 实验表（test；每格为 base → 反事实）

| 条件 | Origin | Repair-model（Qwen3-8B + objects LoRA v6） |
| --- | ---: | ---: |
| 原始矩阵均值（980） | 0.28431 | 0.28431 |
| 全计划重度 target occlusion | 0.28431 → 0.01754；Δ −0.26677，CI [−0.31001, −0.22500] | 0.28431 → 0.01754；Δ −0.26677，CI [−0.31001, −0.22500] |
| 两侧完整的 505 videos | 0.41423 → 0.02772；Δ −0.38651，CI [−0.45683, −0.31838] | 0.41423 → 0.02772；Δ −0.38651，CI [−0.45683, −0.31838] |
| 严格 isolation + positive base（115 videos） | 115 中 114 个下降至少 0.03 | 115 中 114 个下降至少 0.03 |

三种 scheme 的 Objects 视频/条件逐项相同；上表的下降主要是同一 GRiT 后端和遮挡构造的行为，不是 Qwen adapter 带来的视觉增益。

### Repair 相关实验与消融

- Origin、Repair-rule、Repair-model 在全部 12,740 个 Objects video/condition 上逐项相同；没有文本 adapter gain。Repair-rule 的实体拆分与 Repair-model 的 Qwen entities[] 都落到同一 backend entity set。
- 目标遮挡与等面积 background 控制的两侧完整 paired delta：target level 0.5 为 −0.11007，CI [−0.20417, −0.03634]（67 videos / 29 families）；level 0.8 为 −0.35771，CI [−0.48692, −0.23983]（47 / 25）；level 1.0 为 −0.40500，CI [−0.59615, −0.22656]（25 / 16）。这些是选择性完整子集，不能外推到 980 个 endpoint。
- 严格 target isolation 下另一对象标签保留率为 0.83311，等面积 background 对照为 0.99531；这说明目标遮挡可能伴随 detector side effect。两者是各自有效帧条件率，不是人工可见性因果真值。
- 四级完整 target gradient 在容差 0.03 内的描述性单调比例为 0.78020，CI [0.72797, 0.83366]（505 videos）；不能把它写成严格语义通过率。natural preference 和 human semantic accuracy NOT RUN。

### 证据与边界

最新方法与矩阵证据在 H100 项目 /root/wenbiao_zhao/vbench-prompts-compile-git：docs/deterministic-experiments-report.md、docs/EXECUTION-2026-09-20.md、scripts/predict.py、scripts/score_matrix.py。旧 metrics/multiple-objects/IMPLEMENTATION_REPORT.md 中的 threshold=0、SoftMin Repair 和旧 visibility 表不再作为当前方法依据；待初稿确认后再处理历史方法标记。

---

## 8. Background Consistency

### Origin 实现方法

Origin 使用 CLIP ViT-B/32 的 official per-frame cosine 和当前 VBench global/transition aggregation，对整帧编码，没有独立的 background evidence unit。

数据流为：

~~~text
视频 -> 每个 clean / counterfactual 版本独立定位 foreground
     -> foreground mask
     -> CLIP background patch tokens
     -> all-pairs cosine + fixed denominator + gain
     -> Repair background score
~~~

### Origin 问题

整帧 CLIP 同时读取主体和背景：主体变化会污染 background score，背景变化也会通过主体上下文、构图和 patch interactions 改变 embedding。因此 background consistency 会镜像 subject consistency 的 nuisance。

### Repair 方法

当前默认 Repair 是 patch_frame_calibrated：每个实际 clean/反事实版本独立使用 CocoSubjectBoxDetector + MobileSAM 得到 foreground mask，从同一 CLIP 的 background patch tokens 做 pooling，使用 all-pairs 同精度余弦、固定 denominator 和 gain=1.75。构造掩码不传给 scorer；没有 foreground retains full background 的作弊路径。

最新构造仍需区分两层：背景干预构造使用 SegFormer-B0 ADE20K foreground union + GrabCut；评分 localizer 独立用 Mask R-CNN/MobileSAM。新 104-class construction 及新 localizer 没有被 promotion 到冻结默认。

### 实验设置（冻结 holdout 与当前暂停协议）

冻结 holdout：1040 candidates，188 accepted，852 rejected，0 runtime failure，1504 versions。它是当前可引用的完整协议。

后续 104-class construction：680 dev inputs 重跑，10 条人工审核干预 Origin/现有 Repair MAE=0.011578/0.003054，未达到新目标 Origin≥0.10；new localizer paused，不能替代当前冻结 holdout。

### 实验表（冻结 holdout；每格为 clean base → background counterfactual）

| 条件 | Origin | Repair patch_frame_calibrated |
| --- | ---: | ---: |
| full/background intervention | 0.946845 → 0.934706；MAE 0.016240 | 0.962999 → 0.959592；MAE 0.006733 |
| natural preference | 52.12% | 61.22% |

自然 preference 提升 +9.10 pp；background response 88.80% CI 见冻结报告。该数字支持当前冻结 holdout 下的改善，不代表新 104-class 构造已通过。

### Repair 相关实验与消融

- 当前 Repair 的关键核验是独立逐版本定位、background patch pooling、all-pairs 聚合、固定 denominator 和 gain=1.75；没有将其他候选实现提升为默认方法。
- 10 条最新人工审核样本中 Origin MAE 0.011578、frozen Repair 0.003054、Caption 0.002704；new localizer 未 promotion。
- visual audit 发现 8 个错误；188 accepted 中 29 个 Repair full >0.01，最大 0.184038，说明定位质量仍是残差来源。
- 独立定位在每个实际版本重新执行，构造 mask 不进入评分，是当前 protocol 的关键。

### 证据与边界

实现：[metrics/background-consistency/IMPLEMENTATION_REPORT.md](../metrics/background-consistency/IMPLEMENTATION_REPORT.md)；冻结实验：[background_holdout_20260920.md](counterfactual-reports/background_holdout_20260920.md)；阶段暂停：[background_repair_checkpoint_20260920.md](counterfactual-reports/background_repair_checkpoint_20260920.md)。冻结 holdout 已完成；新 construction/localizer paused。

---

## 9. Object Class

### Origin 实现方法

Origin 直接调用锁定上游 compute_object_class，使用 raw object string、GRiT detection 的 label set 和 frame-weighted upstream output；上游丢弃的行保留为 null / dropped status，不在汇总中伪造为 0。

数据流为：

~~~text
prompt / object query
  -> Origin upstream query
视频 -> GRiT instances / labels
  -> exact membership + frame denominator
  -> score 或 null / dropped status
~~~

### Origin 问题

原路径主要是 exact string membership，实例 confidence、box 和别名证据未进入最终判定；采样帧分母和上游异常处理也容易把“没有结果”与“视觉判负”混淆。该问题在 controlled uppercase/alias 构造上可观察，但不等价于自然语义错误率。

### Repair 方法

Repair 保留完整 GRiT instances、box、confidence 和 sampled-frame denominator；使用冻结 deterministic label vocabulary aliases，分别记录 exact_hit、exact_miss_alias_hit、no_detection、other_class_detected、runtime_failure。缺配置、模型失败、unsupported query 和部分覆盖保留 null/status；子集均值只作为诊断。

Object/Color 共用 Qwen3-8B semantic compiler，但 Object 的视觉 Repair 不是由 LoRA 直接打分；object query 先由 deterministic/base/lora 编译成冻结 JSON，再由独立 GRiT 视觉路径消费。

Repair 数据流为：

~~~text
prompt / object query
  -> deterministic 或 Qwen3-8B semantic compiler
  -> canonical object JSON
视频 -> GRiT instances / labels
  -> exact / alias evidence + full denominator
  -> score、coverage、failure status
~~~

### 实验设置（反事实构造）

候选按 normalized prompt hash 固定分组。Object 计划 5 dev / 20 test，实际接收 2 dev / 14 test，拒收 9；63 个请求（原串、大写、首个可用 alias、不存在类 proxy），63/63 两后端有输出、0 runtime failure。反事实只改 dimension_metadata.object_class.object，视频字节和原 prompt 不变。

### 实验表（test；paired median，单格为 base → 反事实）

| 反事实 | Origin | Repair |
| --- | ---: | ---: |
| uppercase target | 1.0 → 0.0；中位绝对变化 1.0，CI [1,1] | 1.0 → 1.0；0，CI [0,0] |
| first frozen alias | 中位绝对变化 1.0，CI [1,1] | 0，CI [0,0] |
| absent class proxy | 0 → 0 | 0 → 0 |

natural Object：1580 videos、2370 pairs、79 prompts，test 1410；deterministic/LoRA=0.3851，base=0.2638，base delta −0.1213，95% CI [−0.1752, −0.0716]。video small queue deterministic/LoRA=0.991071，base=0.633929。不能把它写成 human semantic accuracy。

### Repair 相关实验与消融

- deterministic、base、LoRA 三种 query compiler 的原始视频消融中，14 个 test 原始视频的均值都为 0.991071，没有 LoRA 增益证据；LoRA 训练 300 steps。
- 63/63 trace replay 和 0 runtime failure 证明执行协议完整，不证明 natural object semantics。
- human review=0/200；silver/reference agreement 只作为参考一致性，不能当人工标注精度。

### 证据与边界

实现：[metrics/object-class/IMPLEMENTATION_REPORT.md](../metrics/object-class/IMPLEMENTATION_REPORT.md)；两维方法与配置：[object-color-repair.md](object-color-repair.md)；主表：[object_color_repair_20260920.md](counterfactual-reports/object_color_repair_20260920.md)。controlled 已完成；natural human semantic truth 未完成。

---

## 10. Color

### Origin 实现方法

Origin 直接调用上游 compute_color。当前 Repair 的输入保留两路 box/order 信息，并做 one-to-one IoU association；颜色判定使用词边界、否定和 object scope。Official undefined rows 会被上游 drop；Repair 使用 full sampled-frame denominator 并保留 ambiguous/unbound/unsupported status。

两条路径的数据流为：

~~~text
prompt / object + color query
  -> Origin raw color string 或 Repair semantic/binding compiler
视频 -> GRiT boxes + labels
  -> one-to-one IoU association
  -> word-boundary / negation / object-scope predicate
  -> full-denominator score 或 explicit status
~~~

### Origin 问题

颜色词和对象的 binding 可能依赖 box 顺序、遮挡和首实例；全局字符串命中不能证明颜色属于目标对象。Official 的 undefined/drop 还会改变有效分母，不能把返回为空与视觉判负混在一起。

### Repair 方法

Repair 的 binding、binding_lexical、repair 是分开的 ablation。核心 repair 是目标 object 的唯一 box 关联、确定性颜色谓词（word boundary、negation、object scope）和全帧分母；crimson/navy/maroon 等没有被未经验证地折成 red/blue。歧义不强行绑定。

### 实验设置（反事实构造）

候选计划 5 dev / 20 test，实际接收 2 dev / 5 test，拒收 18；42 requests。Color construction 用独立定位和施工 mask 生成 100/75/50/25/0% nested visibility；mask 只用于构造，评分只读视频和 query。Official 返回 33/42，9 dropped_by_official；Repair 42/42。

### 实验表（test；operational response，单格为 base → 反事实）

| 反事实 | Origin | Repair |
| --- | ---: | ---: |
| five-level visibility | 有定义档位中最大偏离中位 0，CI [0, 0.6]；另有 9/42 undefined/drop | 100% → 0% 中位降分 1.0，CI [0.625, 1.0] |
| synonym control | 0 个有效 Official paired rows | 5/5，Δ=0，CI [0,0] |

Repair 的 5 个 test bases 均五档严格单调；两个 dev bases 的中位降分 0.5，CI [0,1]，仅 1/2 严格单调，dev gate 未通过。由于 Official 有 dropped rows，不能把两列当成同一定义域的 raw score paired CI。

### Repair 相关实验与消融

- binding-only test 条件偏离中位 0，CI [0,0.4]；binding+lexical 中位 0.125，绝对 CI [0,1]，不能单独称稳定性通过。
- binding-only / binding+lexical 是同一真实 GRiT trace 的公式重放，不是两次额外 GPU Official 测量；原始 tuple 重放与 true Official 42/42 空值一致，33 个有效值误差为 0。
- Color LoRA 训练 300 steps，deterministic/base/LoRA 的四级原始 query 视频分数相同，无视频增益证据。
- human review=0/200；natural Color NOT RUN。

### 证据与边界

实现和复现：[object-color-repair.md](object-color-repair.md)；主表：[object_color_repair_20260920.md](counterfactual-reports/object_color_repair_20260920.md)；审查：[object_color_repair_20260920.review.md](counterfactual-reports/object_color_repair_20260920.review.md)。controlled binding 有结果，natural、人类真值和完整 synonym Official control 未完成。

---

## 11. Temporal Style

### Origin 实现方法

当前 package 是 M1 scaffold：vbench 和 audit 都返回 status=not_implemented、score=null；没有导入 ViCLIP/BPE，也没有下载 checkpoint。style text extraction 和 ranking 仍未完成。

### Origin 问题

这里不是已测出的数值 bias，而是实现、checkpoint 和 parity 尚未冻结。不能把计划中的 Temporal Style scorer 当作已有 Origin 实验。

### Repair 方法

没有已冻结的 Repair representation、temporal style counterfactual 或 calibration；设计假设不作为实验结果。

### 实验设置（反事实构造）

没有可运行的 counterfactual family；候选数、实际评分数、失败数、CI、人类偏好和后续验证均没有可报告结果。

### 实验表

| 项目 | Origin | Repair |
| --- | --- | --- |
| score | NOT RUN | NOT RUN |
| candidates / scored / failed | NOT RUN | NOT RUN |
| ablation / natural / human | NOT RUN | NOT RUN |

### Repair 相关实验与消融

已有 prompt token audit 只能作为输入诊断，不能称 Repair ablation。当前 token length audit min/median/max=6/13/37，10/30 超过 30 token；ViCLIP/BPE/checkpoint parity、style construction 和 natural preference 全部 NOT RUN。

### 证据与边界

实现：[metrics/temporal-style/IMPLEMENTATION_REPORT.md](../metrics/temporal-style/IMPLEMENTATION_REPORT.md)。这一维必须保持 NOT RUN。

---

## 汇总表（Dynamic已更新；Motion暂空；其余为旧初稿快照）

Repair列统一使用base → 反事实；Dynamic为2026-09-23当前450组，Motion继续留空，其余维度本轮未重核，不作为当前论文数值的替代。

| 维度 | Origin | Repair（base → 反事实） | 敏感性 / 不变性实验 |
| --- | --- | --- | --- |
| Dynamic Degree | 0.680000 → 0.857778 | 0.629251 → 0.623035 | 8px局部纹理抖动不变性；450源/900CF，MAE0.177778→0.021731；批量改善不等于逐片不变 |
| Motion Smoothness | 暂空 | 暂空 | 待确认后补写 |
| Subject Consistency | 0.936417 → 0.816382 | 0.951413 → 0.941724 | 背景 MAE 0.120035 → 0.010596；独立 Mask R-CNN + MobileSAM 支持平均稳定性改善，严格 ≤0.01 和单帧补充未完成 |
| Scene | 0.30594 → 0.05406（strict synonym） | 0.69094 → 0.69469（strict synonym） | Origin synonym sensitivity；独立 Qwen3-8B Scene model 近似不变（Δ +0.00375，CI [−0.01188,+0.02188]），但 caption evidence 不是人工视觉金标 |
| Human Action | 0.88917 → 0（same-class synonym） | 0.86500 → 0.37833（same-class synonym） | Origin 文本 target 与 UMT top-5 不兼容；Qwen action LoRA 同义仅 25/60 正确，35/60 other，未证明 learned repair 有效 |
| Spatial Relationship | base → base（deterministic mirror，Δ=0） | base → base（deterministic mirror，Δ=0） | 三 scheme 共用 unsigned geometry，Qwen spatial LoRA 不能修复方向符号；real-video model Δ −0.00367，CI [−0.01544,+0.00838] |
| Multiple Objects | 0.28431 → 0.01754（heavy target occlusion） | 0.28431 → 0.01754（heavy target occlusion） | 三 scheme 逐项相同，无文本 adapter gain；严格 isolation + positive base 为 114/115 下降至少 0.03，但 human_reviewed=false |
| Background Consistency | 0.946845 → 0.934706，MAE .016240 | 0.962999 → 0.959592，MAE .006733 | 独立 detector + MobileSAM + patch pooling 支持冻结 holdout；新 104-class/localizer paused |
| Object Class | 1.0 → 0.0（uppercase/alias） | 1.0 → 1.0 | deterministic alias/evidence preservation 通过 controlled invariance；natural human truth 未验证 |
| Color | 有效档位中 100%→0 响应近似 0，9/42 dropped | 100% → 0，median drop 1.0 | object binding Repair 在 5 test bases 上单调；dev 1/2，Official synonym control 无有效配对，natural NOT RUN |
| Temporal Style | NOT RUN | NOT RUN | package 仅 scaffold |

## 当前结论与待确认项

### 可以先保留的结论

1. Subject 最新 v5 的核心确实是主体定位/隔离：构造用 SegFormer-B0 + GrabCut，评分用独立 Mask R-CNN + MobileSAM。
2. Scene 当前是独立 Qwen3-8B Scene model：prompt + Tag2Text caption → supported/contradicted/insufficient；旧四象限/OpenCLIP coverage 路径不是当前方法。
3. Spatial、Human Action、Multiple Objects 当前都是共享冻结 Qwen3-8B + 独立 LoRA 的 prompt-only 结构化解析器，之后分别接官方 GRiT geometry、UMT top-5、GRiT entity conjunction；LoRA 不替换视觉后端或最终确定性评分。
4. Spatial 的当前结果证明官方 unsigned geometry 的方向盲点，不能用 Qwen 文本 Repair 修复；Multiple Objects 的文本三方案完全一致，遮挡敏感性主要属于 GRiT/构造验证。
5. Human Action 的 Qwen action LoRA 同义泛化仍失败；Repair-rule 的词典内稳定不能冒充模型开放表达能力。
6. Background 当前默认是 patch_frame_calibrated；新的 104-class construction 尚未 promotion。
7. Object/Color 当前默认分别是 deterministic evidence-preserving Repair 和 object-bound color Repair；LoRA 是语义编译路径，不应写成视觉模型已被 LoRA 修复。

### 待确认后再做的仓库清理

- 删除或归档旧方法实现，必须在本文件方法和版本边界确认后进行。
- 更新 AGENTS.md、README.md、docs/EXPERIMENT_INDEX.md 和各维度 README 的入口与“当前方法”指向。
- 将重复的旧方法说明改为历史链接，避免同一维度存在多个无版本标签的“Repair”。
- Dynamic Degree已补齐当前方法和450证据，AGENTS/README/索引已同步；Motion仍留空，其他维度未重核前，不把本初稿作为最终总览或覆盖当前论文。

本文件只是一版待确认初稿；原始报告、代码、冻结输出和旧方法在确认前全部保留。
