# VBench Audit 实验结果总览与复现索引

状态快照：2026-09-21。本文是**统一导航和结果台账**，不是第二份原始
结果表。每个数字的权威来源仍是对应报告、冻结配置和机器可读产物；本页把
目的、分母、结果、版本和限制放在同一入口，避免把不同协议的分数混在一起。

Dynamic局部纹理抖动新任务另更新至UTC2026-09-23；其余条目仍为上述旧快照。
此前[联合目标与分区消融报告](counterfactual-reports/dynamic_static_jitter.md#dynamic-outer-support)
覆盖固定32源/128输入及63自然原片，191/191重放、零失败。最大SAM区域支持
候选为0.560659→0.678265，涨幅占Origin的62.72%，仍未达到10%；自然偏好仅
覆盖4个有序对，不能宣称真实运动验证完成。该任务不改写下面历史FPS反事实表。

随后按用户授权完成[冻结V-JEPA 2.1＋小连续头试验](counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-probe)：
210训练/60开发验证官方MP4，810份特征完整；两个固定100步小头，无权重解冻或
检查点选择。联合臂验证相对分数0.566153→0.564112，绝对分差较自然臂降33.31%；
原片运动偏好20/23、两种干预各19/23。预定开发门槛通过，**不是最终修复**；
在该开发阶段结束时，原32组新模型测试及正式留出尚未运行；45条校准预留仍未读取。

最新[真实VBench 1.0冻结验证](counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-validation)
已完成：固定已有32组、预留90个MP4的8px反事实及450条自然测试视频，两后端
各848/848、零失败；30个GIF按固定协议NOT SCORED。90组Origin为
0.700000→0.900000，联合Repair为0.517112→0.514804，冻结相对分数的10%涨幅
数值检查通过；不重新训练或调整映射。自然171有序对的concordance为
81.87%→85.96%，但配对改善CI跨零，未证明显著优于Origin。个体降分、120个新构造
旧质量警示、CF偏好仅6有序对均在报告保留。绝对强度/运动类型人工审核NOT RUN，
不改默认、不宣告完整goal完成，也不替换下面的历史FPS协议结果表。

后续[450源扩展及静态诊断](counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-expansion450)：
全部1800输入/后端、0失败，Origin0.680000→0.857778、旧联合Repair
0.518089→0.516319；全部450人类配对现在两端都有CF。与此同时，单图静止
0.456649、8px平移0.442835暴露了运动排序反例，不能只由不变性宣称成功。
用户已明确授权[锚点监督模型重训练](counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-anchored)，
仅用独立于该单例/测试集的DEV210训练、DEV60验证，已实际完成300步。
新头单例静止/8px平移/32px平移为0.008533/0.134990/0.293819，开发三种位移
排序均60/60；但自然偏好20/23→18/23，未过预定≥19/23，**模型整体验收失败**。
首次数值失败保留；独立统计核验通过不等于模型通过。用户随后授权的
[新头450组评分](counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-anchored450)
现已完成：1800/1800、零失败，新Repair **0.171554→0.166466**、MAE0.015694，
原片自然偏好143/171（旧头147/171），两CF为145/146。抖动平均涨幅数值检查通过，
但原开发门槛失败保留，不改默认。Origin使用既有同批锁定结果，新旧头不混用。

最新[模型内尺度监督及用户收尾决定](counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-aligned)：
270官方原片Origin补测、一次300步训练及独立复核均完成；DEV60新Repair
**0.657326→0.651620**，MAE0.010285。用户接受0.6档，不再为贴近Origin调参。
原严格检验10/12通过，尺度差0.142674与自然偏好17/23仍失败，不改写；静止低分
和位移排序保持。该训练交接时450组NOT RUN；用户随后授权的
[当前aligned模型450组](counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-aligned450)
已完成1800/1800、零失败：**0.629251→0.623035**，MAE0.021731；原片及两CF自然
偏好均144/171。1800特征与450编码控制精确一致。批量涨幅通过，但最大单条
降分0.377932、28/900条降分>0.1，逐片不变性仍有失败。Origin复用锁定同批结果，
无训练/尺度修改，不改默认，不把历史旧头450数字冒充当前模型结果。

## 读取规则

2026-09-23论文对接：[Dynamic论文风格方法](VBENCH_DIMENSION_METHODS_PAPER_STYLE.md#1-dynamic-degree)
已补齐；相邻Overleaf方法§2.2.2及Table 1第六行采用当前aligned450结果，
详见[方法、版本与边界](../../overleaf/docs/dynamic-degree-stability.md)。这里归类为
局部纹理8px抖动**稳定性/不变性**；不把连续Repair误写成二值比例，也不替换历史FPS表。

- 反事实主表的唯一权威来源是
  [CONSOLIDATED.md](counterfactual-reports/CONSOLIDATED.md) 及其
  [table2.json](counterfactual-reports/table2.json) /
  [table2.csv](counterfactual-reports/table2.csv)。自然集人工偏好、开发集和
  后续验证不并入反事实 CPA。
- output/ 是被 Git 忽略的逐视频、掩码、特征和运行日志区；本页链接这些本地
  证据，但不把大文件加入 Git。远端 H100 路径在各报告中保留。
- NOT RUN 表示没有实际评分；PARTIAL 表示有产物但不能当全量结果；
  ARCHIVED 表示旧版本或旧协议，不能替代当前默认值；FAIL/REJECT 保留在
  分母说明中，不能改写成零分。
- Origin/Official 是原始 VBench 路径；Repair 是相应报告中冻结的候选后端。
  CI 默认是报告所述的 95% cluster/bootstrap CI；若是点估计或不同统计单位，
  在表中明确注明。

## 1. 范围与入口

本轮 11 维为 dynamic_degree、motion_smoothness、
subject_consistency、scene、human_action、spatial_relationship、
multiple_objects、background_consistency、temporal_style、
object_class、color。overall_consistency 是保留的 legacy/out-of-scope
包，不作为第 12 维；temporal_flickering、appearance_style、
aesthetic_quality、imaging_quality 只完成源码定位，不在本轮范围。

复现入口：

- 工作区规则、冻结边界和测试：[AGENTS.md](../AGENTS.md)；
- 统一命名与包入口：[README.md](../README.md)、[架构](architecture.md)、
  [CLI](cli.md)；
- 各维度 Origin/Repair 方法、反事实构造、逐维实验表和最终压缩表：
  [DIMENSION_METHODS_AND_EXPERIMENTS.md](DIMENSION_METHODS_AND_EXPERIMENTS.md)；
- 论文用 2-panel Scene / Spatial 案例图、排版脚本和证据 provenance：
  [vbench_counterfactual_cases](figures/vbench_counterfactual_cases/README.md)；
- 反事实数据集与七维主证据：[counterfactual-reports/README.md](counterfactual-reports/README.md)；
- 自然集补充实验手册：[supplementary-experiments.md](supplementary-experiments.md)；
- Object/Color 的独立实现与复现：[object-color-repair.md](object-color-repair.md)。

## 2. 反事实实验：全部七维

数据集共 205 个 base、815 个派生 clip，七维均有 Official/Repair 两路完整
覆盖；所有行的 score coverage 均为 100%，本轮没有运行失败被伪装成零分。
以下“候选/评分”指进入该维度的 base/派生 clip；更早的选择拒收和失效前提
见每行报告及 review。

| 维度（目的/协议） | 候选与实际评分；排除/失败 | Official → Repair；Repair−Official 及 95% CI | 有证据支持的解读 |
|---|---|---|---|
| scene（错误场景与目标场景覆盖率单调） | 25 bases / 125 clips；125/125 两路完成；无运行失败。 | 0.3850 → 0.9300；+0.5450，CI [+0.425,+0.680] | CPA 数字很大，但 review 证明表头与自身证据不一致，Official 在 25 个 coverage_000 上全为 0；不作为 Repair 胜利。 |
| subject_consistency（主体区域开头/中间/结尾干预） | 25 / 100；100/100 两路完成；无运行失败。 | pooled 0.5917 → 0.8500；+0.2583，CI [+0.158,+0.350]。按合同拆分：sensitivity Official/Repair 0.9333/0.7833，invariance 0.2500/0.9167。 | 只有按声明的 split-half 报告时，invariance/稳定性部分是可辩护的 Repair 改善；pooled CPA 不能替代两半。 |
| multiplt_object（弱对象逐级遮挡；conjunction_control 单列） | 25 / 150；150/150 两路完成；49 个候选中 24 个因不可检测拒收；published ladder 只保留 25。 | 0.5450 → 0.7850；+0.2400，CI [+0.155,+0.330]。 | 在 occlusion-only ladder 上是干净的有序改善；control predicate 仍单列：Repair 18/25、Official 22/25 满足 control ≤ full occlusion。弱目标面积跨 62 倍，面积归一化实验 NOT RUN。 |
| motion_smoothness（局部重复/跳帧/反转造成的 jerk） | 25 / 125；125/125 两路完成；无运行失败。 | 当前报告所列方向感知旧重评分：0.8300 → 0.8800；+0.0500，CI [−0.015,+0.110]，即 parity。更早版本 0.8300 → 0.7250，−0.1050，CI [−0.200,−0.020]，ARCHIVED。 | feeb770 版本从 loss 回到 parity；但 4d53fa2 已改默认方向、top-k=3 和 0.5/0.5 权重，当前默认值需要重评分，不能把旧行当现行结论。 |
| dynamics_degree（固定时长改变 FPS，分数应不依赖采样间隔） | 40 / 160；160/160 两路完成；无运行失败。 | CPA 0.8333 → 0.7722；−0.0611，CI [−0.150,+0.017]，CPA 不诊断。主合同斜率 p：Official +0.4908，Repair v2 +0.0187；层级均值 0.0878→0.0910→0.0932→0.0906。 | v2 修复聚合层面的 FPS 依赖，但不是自然偏好替代品；同一 40 bases 选出的 alpha=0.5 在独立 30-base holdout 中 median fps2/fps8=0.9676，仅 20% 落在 ±20%。 |
| human_action（视频字节不变，只改文件名） | 25 / 75；75/75 两路完成；无运行失败。 | CPA 1.0000 → 1.0000；0，CI [0,0]；CV Official 1.4142、Repair 0。 | 这是构造身份结果：Official 目标本来来自 filename，Repair query 从未被改变且视频字节相同；不能解释为有意义的 Repair 能力。 |
| spatial_relationship（水平/垂直镜像，预期 original > flip） | 40 / 80；80/80 两路完成；无运行失败。当前 gating 若重建会扫描 84、拒收 78，published fixture 不满足关系前提。 | 0.3667 → 0.0667；−0.3000，CI [−0.467,−0.133]。 | 不能称 Repair 变差：30 个 test pairs 中 25 个全 tie，Repair 在 80 clips 中 70 个为 0；508/640 frames 缺主体/对象。上游 Official 取 abs(x_distance)，对方向无符号。该 family 的数据前提失效。 |

反事实统计和每一维的独立审查：[SUMMARY.md](counterfactual-reports/SUMMARY.md)、
[CONSOLIDATED.md](counterfactual-reports/CONSOLIDATED.md)，以及
[dynamics_degree.review.md](counterfactual-reports/dynamics_degree.review.md)、
[motion_smoothness.review.md](counterfactual-reports/motion_smoothness.review.md)、
[subject_consistency.review.md](counterfactual-reports/subject_consistency.review.md)、
[scene.review.md](counterfactual-reports/scene.review.md)、
[human_action.review.md](counterfactual-reports/human_action.review.md)、
[multiplt_object.review.md](counterfactual-reports/multiplt_object.review.md)、
[spatial_relationship.review.md](counterfactual-reports/spatial_relationship.review.md)。

## 3. 自然集与人工偏好

下表把人类偏好单独报告。分数是 pair accuracy，不是反事实 CPA；括号内是
报告中的 95% CI。不同维度的自然集规模并不相同。

| 维度/来源 | 候选、实际评分与失败 | Official/Origin | Repair | 变化与适用条件 |
|---|---|---:|---:|---|
| background_consistency holdout | 1,040 videos、1,560 pairs、52 prompts、4 generators；自然两路 1,040/1,040，0 运行失败。干预另有 1,040 candidates，其中 188 接收、852 拒收、0 构造失败。 | 813/1560 = 52.12% [48.21,55.90] | 955/1560 = 61.22% [57.31,65.00] | +9.10 pp，CI [+6.03,+12.18]；仅适用于该 holdout 和冻结粗定位协议。 |
| dynamic_degree P1.1 | 1,440 videos、2,160 pairs；dev/test 72/72 prompts，1,440/1,440 两路成功。另有 30 个 prompt/video-disjoint holdout bases（120 clips）。 | test tie-aware 0.6845 [0.6597,0.7093]；dev 0.6034 | v2 test 0.5690 [0.5426,0.5953]；dev 0.3471 | paired −0.1155，CI [−0.1496,−0.0806]。alpha=0.5 修合同但在自然偏好上显著下降；结论是 audit/refinement，不是 replacement。 |
| motion_smoothness P1.2（pre-4d53fa2） | 1,440/1,440 Official 与 Repair 成功，2,160 pairs；无运行失败。 | 0.6364 [0.6101,0.6636] | 0.3248 [0.2992,0.3512] | paired −0.3116，CI [−0.3473,−0.2760]，Repair 低于 chance；只适用于 feeb770/旧默认，当前默认 NOT VERIFIED。更早补充表中的 0.636/0.395 是旧版本快照。 |
| subject_consistency official extension | 1,440 videos、2,160 pairs；580/860 videos、870/1,290 pairs dev/test；0 运行失败。 | test Origin 58.53% | 原 aggregate Repair 59.61% | +1.09 pp，CI [−0.16,+2.33]；CLS+fallback 后 57.05% 对 Origin 58.45%（−1.40 pp，CI [−3.88,+1.09]）。裁剪版只覆盖 1,262/1,440 videos、1,011/1,290 test pairs，完整分母为 43.41%，不可当全覆盖。 |
| human_action supplementary official | 2,000 videos、3,000 pairs、100 prompts；Official 3,000/3,000，无失败。 | 0.5533 [0.4961,0.6161] | NOT RUN（0 pairs） | 没有自然 Repair 结论。 |
| spatial_relationship supplementary official | 3,240 预期 pairs；Official 2,520/3,240、84/108 prompts 支持，coverage 0.7778；无 Repair 结果。 | zero-margin 0.5054；tie-aware 0.5252 [0.4762,0.5728] | NOT RUN | 支持集本身不满覆盖；不能用 Official 结果修复反事实 family 的无效前提。 |
| object_class natural continuation | 1,580 videos、2,370 pairs、79 prompts；Official、deterministic、base、LoRA 均 1,580/1,580 成功；test 1,410 pairs、dev 960 pairs，无运行失败。 | test 0.3851 [0.3284,0.4418] | deterministic 0.3851（与 Official 完全相同）；base 0.2638 [0.2064,0.3248]；LoRA 0.3851 | deterministic/LoRA Δ=0；base 对 Official −0.1213，CI [−0.1752,−0.0716]。deterministic 是 primary，LoRA 全 cohort 含训练 prompt，仅作描述，不能宣称修复增益。 |
| color natural | manifest/compiled prompts 已生成，但没有 Color score 目录或 pair statistics。 | NOT RUN | NOT RUN | 不能从 5 个 controlled test bases 推断自然人工偏好。 |
| scene、multiple_objects supplementary | 主表显示 --，没有自然人工偏好评分。 | NOT RUN | NOT RUN | 仅有反事实结果；不要把 scene 的 CPA 当自然泛化。 |
| temporal_style | 100 条 prompt 的离线 token audit；无 GPU 视频分数、无偏好标注。 | NOT RUN | NOT RUN | 10/100 条超过 30-token 预算，全部来自同一长场景的 10 种风格；这是 prompt 截断审计，不是 style 效应结论。 |
| overall_consistency | legacy 包保留；无本轮自然评分。 | NOT RUN | NOT RUN | out of scope；与 temporal_style 使用同一估计器且缺少人工条件。 |

自然集原始快照还保留在
[supplementary_main_table.csv](../figures/supplementary_main_table.csv)、
[supplementary_official_metrics.csv](../figures/supplementary_official_metrics.csv)、
[supplementary_repair_metrics.csv](../figures/supplementary_repair_metrics.csv)；
这些是 2026-09-15 的补充快照，不覆盖后续 P1 报告。

## 4. 正式干预与受控 test 结果

### Background：自然偏好改善，但语义掩码仍是限制

正式 holdout 的全片主体糊化：

| 条件 | Origin 平均绝对变化 | Repair 平均绝对变化 | Repair 95% CI |
|---|---:|---:|---|
| full | 0.016240 | 0.006733 | [0.004691,0.008955] |
| start | 0.019531 | 0.006717 | — |
| middle | 0.008645 | 0.006305 | — |
| end | 0.008051 | 0.006534 | — |

full 的配对 MAE 改善为 0.009507，CI [0.006183,0.012641]，相对降低
58.54%；背景 start/middle/end 切换的平均响应保留 88.80%，CI
[84.13%,94.42%]。18 项冻结数值/像素/覆盖检查通过，但后续 8 条目测审计发现
冰川、面包店等样本有背景误选或主体漏分；这证明的是 mask 像素一致性，不是
人工 IoU 或真实语义分离。最大 repair 差例为 0.184038，29/188 条 full
repair 绝对变化 >0.01。权威报告：
[background_holdout_20260920.md](counterfactual-reports/background_holdout_20260920.md)；
开发、空掩码、caption/localizer 和暂停状态见
[background_development_20260920.md](counterfactual-reports/background_development_20260920.md)、
[background_empty_masks_20260920.md](counterfactual-reports/background_empty_masks_20260920.md)、
[background_caption_localizer_20260920.md](counterfactual-reports/background_caption_localizer_20260920.md)、
[background_repair_checkpoint_20260920.md](counterfactual-reports/background_repair_checkpoint_20260920.md)。

### Subject：最新 720 条主交接

v9 固定 72 prompts、每 prompt 10 条、四生成器各 180 条，共 720 candidates：
241 条数值构造通过、479 条拒收、0 运行失败；拒收为全空 103、部分空 330、
面积失败 16、train 类别未映射 30。241 条通过项的 2,169 个版本全部完成主评分，
已知错误 truck 排除后预定主分析为 240 条。开头四分之一：

| 方法 | clean 均分 | 背景糊化均分 | 平均绝对变化（95% CI） |
|---|---:|---:|---:|
| Origin | 0.936417 | 0.816382 | 0.120035 [0.106987,0.134431] |
| v5 Repair | 0.951413 | 0.941724 | 0.010596 [0.008069,0.013583] |

Repair 有 180/240 条 ≤0.01，最坏 0.149715；Origin 有 120/240 条
≥0.10。主体平均降分为 0.099319，CI [0.093839,0.104371]，221/240
≥0.05；背景门槛联合成功 74/240，加主体 ≥0.05 后 70/240。均值符合用户
接受的“0.01 左右”实用要求，但预注册严格 repair ≤0.01 仍是未通过。
中间/结尾/整段的 Origin/Repair MAE 为 0.054624/0.011417、
0.046592/0.012324、0.053310/0.024784（整段仅 149/240 可评分）。

补充单首帧评分在 204/241 条构造完成时按用户要求停止，产物部分保留，
completed=false，不能称全量完成，也不能替代主实验。人类偏好本轮
NOT RUN。权威报告及原始产物：
[subject_stability_20260920.md](counterfactual-reports/subject_stability_20260920.md)、
[v9 协议](../configs/subject-repair/stability_official720_protocol_v9.json)、
[v9 评分协议](../configs/subject-repair/stability_official720_scoring_v9.json)、
output/subject-repair/subject-stability-official720-v9-20260920/。

### Object / Color：controlled claims 与自然扩展分开

Object controlled test 为 14 个大写、13 个别名、14 个不存在类别代理请求；
Official 中位绝对变化均为 1、Repair 均为 0；两路 absent proxy 中位分均为 0。
Color controlled test 为 5 bases、五档可见性：Official 最大绝对偏离中位
0（CI [0,0.6]），Repair 100%→0% 降分中位 1（CI [0.625,1]），5/5
严格单调；但 2 个 dev bases 仅 1 个严格单调，dev gate 未通过。Color 的
Official 同义控制 5 个 eligible 全部无定义配对，不能证明 Official 控制成立；
Repair 5/5 Δ=0。构造与后端覆盖为 Object 63/63、Color Official 33/42
（9 dropped_by_official）、Color Repair 42/42，运行失败/unsupported 为 0。

这两维的完整 candidates 为 Object dev/test 5/20，接收 2/14，拒收 9，
63 requests；Color dev/test 5/20，接收 2/5，拒收 18，42 requests。
Qwen3-8B 两个独立 LoRA 各 300 steps，但人工审核为 0/200，不能声称标签
准确率或 LoRA 提升。权威来源：
[主报告](counterfactual-reports/object_color_repair_20260920.md)、
[方法审查](counterfactual-reports/object_color_repair_20260920.review.md)、
[实现复现](object-color-repair.md)。

## 5. 开发、旧版本和后续验证

以下项目没有被“最好的一次结果”覆盖，全部保留用于解释为什么当前结论
有限。

| 项目 | 实际完成与保留结果 | 当前状态 |
|---|---|---|
| Background 原生 dev | 680 candidates；142 接收、538 拒收、0 构造失败。Origin MAE 0.017240，选定 dev repair 0.005926；自然偏好 52.35%→57.84%，+5.49 pp，CI [+1.67,+9.12]。v2、union、v3–v7 和所有未选候选均留档。 | dev 选择已完成；不把 dev 结果当 holdout。 |
| Background 扩展/定位 | 104 类重构把 full-empty 从 450 降到 89，把数值候选从 7 增至 49；暂停时 12 可用、18 拒收、19 待复核，10 条已评分。10 条 MAE Origin/frozen19/Caption-v2 = 0.011578/0.003054/0.002704；Caption vs frozen natural −1.18 pp，CI [−2.65,+0.20]。 | frozen19 保持默认；新增 localizer 未晋升；49 条完整语义审计未完成。 |
| Background 质量审计 | 8 条目测样本发现语义 mask 错误；71 条 severity paired reversal 的 repair 优势 CI [−1.39,+18.07] pp 跨零。 | 诊断证据，不是总体失败率；继续保留限制。 |
| Subject 小规模 isolation dev | 60 source 中 7 pass/53 reject，实际 7 条运行；Origin median 0.0286，aggregate 0.0316，masked 0.0294，pre-encode crop 0.0072。自然/E0/mosaic NOT RUN。 | 开发探针，不作正式主实验。 |
| Subject quality2 | 咖啡、吉他 2 条；Origin MAE 0.014485→pre-encode crop 0.005910，paired CI [0.000054,0.017097]，主体局部响应 6/6。 | 样本过小，且咖啡 mask 漏身体；不支持总体结论。 |
| Subject 34 条开发/160 follow-up | 34 条通过背景构造，254 拒收；历史 v5 start Origin/Repair 0.131364/0.009979，joint 10/34。follow-up 160 中 60 主分析；1×/2×/4× start MAE 0.080759/0.006353、0.088412/0.006389、0.088890/0.006455，joint 5/60、9/60、7/60；单首帧 0.102377/0.001275、joint 21/60，但主体降分仅 0.031723。 | 旧负结果和剂量/机制对照均是历史证据；不改变 720 主实验。 |
| Dynamic alpha holdout | 排除 counterfactual 40 bases 后，prompt/video-disjoint 30 bases、120 clips；alpha 0/0.5/1 的 median fps2/fps8 为 1.9352/0.9676/0.4838，±20% 命中 13.3%/20.0%/16.7%。 | aggregate validation 支持 0.5；逐 base dispersion 仍大。 |
| Object/Color training | Object/Color LoRA 训练样本 424/465；参考标签完整 JSON 一致性只作工程诊断；silver 有网络失败 4 条被隔离，人工审核 0/200。视频小队列 ablation：Object Official/Repair/LoRA 均 0.991071，Color 1.000/0.975/0.975。 | 不报告人工语义精度，不把 LoRA 视为增益证据。 |
| Temporal style audit | 100 prompts，token min/median/max=6/13/37；10/100 超过 30-token 预算，来自同一长场景的十种风格。 | 离线 CPU audit 完成；视频评分、style 人工关系和自然验证 NOT RUN。 |
| H200 smoke | group 310，4 videos、物理 GPU 4–7、全部成功，墙钟 16.842 s；远端 checkout 1878d39，上游 fd18b3d。 | 仅计时/管线 smoke，不是维度结果。 |
| P2/P3 | 计划中的人工复核和可选后续实验没有完成。 | NOT RUN / out of scope；不从 pooled CPA 推导总体结论。 |

## 6. 版本、协议、输出和证据地图

共同上游 VBench 为 fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490，但各实验
的评分 checkout、渲染 commit 和实际 scoring commit 可能不同；下表优先写
报告明确的 scoring provenance，尤其不把 motion 的 report-render SHA
误当成评分版本。

| 实验 | 代码/协议 | 本地输出与原始证据 | 远端/运行状态 |
|---|---|---|---|
| 七维 VBench-CF | dataset build 66c4a99d…；初始 scoring a044ac9；dynamics v2、motion feeb770 后重算；各维报告首页保留 code SHA。 | docs/counterfactual-reports/{CONSOLIDATED,SUMMARY,table2.*}；逐维报告/review；逐视频树在 output/counterfactual/（scratch/ignored）。 | H100 h100-server：/root/wenbiao_zhao/datasets/counterfactual-vbench/{scores,reports}；七维 score coverage 100%。 |
| P1 natural/control | scripts/evaluate_pairwise_statistics.py，2,000 bootstrap，seed 2026；dynamic alpha holdout 另有固定协议。 | docs/counterfactual-reports/P1_NATURAL_AND_CONTROL_RUNS.md、docs/supplementary-experiments.md、figures/、output/supplementary_20260914/。 | /root/wenbiao_zhao/datasets/natural-preference-runs/；motion/dynamic 的版本限制见报告。 |
| Background holdout | configs/background-repair/holdout_protocol_v1.json，SHA f04e5db8…；默认 patch_frame_all_pairs_calibrated。 | output/background-repair/background-holdout-analysis-v1-20260920/、background-test-verification-v1-20260920/、background-default-cli-check-v1-20260920/。 | /root/wenbiao_zhao/tmp/background-holdout-v1-20260920*；两张 H100，运行完成。 |
| Subject official 1440 | configs/subject-repair/natural1440_protocol_v2.json；上游 fd18b3d；H100 extension 报告保留各源码 hash。 | output/subject-repair/subject-natural1440-*、CLS 候选和 34 条背景分析。 | /root/wenbiao_zhao/tmp/ 同名运行目录；全量 natural 与背景候选完成，整体修复仍有限。 |
| Subject stability 720 | configs/subject-repair/stability_official720_protocol_v9.json + stability_official720_scoring_v9.json；控制入口 run_subject_official_census.py。 | output/subject-repair/subject-stability-official720-v9-20260920/；单帧 partial 输出单独留存。 | /root/wenbiao_zhao/tmp/subject-stability-official720-v9-20260920；主评分完成，单帧按用户停止。 |
| Object/Color controlled | configs/four_dimension/object_color_family_manifest.json；独立包、独立 LoRA；H100 旧 checkout HEAD 9f2cf67…，不等于评分源码唯一版本。 | output/object_color_20260920/；主报告的 main.csv、per_base.csv、execution_manifest.json、verification.json。 | H100 /root/wenbiao_zhao/vbench-audit/output/object_color_20260920/；controlled 队列完成。 |
| Object natural | 当前协议为 output/object_color_natural_20260920/protocol.json，脚本为 scripts/object_color_natural*.py；snapshot/provenance 写入 output。 | output/object_color_natural_20260920/analysis/object_class/{report.json,summary.csv,parity.json}；Color 只有 manifest/compiled，无 score。 | object-complete.json=true，1580×4 methods 全成功；Color NOT RUN。 |
| Temporal/Overall | docs/paper/unaudited-dimensions-triage.md 只记录离线审计；无视频评分协议。 | prompt 文件、token audit 和 legacy 包；没有 GPU output。 | NOT RUN。 |

## 7. 本轮目录整理与冻结边界

本轮采用“建立入口、统一引用、保留原始产物”的整理方式：

- 新增本文件作为唯一总览入口；README 增加入口链接，反事实报告 README
  继续指向 CONSOLIDATED/SUMMARY/table2，沒有创建第二份原始权威表。
- 目录职责固定为：configs/ 协议和 manifest、scripts/ 复现入口、
  tests/ 合约测试、docs/ 报告与索引、figures/ 论文快照、output/
  大型忽略产物。
- data/、results/、splits/、runs/ 在本轮没有移动、删除、改写或重算；
  模型权重、大型媒体、特征和 GPU 日志不入 Git。无法确认用途的旧文件留在原处，
  由本页和各历史报告建立索引。
- 已有未提交的 Object natural 脚本/协议/测试及 temporal-style token audit
  没有被覆盖；它们的结果在本页按实际完成度引用。实验输出仍以 ignored
  output/ 为准，提交只包含可复现代码、协议和文档。

## 8. 结论边界和未完成清单

当前证据支持：

1. Background 在冻结 holdout 上同时降低主体糊化敏感度并保留背景切换响应，
   且自然偏好提高；但语义 mask 目测审计限制“真实主体/背景不变”的表述。
2. Subject v9 在 720 候选中得到 Origin MAE 0.120035、Repair MAE 0.010596
   和主体降分 0.099319；可称达到实用的约 0.01 均值表现，不可称严格
   ≤0.01 通过，也不可隐藏 479 拒收和 70/240 联合成功。
3. Dynamic alpha=0.5 使 FPS 合同在 aggregate/holdout 上更平，但自然偏好显著
   下降；Motion 旧方向修复只到 parity，当前默认未重测。
4. VBench-CF 中可稳健引用的是 Subject split-half 的有限改善和
   Multiple Objects occlusion-only 的有序改善；Scene、Human Action、
   Spatial 的数字不能按 headline CPA 解释。
5. Object/Color 的 controlled test 只证明局部、预定义响应；Object natural
   deterministic/LoRA 与 Official 完全相同，Color natural 尚未运行。

仍未完成或不能宣称：

- Subject 严格 repair ≤0.01、逐例 joint gate、全量单帧和人类偏好；
- Motion 4d53fa2 当前默认的 counterfactual/natural 重评分；
- Background 49 条扩展候选的完整语义审核与新联合目标；
- Color natural score、自然 human preference；Human Action/Spatial Repair
  natural score；Scene/Multiple Objects natural score；
- Temporal Style 的视频/人工 style 关系，Overall 的本轮实验；
- P2/P3 人工验证、冻结 E0 的全套 model/CUDA/weight parity，以及任何基于
  LoRA/自动标签的人工语义准确率。

因此本页的总判断是：这些结果足以交付一个带版本和失败记录的审计台账，
但不能合并成“所有维度 Repair 都优于 Official”的结论。

## 9. 最小复现与核验入口

不启动新实验时，可执行以下不改冻结研究内容的检查：

~~~bash
uv lock --check
uv run --no-sync --group test pytest tests metrics
uv run --no-sync python scripts/validate_supplementary_outputs.py \
  --main-csv figures/supplementary_main_table.csv
git diff --check
~~~

真实模型 parity 需要各报告列出的 H100 环境、权重和源码快照；本机 CPU
测试不替代它。对本页引用的 ignored output，应优先核对对应的
protocol.json、execution_manifest.json、report.json 或 statistics.json
的 complete/status/coverage 字段，再读取汇总数字。
