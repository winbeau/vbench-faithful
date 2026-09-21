# Background consistency：表示修复开发与独立验证

日期：2026-09-20。状态：**开发完成，冻结候选已通过独立联合验收**；见[正式测试报告](background_holdout_20260920.md)。
本报告对应新目标“糊化主体、保留背景”，不替换已发布反事实数据集的历史主表。

## 当前结论

已在开发集选出 `patch_frame_all_pairs_calibrated`：保持原 CLIP 权重，按独立前景掩码的补集
对 patch 特征池化，使用全帧对聚合，固定增益 1.75。142 条官方开发基底的完整主体糊化
平均绝对变化从 **0.017240 降至 0.005926**（约 65.6%），配对改善区间
[0.008609, 0.014151]；start/middle/end 分别为 0.009100、0.008764、0.008985。
自然开发准确率由 **52.35% 提高到 57.84%**，配对差 +5.49 个百分点，区间 [+1.67, +9.12]。
背景切换响应保留 origin 的 86.14%，归一化的主体干扰也有改善，不能把收益只归结为分数缩放。

这些仍是开发结果。方法、源码哈希与验收门槛已在
[`holdout_protocol_v1.json`](../../configs/background-repair/holdout_protocol_v1.json) 冻结，
冻结时间为 2026-09-20 16:00:16 UTC，协议 SHA256 为
`f04e5db8433b00a6cc9231f9984079f53cc6fbcb96f72cd3a744c5f782c6d4ca`。
独立测试包括 1,040 条自然视频、1,560 对偏好，以及 188 条干预基底的 1,504 个版本。
独立检验现已完成并通过 18 项冻结门槛，未按测试结果更换候选或参数。
以下数值仍为开发结果，正式测试结论及失败案例见正式报告。
旧的填灰、低阈值与其他负结果完整保留在下文。

## 实现和输入核验

`metrics/background-consistency/` 已由接口占位接通真实 CLIP 评分。
官方路径保留 VBench `fd18b3d` 的全部帧解码、float32 原始 RGB、
`clip_transform(224)`、中心裁剪和 fp16 逐项余弦计算。
680 条自然开发视频、34 条诊断基底的 origin 均与锁定上游函数逐条对齐，最大误差 **0.0**。
另外，H100 上实际执行了 `--both` 单视频 CLI，两种后端均完成，分数与批量结果一致。
随后用检出了前景的 harbor 片段复核，发现 CLI 默认 cuDNN TF32 设置与实验脚本不同，
repair 相差 0.0003128；已在公共评分入口统一确定性和精度设置。
修复后该片段 origin 为 0.9774856567、repair 为 0.9789047241，均与批量值精确一致。
原始不一致输出保留在远端 `background-cli-foreground-parity-20260920/`，
修复后证据在 `background-cli-foreground-parity-v2-20260920/`。

Repair 以独立 Mask R-CNN 自动框提示 MobileSAM，逐个实际视频版本重新定位。
没有用构造掩码评分，没有将干净视频的掩码代入糊化版本。定位采用用户允许的粗粒度标准，
没有新增逐条人工审核门槛。MobileSAM 与 COCO box adapter 已提取到 `audit-models`，
subject 保留兼容导入；background 不导入另一个 metric。

| 方法 | 表示 | 时间聚合 |
| --- | --- | --- |
| official | 原始整帧 CLIP | 相邻帧与首帧锚点 |
| aggregation | 原始整帧 CLIP | 全部无序帧对 |
| frame_official | 每帧评分主体掩码内填充 RGB 128，保留整幅背景几何 | 官方聚合 |
| frame_all_pairs | 同上 | 全部无序帧对 |
| union_official | 同一实际片段内的评分主体掩码取时间并集，固定抑制这些位置 | 官方聚合 |
| union_all_pairs | 同上 | 全部无序帧对 |

全帧对实现用 float32 归一化/点积和 float64 求均值，官方逐项计算保留 fp16。
这是额外的数值精度差异；极小分差不能单独归因于时间聚合，v3 之后已补同精度聚合对照，见下文。

时间并集会连同主体曾遮挡过的部分背景一起去除，可能损失有效证据。
背景剩余面积小于 5% 的帧按固定分母零贡献处理；前景未检出则背景视图保留整帧并计数。
自然开发集 463/680 条没有检出前景，2 条的时间并集有背景不足帧。
这两种情况不同，不能都称为“空掩码导致失败”。

冻结 E0 清单包含 1,720 条 background 视频、2,580 对 background 偏好。
远端四种生成器各 430 条实际视频已核验存在，CogVideo 原件为 GIF。
旧 E0 路径有冠词、扩展名和生成器目录错误；v1 尝试因此 665/680 条找不到文件，
15 条成功记录和全部失败保留。v2 按 `(prompt, seed, generator)` 使用共享 scene 视频的
官方媒体路径纠正了 1,710/1,720 条映射，原 UID、background 标签和 prompt 分组划分不变。
**只使用 Scene 的视频映射，不使用 Scene 人工偏好标签。**

## 34 条复用诊断：主体糊化

数据来自前一轮 subject 的 288 条固定候选；34 条构造通过、254 条拒收全部保留。
这些是已知 subject 来源的诊断材料，不是官方 background 的独立测试集。
每条 clean 加 full/start/middle/end 的主体、背景糊化，共 306 个实际版本。
评分缓存对应原构造索引 SHA；四个原评分 shard 的索引哈希已逐一核对一致。
每个 PNG 与独立评分掩码的哈希也已核对。

完整主体糊化：

| 方法 | clean 均分 | 主体糊化后 | 均值带符号变化 | 逐视频平均绝对变化 |
| --- | ---: | ---: | ---: | ---: |
| official | 0.947407 | 0.936325 | −0.011082 | 0.021548 |
| aggregation | 0.941957 | 0.928407 | −0.013550 | 0.024977 |
| frame_official | 0.928131 | 0.928734 | +0.000603 | 0.016230 |
| frame_all_pairs | 0.923976 | 0.922337 | −0.001639 | 0.017035 |
| union_official | 0.956881 | 0.948349 | −0.008532 | 0.013095 |
| union_all_pairs | 0.955032 | 0.945201 | −0.009831 | 0.013567 |

例如 `frame_official` 的总体均值几乎不变，但逐视频绝对变化仍为 0.01623；
不能把正负抵消称为稳定。

| 主体糊化位置 | origin 平均绝对变化 | union_official 平均绝对变化 |
| --- | ---: | ---: |
| full | 0.021548 | 0.013095 |
| start | 0.051155 | 0.008487 |
| middle | 0.022351 | 0.006176 |
| end | 0.017294 | 0.004697 |

局部时窗背景糊化时，origin 的均值降分依次为 start 0.062959、middle 0.030766、end 0.026432；
`union_official` 分别为 0.053424、0.053687、0.020239。响应仍存在，但糊化与背景语义
不一致不能等同；全片统一糊化也不必然降低时序一致性。因此后续官方干预实验采用明确的局部背景场景切换对照。

## 官方自然开发集

680 条视频、34 个 prompt、1,020 对 background 人工偏好，运行失败 0，评分覆盖率 100%。
主比较使用所有方法统一的零平局容差，保留三值人工标签；以下均为开发结果。

| 方法 | 准确率 | 相对 origin 的配对变化，百分点 | 95% prompt 聚类区间，百分点 |
| --- | ---: | ---: | --- |
| official | 52.35% | 0 | — |
| aggregation | 52.35% | 0 | [−1.86, +1.86] |
| frame_official | 52.16% | −0.20 | [−1.86, +1.37] |
| frame_all_pairs | 52.75% | +0.39 | [−2.35, +3.04] |
| union_official | 50.78% | −1.57 | [−3.43, +0.10] |
| union_all_pairs | 51.08% | −1.27 | [−3.53, +0.88] |

`frame_official` 的自然表现较接近 origin，但诊断集完整主体糊化改善区间跨零。
在上述 v2 阶段尚无同时满足全部目标的证据；后续候选选择见 v7。
其余 1,040 条视频、1,560 对偏好仍保留为未评分测试集。

## 正在推进的实验

已冻结 [`construction_dev_v1.json`](../../configs/background-repair/construction_dev_v1.json)，
将全部 680 条官方 background 开发视频作为候选：SegFormer 的粗语义主体并集用于构造，
保留背景并糊化主体；同时在 start/middle/end 的四分之一时窗内，把背景替换为另一场景。
供体按同划分、同生成器、同种子的下一字典序 prompt 固定，不读评分或人类标签。
每个干预均做相反区域像素零改变断言，所有候选与拒收保留。
不要求镜像区域或精细人工轮廓。

构造已在 H100 物理 GPU 2 完成，680 条候选中 **142 条接受、538 条拒收、运行失败 0**：
`/root/wenbiao_zhao/tmp/background-native-construction-v1-20260920`。
接受项每条有 clean、4 个主体糊化版本、3 个背景场景切换版本，共 1,136 个版本。
142 条接受项的所有实际 PNG 已通过逐像素重放，538 条拒收原因也已重新计算核验。
验证记录：`output/background-repair/background-native-verification-v1-20260920/verification.json`。
独立逐版本评分已在 H100 物理 GPU 3、4 完成，运行目录为
`/root/wenbiao_zhao/tmp/background-native-scores-v1-20260920/shard{0,1}`。
142 条 clean 仅在完整帧内容、视频、模型和实现哈希一致时复用自然集结果；994 个干预版本重新定位和编码。
两 shard 的完成标志、分数哈希、680 个唯一候选身份和完整方法/版本网格已核验，评分失败为 0。

完整主体糊化结果（142 条、27 个有接受项的 source prompt）：

| 方法 | clean 均分 | 主体糊化后 | 逐视频平均绝对变化 | 配对绝对变化缩减 95% 区间 |
| --- | ---: | ---: | ---: | --- |
| official | 0.949269 | 0.933825 | 0.017240 | — |
| aggregation | 0.944961 | 0.928102 | 0.018620 | 见逐例统计 |
| frame_official | 0.934996 | 0.927674 | 0.016796 | [−0.001054, +0.002158] |
| frame_all_pairs | 0.929586 | 0.923007 | 0.016366 | 见逐例统计 |
| union_official | 0.950374 | 0.937210 | 0.015375 | [−0.000701, +0.005302] |
| union_all_pairs | 0.945713 | 0.931602 | 0.016544 | 见逐例统计 |

`union_official` 的改善约 10.8%，但不足以证明可靠修复。局部主体糊化改善更明显：

| 位置 | origin 主体糊化绝对变化 | union 主体糊化绝对变化 | origin 背景切换降分 | union 背景切换降分 |
| --- | ---: | ---: | ---: | ---: |
| start | 0.021368 | 0.011845 | 0.149559 | 0.128765 |
| middle | 0.010428 | 0.006925 | 0.068470 | 0.063961 |
| end | 0.009149 | 0.006220 | 0.056366 | 0.054325 |

主体糊化用 source prompt 聚类。背景对照的 donor 是开发 prompt 环，不能当成相互独立的
source prompt；分析按 source/donor 图的连通分量一起重采样。接受项形成 5 个分量，
含基底 70、41、20、7、4 条，独立聚类数较少，区间仅用于开发诊断。
union 背景降分的区间依次为 [0.120106, 0.159478]、[0.059223, 0.074213]、
[0.049828, 0.061052]；背景降分大于同位置主体绝对变化的比例依次为 98.59%、94.37%、94.37%。
背景敏感性仍在，但它不能抵消完整主体糊化和自然偏好两项的不足。

评分定位在 clean 的平均有前景帧比例为 37.36%，完整主体糊化后为 17.05%；
完全未检出的片段分别为 46/142 和 55/142。这是粗定位缺失的诊断，非人工 IoU。
未检出的片段仍保留在所有分母中；自动构造标签和像素重放也不等于人工确认主体语义正确。

v3 开发方案记录于 [`development_localizer_v3.json`](../../configs/background-repair/development_localizer_v3.json)：
固定检测阈值从 0.8 改为 0.3，对比 SAM 与粗矩形框、逐帧与时间并集，以及官方聚合与同精度全帧对。
同精度对照保留 CLIP fp16 和官方逐项余弦语义，旧 float32 消融保持原样。
全部 680 条自然开发视频、142 条官方干预基底和 34 条 subject 来源诊断基底均进入该轮；
逐个实际版本独立定位，原始输入重新编码对齐 origin。v3 自然和诊断两部分已完成，均无运行失败，
官方干预部分也已完成；不替换当前 repair 默认值。

| v3 方法 | 自然开发准确率 | 34 条诊断完整主体糊化绝对变化 |
| --- | ---: | ---: |
| sam03_frame_official | 53.73% | 0.033117 |
| sam03_frame_all_pairs_fp16 | 54.22% | 0.024247 |
| sam03_union_official | 51.08% | 0.037597 |
| sam03_union_all_pairs_fp16 | 50.69% | 0.036513 |
| box03_frame_official | 52.94% | 0.023358 |
| box03_frame_all_pairs_fp16 | 52.94% | 0.024563 |
| box03_union_official | 46.67% | 0.033935 |
| box03_union_all_pairs_fp16 | 47.25% | 0.035579 |

对照 origin 为 52.35% / 0.021548。`sam03_frame_all_pairs_fp16` 的自然配对差区间为
[−1.18, +4.90] 个百分点，诊断鲁棒性仍不足。粗框时间并集明显损害自然表现。
在官方 142 条基底上，sam03 的逐帧官方/全帧对绝对变化为 0.017742/0.015425，
时间并集为 0.056993/0.057374；box03 对应为 0.022514/0.021112 和 0.061801/0.062075。
所有低阈值候选都未同时通过开发门槛。
v2 诊断沿用 subject 来源的目标类定位，v3 诊断改为全 19 类定位，因此诊断差异包含类别范围变化；
官方 native 的 v2/v3 均为全 19 类，可在那里比较固定阈值变化。
同精度实现已经在 270 组真实 fp16 CLIP 特征上与逐对标量参考比较，最大误差 0。
随机 float32 CUDA 验证出现约 8.3e−10 的批量计算舍入差异，不宣称任意 dtype/设备逐位相同。

已记录 [`development_patches_v4.json`](../../configs/background-repair/development_patches_v4.json)：
保持 CLIP 权重和 v2 独立掩码，将背景覆盖率作为图像块特征的池化权重，比较整幅 patch 均值、
逐帧背景和时间并集背景。没有把图像填灰；同时保留 CLS origin 和表示/聚合消融。
patch 已经过全局 attention，依然可能受到主体影响，不把它当作保证完全隔离的表示。
v4 三个完整开发部分现已全部测量，运行失败 0。逐帧背景全帧对的原始主体糊化绝对变化为
0.003386，背景响应保留 49.22%；自然准确率为 57.84%。patch 全局平均消融也有较大收益，
因此不能把全部改善归因于定位或掩码。

### 冻结前的尺度与时间聚合校准

v5 对所有 patch 消融统一施加 `clamp(1-2*(1-score), 0, 1)`，以恢复背景响应尺度。
这同样放大主体干扰；不能人为改善干扰/背景响应比值。frame 全帧对的完整主体绝对变化为
0.006773、背景响应保留 98.45%，但 start/middle/end 为 0.010400/0.010016/0.010268。
v6 测试唯一的 0.5/0.5 官方/全帧对混合，改善 middle/end，start 仍为 0.011317，未通过更严格的位置门槛。
两轮结果均归档，未读取测试分数。

v7 保留均匀全帧对，固定增益 1.75，在开发集通过全部位置的 0.01 门槛。最终所选方法：

| 项目 | origin | 冻结候选 |
| --- | ---: | ---: |
| 142 条 clean 均分 | 0.949269 | 0.970008 |
| 完整主体糊化后均分 | 0.933825 | 0.964775 |
| 完整主体糊化平均绝对变化 | 0.017240 | 0.005926 |
| 34 条诊断完整主体糊化平均绝对变化 | 0.021548 | 0.006221 |
| 自然开发偏好准确率 | 52.35% | 57.84% |

全部真实版本独立定位的来源不变；每次校准保留原始数值、所有失败与源码快照。
middle/end 相对 origin 的改善区间仍跨零，不能把完整主体糊化的显著改善推广为每个位置均显著改善。
最终方法的开发集背景响应保留区间为 [81.82%, 95.14%]；归一化干扰比值由 0.188486 降到 0.075216，
配对改善区间 [0.076165, 0.133411]。

### 输出缩放检查

新增 [`holdout_gate_addendum_v1.json`](../../configs/background-repair/holdout_gate_addendum_v1.json)，
在测试评分前补充防止统一压缩分数冒充修复的条件：背景响应均值至少保留 origin 的 75%，
且“完整主体绝对变化 / 三个位置背景降分均值”的配对改善区间下界大于 0。
原协议的绝对变化、自然非劣和覆盖门槛均保留；不删除分母非正的重采样。
单元测试验证统一缩放的对照归一化改善为 0。
历史 v2 union 的该比值为 0.186707，origin 为 0.188486，改善区间 [−0.009072, +0.013025]，
仍不能证明有选择地消除了主体干扰；背景响应保留 90.03%。

独立测试的构造规则另行固定在 [`construction_test_v1.json`](../../configs/background-repair/construction_test_v1.json)：
52 个测试 prompt 按排序两两互为 donor，形成 26 个互不共享 prompt 的组，其余构造条件不变。
该数据准备未读取评分或偏好标签；最终方法与阈值随后已在查看测试分数前冻结。
1,040 条测试候选已构造完毕：188 条接受、852 条拒收，构造失败 0；188 条全部 PNG 已通过像素重放，
852 条拒收原因也已核验。测试评分与联合统计随后已完成，见[正式测试报告](background_holdout_20260920.md)。

## 产物和复现

- [开发方法协议](../../configs/background-repair/development_protocol_v2.json)；
  [纠正后的自然视频清单](../../configs/background-repair/natural1720_manifest_v2.jsonl)。
- [逐例主体糊化预览](../../output/background-repair/background-region-preview-v2-20260920/subject_blur_review.html)：
  34 条各取首/中/尾帧，共 204 张实际无损 PNG。
- 诊断统计：`output/background-repair/background-region-analysis-v2-20260920/statistics.json`，
  同目录 `region_per_case.csv` 保留所有位置与方法。
- 自然统计：`output/background-repair/background-natural-dev-analysis-v2-20260920/statistics.json`，
  同目录 `natural_dev_scores.jsonl` 保留 680 条结果。
- 官方干预统计：`output/background-repair/background-native-analysis-v1-20260920/statistics.json`，
  同目录 `native_per_case.csv` 保留所有 142 条、4 个位置、6 种方法的逐例变化。
- v3 已完成部分：`output/background-repair/background-natural-candidate-analysis-v3-20260920/statistics.json`、
  `output/background-repair/background-diagnostic-candidate-analysis-v3-20260920/statistics.json`。
- 缩放检查：`output/background-repair/background-native-scale-analysis-v1-20260920.json`；
  数值验证：`output/background-repair/background-precision-validation-v3-20260920.json`。
- 完整 v3/v4 联合统计：`background-joint-analysis-v{3,4}-20260920/statistics.json`；
  校准、混合、最终尺度统计：`background-calibration-analysis-v5-20260920`、
  `background-balance-analysis-v6-20260920`、`background-scale-analysis-v7-20260920`，均在 `output/background-repair/`。
- [官方 142 条实际主体糊化预览](../../output/background-repair/background-native-preview-v1-20260920/subject_blur_review.html)：
  首/中/尾帧共 852 张 PNG，包含文件来源与哈希。
- 远端不可变代码快照：`/root/wenbiao_zhao/tmp/background-dev-v2-20260920-src`。
  诊断与自然运行目录分别为 `background-region-dev-v2-20260920`、
  `background-natural-dev-v2-20260920/shard{0,1}`，均位于同一 `tmp/`。
  权重、源码逐文件 SHA、物理 GPU、起止时间与环境记录在各自 `run.json`。

在已有模型环境与 `PYTHONPATH` 下：

```bash
python -m scripts.counterfactual.run_background_development \
  --mode natural_dev --num-shards 2 --shard-index 0 --output NEW_OUTPUT/shard0
python -m scripts.counterfactual.run_background_development \
  --mode region_diagnostic --dataset EXISTING_SUBJECT_DATASET \
  --cached-scores EXISTING_INDEPENDENT_SCORING_RUN --output NEW_REGION_OUTPUT
python -m scripts.counterfactual.build_background_interventions --output NEW_NATIVE_DATASET
python -m scripts.counterfactual.verify_background_interventions \
  --dataset COMPLETE_NATIVE_DATASET --output NEW_VERIFICATION --workers 4
python -m scripts.counterfactual.score_background_interventions \
  --dataset COMPLETE_NATIVE_DATASET --natural-run COMPLETE_NATURAL_RUN \
  --num-shards 2 --shard-index 0 --output NEW_NATIVE_SCORES/shard0
python -m scripts.counterfactual.analyze_background_native \
  --run COMPLETE_NATIVE_SCORES --dataset COMPLETE_NATIVE_DATASET --output NEW_ANALYSIS
python -m scripts.counterfactual.run_background_candidates \
  --cohort native --previous-run COMPLETE_NATIVE_SCORES \
  --dataset COMPLETE_NATIVE_DATASET --output NEW_CANDIDATE_SCORES/shard0
```

各 shard 使用不同输出目录与可见 GPU；输入、版本和完整覆盖校验不通过时拒绝合并。
验证与评分代码快照为 `/root/wenbiao_zhao/tmp/background-native-run-v1-20260920-src`。
最终全套检查为 604 passed、3 skipped，包含池化、校准、冻结协议、统计基准与 CLI 接线测试。
`uv lock --check`、`uv sync --locked --group test`、CPU torch overlay 和
原八个入口加 background 的 9 个 `--help` 均通过。未下载任何模型权重，未改写冻结研究目录或上游 checkout。
