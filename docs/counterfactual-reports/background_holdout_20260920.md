# Background consistency：独立测试与交付报告

日期：2026-09-20。**冻结的主方法通过全部 18 项联合检查，已接为默认 `repair`。**
结论限于本报告的官方自然集与自动粗定位干预协议；不声称逐片段完美稳定或定位已达到人工 IoU 80%。
开发过程与全部负结果保留在[开发报告](background_development_20260920.md)。

**后续目测复核（同日）：** [8 条真实构造样本检查](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/counterfactual-reports/background_mask_visual_audit_20260920.md)发现冰川、面包店等已接受项存在背景误选、显著主体漏分或缺乏明确前景。18 项检查没有检验主体的语义正确性；下文的“主体/背景不变”仅由预测掩码的像素检查保障，不能等同于真实主体/背景不变。原数值保留，但纯主体干预及修复机制的解释须受此限制。此复核是诊断选样，不能推断全体失败率；自然偏好实验不受人工糊化构造的影响。

## 直接回答：origin 如何变，repair 如何变

主体糊化而背景不变时，background 分数理应基本稳定；origin 的实际变化方向由实验决定。
完整糊化在这批视频上使两者平均降分，repair 的逐片段绝对变化下降 58.54%。
不能用均值的正负抵消判断稳定性，因此下表同时报告带符号变化、平均绝对变化和绝对变化中位数。

| 糊化位置 | 方法 | clean 均分 | 糊化后均分 | 平均带符号变化 | 平均绝对变化 | 绝对变化中位数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| full | origin | 0.946845 | 0.934706 | -0.012140 | 0.016240 | 0.010227 |
| full | repair | 0.962999 | 0.959592 | -0.003406 | 0.006733 | 0.003404 |
| start | origin | 0.946845 | 0.928828 | -0.018018 | 0.019531 | 0.007983 |
| start | repair | 0.962999 | 0.956387 | -0.006612 | 0.006717 | 0.002134 |
| middle | origin | 0.946845 | 0.938444 | -0.008401 | 0.008645 | 0.004411 |
| middle | repair | 0.962999 | 0.956885 | -0.006114 | 0.006305 | 0.002091 |
| end | origin | 0.946845 | 0.939263 | -0.007583 | 0.008051 | 0.004378 |
| end | repair | 0.962999 | 0.956782 | -0.006217 | 0.006534 | 0.002090 |

完整主体糊化的 repair 平均绝对变化 95% 区间为 [0.004691, 0.008955]；
相对 origin 的配对绝对变化缩减为 0.009507，区间 [0.006183, 0.012641]。
四个位置的 repair 平均绝对变化均低于预先冻结的 0.01。start/middle/end 的配对改善区间也均为正，
完整逐例数据与各自区间保存在 `intervention_per_case.csv` / `statistics.json`，未删除差例。

## 背景变化仍有响应

对照在 start/middle/end 四分之一时窗内，将主体以外的背景换成固定供体场景；主体像素逐位保留。
这是背景时序切换，不能将普通全片糊化本身解释为时序不一致。

| 时窗 | origin 切换后均分 | origin 降分 | repair 切换后均分 | repair 降分及 95% CI | repair 背景降分 > 同位置主体绝对变化 |
| --- | ---: | ---: | ---: | --- | ---: |
| start | 0.797844 | 0.149001 | 0.882064 | 0.080935 [0.074055, 0.087726] | 97.87% |
| middle | 0.877976 | 0.068869 | 0.880444 | 0.082555 [0.075977, 0.089213] | 99.47% |
| end | 0.890224 | 0.056621 | 0.882751 | 0.080248 [0.073362, 0.087324] | 98.40% |

三个时窗平均的背景响应保留 88.80%，95% CI [84.13%, 94.42%]，超过 75% 门槛。
该保留率是三个时窗的平均：start 响应弱于 origin，middle/end 强于 origin，不把平均值描述成每个位置都保留同样比例。
完整主体绝对变化 / 三时窗平均背景降分从 0.177487 降至 0.082868，
配对改善区间 [0.053729, 0.137052]，避免仅以压缩输出尺度冒充修复。

## 独立自然偏好

1,040 条视频、52 个 prompt、1,560 对 background 自身的人工偏好；四种生成器各 260 条视频。
开发/测试按 source prompt 完全分离。Scene 仅用于定位共享媒体路径，没有读取 Scene 的偏好标签。

| 方法 | 正确对数 / 全分母 | 准确率 | 95% prompt 聚类区间 |
| --- | ---: | ---: | --- |
| origin | 813 / 1560 | 52.12% | [48.21%, 55.90%] |
| repair | 955 / 1560 | 61.22% | [57.31%, 65.00%] |

配对提升 9.10 个百分点，95% CI [6.03, 12.18] 个百分点。
视频和配对覆盖率均为 100%，没有运行失败。零差严格作为平局（预测 0.5），与原始 human_label 对比；
不调平局阈值。若有评分失败，该对留在全分母并计为错误。

## 方法、冻结和数据审计

默认 `--audit` / `--audit-variant repair` 现映射到 `patch_frame_calibrated`，实验名为
`patch_frame_all_pairs_calibrated`。沿用官方 CLIP ViT-B/32、全部解码帧和 `clip_transform(224)`。
每个实际版本用 COCO Mask R-CNN（阈值 0.8）提示 MobileSAM，评分背景取前景补集，
将相同裁剪后的背景面积权重池化到同一 CLIP 的 contextualized patch tokens；
使用 fp16 的全部无序帧对余弦，最后 `clamp(1 - 1.75 * (1 - raw), 0, 1)`。
背景不足 5% 的帧对按固定分母零贡献；未检出前景则保留全部 patch。
每个实际干预版本重新定位；评分路径不读取构造掩码，不复用 clean 掩码制造不变性。
公共模型适配位于 `audit-models`，metric 包之间无互相导入。本轮未新增训练或 LoRA。

- 方法及参数冻结：2026-09-20 16:00:16 UTC；启动前见证：16:10:17 UTC。
- [冻结协议](../../configs/background-repair/holdout_protocol_v1.json) SHA256：
  `f04e5db8433b00a6cc9231f9984079f53cc6fbcb96f72cd3a744c5f782c6d4ca`。固定唯一主方法，没有根据测试结果挑选另一消融。
- 正式评分源码在 `/root/wenbiao_zhao/tmp/background-holdout-v1-20260920-src`；
  11 个评分源码、6 个配置的 SHA256 已在评分前固定且复核未变。
- 测试构造共 1,040 候选：188 接受、852 拒收、构造失败 0；接受项覆盖 43 个 source prompt。
  拒收原因可重叠：前景均面积不足 835 次、出现帧比例不足 792 次、背景不足 1 次。
  所有候选与拒收原因保留，未按结果筛样本。
- 构造使用独立 SegFormer 语义前景并集，Gaussian sigma 18、仅向掩码内部羽化 1.5 像素。
  188 × 8 = 1,504 个版本（clean、full/start/middle/end 主体糊化、三处背景切换）。
- 52 个测试 prompt 预先固定 26 个互不共享 prompt 的供体配对；接受项形成 25 个依赖组。
  主体统计按 43 个 source prompt 聚类；背景及响应比值按 25 个 source/donor 组联合重采样。
  自然准确率按 52 个 prompt 聚类；均 10,000 次 bootstrap，seed 20260920。
- 1,040 条 origin 与锁定 VBench 核心函数最大误差 **0.0**；188 条 clean 的原视频/PNG 独立评分
  在全部 19 种方法上最大误差也为 **0.0**。
- 像素重放验证 188/188，拒收核验 852/852；主体糊化的掩码外像素变化为 0，背景切换的主体像素变化为 0。
- 全部 5,088 个评分掩码/特征文件（10,458,605,616 字节）已复核 SHA256；
  188 条预览的 1,128 张 PNG 也逐文件核验。

## 消融与适用边界

以下全为预先列明的消融，不能用其测试成绩替换冻结的主方法。

| 方法 | 自然准确率 | 完整主体糊化平均绝对变化 | 三时窗背景响应保留率 |
| --- | ---: | ---: | ---: |
| `aggregation` | 54.81% | 0.016015 | 147.27% |
| `aggregation_fp16` | 54.74% | 0.016029 | 147.30% |
| `official` | 52.12% | 0.016240 | 100.00% |
| `patch_frame_all_pairs` | 61.22% | 0.005686 | 49.94% |
| `patch_frame_all_pairs_calibrated` | 61.22% | 0.006733 | 88.80% |
| `patch_frame_all_pairs_gain2` | 61.22% | 0.006935 | 101.48% |
| `patch_frame_balanced_gain2` | 61.22% | 0.007313 | 84.21% |
| `patch_frame_official` | 61.15% | 0.006436 | 32.36% |
| `patch_frame_official_gain2` | 61.15% | 0.007907 | 66.93% |
| `patch_global_all_pairs` | 61.03% | 0.003162 | 49.95% |
| `patch_global_all_pairs_gain2` | 61.03% | 0.006324 | 99.91% |
| `patch_global_balanced_gain2` | 60.96% | 0.006347 | 82.85% |
| `patch_global_official` | 61.09% | 0.003263 | 32.90% |
| `patch_global_official_gain2` | 61.09% | 0.006526 | 65.80% |
| `patch_union_all_pairs` | 60.58% | 0.003814 | 51.93% |
| `patch_union_all_pairs_gain2` | 60.58% | 0.007628 | 105.80% |
| `patch_union_balanced_gain2` | 60.71% | 0.007533 | 87.62% |
| `patch_union_official` | 61.03% | 0.003839 | 33.76% |
| `patch_union_official_gain2` | 61.03% | 0.007677 | 69.45% |

全局 patch 平均对照也有明显改善，不能把全部收益归因于掩码或主体分离；patch 经过全局 attention，
仍可能携带主体信息。正向线性校准不会提高无截断样本的自然排序，改善主要来自表示和聚合。
构造背景切换与主体糊化的语义、面积和强度不同，比例只在此固定实验协议内解释。

粗定位符合本轮不逐边界人工审核的工作方式，但不是人工语义确认或 IoU 测量。自动分割可能在
面包店等弱主体场景出现误检；像素自证只能证明改变了指定掩码，不能证明掩码一定是正确主体。
评分 clean 中 57/188 条完全未检出前景，完整糊化后为 87/188；平均有前景帧比例为 36.60% / 14.73%。
独立自然视频中 778/1,040 条未检出前景，这些仍保留在端到端评估中。
前景检测失败与“背景证据不足”不同：自然集共 12/21,060 帧背景不足，干预 clean 为 11/3,654 帧。
固定零贡献/校准规则产生 1 条自然零分和 7 个干预版本零分，未剔除。

**总体通过不等于每条通过。** 完整糊化仍有 29/188 条 repair 绝对变化大于 0.01。
最大差例 `v_8ff0804118129c1786b3`（parking lot / videocraft）从 0.000000 变为 0.184038；
`v_2512b62c1fec505e11af`（aquarium / cogvideo）从 0.961587 降到 0.847952。
额外描述性分层中，clean 与糊化后都至少一半帧检出前景的 16 条，origin / repair 绝对变化为
0.027954 / 0.024219；其余 172 条为 0.015150 / 0.005106。该分层不用于主验收或选样，
也不意味着前景覆盖高的困难片段已普遍解决。61.22% 的自然准确率仍有明显提升空间。

## 运行、入口和复现

远端资产和运行环境使用既有文件，无模型下载，上游 checkout 和 `data/results/splits/runs` 未改写。
- 上游：`/root/wenbiao_zhao/VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。
- 解释器：`/root/wenbiao_zhao/venvs/vbench/bin/python`，Python 3.10.20、torch 2.5.1+cu121。
- CLIP：`/root/.cache/clip/ViT-B-32.pt`；Mask R-CNN / MobileSAM：
  `/root/wenbiao_zhao/models/subject-repair/`，完整 SHA 在冻结协议及资产复核 JSON。
- 原始媒体：`/root/wenbiao_zhao/datasets/vbench-1.0-human-preference/videos`。
- 两张 H100 物理 GPU 2、5；每卡一个隔离 shard，逻辑 `cuda:0`。
- 自然集原视频媒体时长合计 2,418 秒，每视频时长保存在逐视频 JSONL；实际分析全部 21,060 帧。
  两卡自然评分墙钟 382.75 秒，干预评分 498.86 秒。
  每卡自然后接干预，两个阶段有少量重叠；完整评分墙钟 875.40 秒（约 14.59 分钟）。
  这包含独立定位、19 方法、上游 parity 与产物写入，不是单一后端的独占基准耗时。
- 598 项原完整测试之后新增统计核验；最终 **604 passed、3 skipped**，
  `uv lock --check`、锁定同步、CPU torch overlay、9 个 CLI help 和 `git diff --check` 通过。
  3 项跳过属于 dynamic / human-action / spatial 的外部真实模型 parity，用各自 opt-in 环境开关控制；background 检验没有跳过。
- 默认 `--both` 在含前景的 harbor 开发视频上再次真实执行：origin 0.9774856567、
  repair 0.9887475678，和冻结实验的最大误差 0.0。入口提升仅改变别名，未改冻结评分公式。

用户入口（已有模型环境中）：

```bash
background-consistency --both --video /path/to/video.mp4 \
  --model-config configs/background-repair/h100-models.toml --output NEW_OUTPUT
```

完整重跑须使用固定源码快照和现有权重；新的输出目录不得覆盖本次结果。
在 h100-server 设置已有 Python 模型环境和源码路径（每个 shard 分别选空闲物理卡）：

```bash
BG_SRC=/root/wenbiao_zhao/tmp/background-holdout-v1-20260920-src
export PYTHONPATH="$BG_SRC:$BG_SRC/metrics/background-consistency/src:$BG_SRC/packages/audit-core/src:$BG_SRC/packages/audit-models/src:/root/wenbiao_zhao/models/subject-repair/MobileSAM-f706ad9"
export VBENCH_AUDIT_WORKSPACE="$BG_SRC"
export VBENCH_AUDIT_UPSTREAM=/root/wenbiao_zhao/VBench
export CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
source /root/wenbiao_zhao/venvs/vbench/bin/activate
cd "$BG_SRC"
```

评分示例：

```bash
python -m scripts.counterfactual.run_background_holdout \
  --mode natural --protocol configs/background-repair/holdout_protocol_v1.json \
  --num-shards 2 --shard-index 0 --output NEW_NATURAL/shard0
python -m scripts.counterfactual.run_background_holdout \
  --mode interventions --protocol configs/background-repair/holdout_protocol_v1.json \
  --dataset COMPLETE_TEST_DATASET --verification COMPLETE_PIXEL_VERIFICATION.json \
  --num-shards 2 --shard-index 0 --output NEW_INTERVENTIONS/shard0
```

分别用另一张可见 GPU 和 `--shard-index 1` 运行第二 shard。完成后按以下命令重建本次统计：

```bash
uv run --no-sync --group test python -m scripts.counterfactual.analyze_background_holdout \
  --protocol configs/background-repair/holdout_protocol_v1.json \
  --natural-run output/background-repair/background-test-natural-scores-v1-20260920 \
  --intervention-run output/background-repair/background-test-interventions-scores-v1-20260920 \
  --dataset output/background-repair/background-test-construction-v1-20260920 \
  --verification output/background-repair/background-test-verification-v1-20260920/verification.json \
  --freeze-receipt output/background-repair/background-holdout-freeze-receipt-v1-20260920.json \
  --output NEW_ANALYSIS
```

本地结果根为 `output/background-repair/`；远端同名产物位于 `/root/wenbiao_zhao/tmp/`。
远端保留完整 PNG、构造掩码、评分掩码及特征，本地保留全部 manifest、原始评分、统计与预览。

- [主统计及 18 项门槛](../../output/background-repair/background-holdout-analysis-v1-20260920/statistics.json)
- [每片段、每方法、每位置原始变化](../../output/background-repair/background-holdout-analysis-v1-20260920/intervention_per_case.csv)
- [188 条实际主体糊化预览](../../output/background-repair/background-test-preview-v1-20260920/subject_blur_review.html)
- [像素和拒收核验](../../output/background-repair/background-test-verification-v1-20260920/verification.json)
- [特征、评分掩码和资产哈希核验](../../output/background-repair/background-holdout-artifact-verification-v1-20260920.json)
- [冻结评分源码压缩包](../../output/background-repair/background-holdout-v1-20260920-src.tar.gz)
- [冻结构造源码压缩包](../../output/background-repair/background-test-construction-v1-20260920-src.tar.gz)
- [冻结见证](../../output/background-repair/background-holdout-freeze-receipt-v1-20260920.json)
- [默认 CLI parity](../../output/background-repair/background-default-cli-check-v1-20260920/parity.json)
- [逐视频及干预版本的原媒体时间表](../../output/background-repair/background-holdout-analysis-v1-20260920/media_durations.csv)
  （PNG 序列沿用原视频播放时序，没有另行编码或假设固定 GIF 帧率）
- [完整测试日志](../../output/background-repair/background-final-tests-v1-20260920.log)

统计文件 SHA256：`c58287e9c7b87535597647be3fd29954ea20ec97a4d6104935a2bcff10f1c162`。
统计脚本及依赖的源码快照在该结果目录的 `analysis_source_snapshot/`。Qwen/LoRA 与语义 head 人工审核实验为可选分支，本轮 NOT RUN。
