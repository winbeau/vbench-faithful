# Background repair 阶段总结（2026-09-20）

**状态：按用户“先这样”指令暂停实验，未达到新的联合目标。** 本次仅整理、验证并提交现有工作，
不继续启动实验。当前默认仍为 frozen19 背景 patch 池化与全帧对聚合；新的定位分支没有晋升默认。

## 构造范围与扩类效果

使用 background 自己的官方 680 条开发视频与 background 偏好标签，未把 subject 视频换作本维度输入。
复用 SegFormer + GrabCut：对主体掩码 M 做全片 Gaussian blur，主体外像素保持不变；
subject 的对偶构造则糊化 1−M。主体面积范围仍为平均 15%–25%，允许粗边界，不扩大掩码凑面积。

| 同一批 680 条输入 | 原 10 类 | 扩展 104 类 |
|---|---:|---:|
| 全空构造掩码 | 450 | 89 |
| 平均面积 15%–25% | 12 | 75 |
| 全部数值构造条件通过 | 7 | 49 |
| 运行失败 | 0 | 0 |

扩展加入家具、器具、设备和独立建筑等类别，仍排除 46 类场景表面。
同时修正中位数全为零时误选从未检出类别的问题。89 条剩余全空中，62 条没有配置内检出、
27 条细化全空，均不代表原视频空白。全部 680 条已完成独立数值重放；这不等于语义正确率验证。

按用户要求间隔逐条看图。暂停时，49 条候选中 **12 条粗审可用、18 条拒收、19 条待复核**。
其中 10 条已经评分；刚接受的浴室反光板 `v_ee7856ce8e90297a6f80` 与 ModelScope 校园建筑群
`v_b068eafd5a88f2e4321a` 尚未评分，不能计入下表。每个决定均保留理由和原图/构造预览哈希。

## 已完成的十条干预结果

这十条来自八个 prompt，是分阶段质量审核的开发子集，不是 680 条的代表性估计或新独立测试。
以下均为全片主体糊化前后的逐视频绝对分差均值，不是原片与糊化片之间的图像相似度。

| 方法 | 主体糊化 MAE | Repair MAE 的 95% prompt 聚类区间 |
|---|---:|---:|
| Origin | 0.011578 | — |
| 现有 frozen19 repair | 0.003054 | [0.001805, 0.005049] |
| COCO80 repair | 0.003346 | [0.002131, 0.005081] |
| COCO80 + Caption v2 repair | 0.002704 | [0.001585, 0.004587] |

Repair 平均分差低于 0.01，但 **origin 最大绝对变化仅 0.033325，达到 0.10 / 0.20 均为 0/10**。
所以“repair ≤0.01 且 origin ≥0.10”的联合目标未达成，不能写成“正确且效果优秀”。
官方公式比较视频内部相邻帧与首帧特征；全片持续同类糊化不必然破坏时序一致性。
不通过筛选分数、扩大到背景、改写 origin 或替换干预类型制造较大变化。

真实背景切换的 start/middle/end 平均降分分别为 origin 0.160002/0.068832/0.059103、
现有 repair 0.077736/0.078985/0.079234、Caption v2 0.081054/0.082989/0.083395。
这些控制与主条件分开报告。三种定位共 240 项实际版本输入/Origin 一致，30 项自然/PNG
原片分数对齐误差为 0，20 组自然/PNG 评分掩码逐位一致；官方 parity 误差为 0。
每种定位 80 个版本、1,960 帧，物理卡 1 评分段耗时 54.88/71.55/268.03 秒，不含模型初始化。
五条 MP4 共 10 秒，另外五条 GIF 时长未知，未用假定帧率补造时长。

## 全量自然偏好与定位限制

680/680 条自然开发视频完成，失败 0；全部 1,020 对 background 偏好可评分，34 个 prompt 聚类。

| 方法 | 正确对数 | 准确率 |
|---|---:|---:|
| Origin | 534 / 1,020 | 52.35% |
| 现有 repair | 590 / 1,020 | 57.84% |
| COCO80 | 591 / 1,020 | 57.94% |
| Caption v2 | 578 / 1,020 | 56.67% |

Caption v2 相对现有 repair 为 −1.18 pp，95% CI [−2.65,+0.20] pp，未证明在预注册 2 pp
界限内非劣。区间含零，也不能声称已经证实显著退化。评分空视频 332→84 只是检出覆盖改善。

主要问题是空定位和非空但错位：在建塔楼全空；木桌/窗帘常抓到电视或床，对实际糊化主体的
粗参考覆盖不足 1%；校园建筑只有 14.64% / 27.16%。这些案例没有从分母删除。
事后构造区域重叠只作诊断，不是人工 IoU，构造掩码未提供给独立评分。
已有消融表明，全图 patch 本身也能提高稳定性，不能将全部收益归于正确主体分离。

修正 `curtain covering window` 名词头的独立探测已通过相关测试。进一步的多物体 union
实现保留原词表、框门槛和 SAM 权重，30 张既有实际输入的探测已在暂停前完成，耗时 2.254 秒；
**掩码质量分析、完整评分和自然偏好验证尚未运行**。该代码是开发探测，不代表已修好，未接入默认。
最后一轮 union/关系解析/旧解析回归检查为 27 passed；提交范围的验证另记录在提交说明中。

提交前对实际暂存快照另建隔离工作区，锁文件检查、`uv sync --locked --group test` 与
CPU torch 2.14.0 overlay 完成，完整 `pytest tests metrics -q` 为 **694 passed、3 skipped**，
12 个已安装 CLI 的 `--help` 全通过。首次测试的唯一失败来自临时上游软链接与 sibling 路径
断言不一致；换成同一固定 SHA 的本地独立 checkout 后全套通过，未修改算法或测试断言。
本次验证只检查代码和协议，不重新运行模型实验。详情见
[提交检查记录](background_repair_checkpoint_20260920/repository_checks.json)。

## 可复查产物与未完成项

本目录的 [紧凑证据](background_repair_checkpoint_20260920/provenance.json) 保留来源和文件 SHA，
包含构造前后统计、十条逐例分数与校验、自然偏好统计、评分时审核快照、暂停时审核快照和探测状态。
图像、视频、模型权重及大型中间产物按仓库规则不入 Git：

- 本地：`output/background-repair/`；完整远端：`h100-server:/root/wenbiao_zhao/tmp/`。
- 构造：`background-refined-dev-v3-expanded-20260920` 及其 `-verification`、`-comparison`、`-review`。
- 十条评分：`background-refined-dev-v3-eighth-{frozen19,coco80,caption-v2}-20260920` 及对应分析。
- 自然评估：`background-natural-caption-dev-v2-20260920` 及其 `-analysis`。
- 最后探测：`background-caption-union-probe-v1-20260920`；源码叠加归档 SHA 为
  `3b3cf99dcebee28a2941d23c22a2338dc41ad9d9c1a2b4e6cc680bb6d8ca31d0`。

构造协议为 `configs/background-repair/construction_refined_dev_v3.json`，冻结评分协议为
`holdout_protocol_v1.json`，候选为 `scoring_coco80_caption_dev_v2.json`。评分源码归档 SHA 为
`63642379d9c75a8d2603d8f715402164391346cc96cb407b5045efd463babdd6`；上游固定 VBench
`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`，未下载新权重或改写上游。

暂停时未完成：19 条候选的输入复核、2 条新接受输入的评分、union 定位探测质量验证，
以及新的联合目标与定位候选的可靠自然验证。旧测试集已暴露，后续再跑只能称为复测。
旧协议的测试结果保留为历史证据，不替代修订后构造的正确性验证。

详细记录：[构造与空掩码](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/counterfactual-reports/background_empty_masks_20260920.md)、
[定位修复及全部逐批结果](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/counterfactual-reports/background_caption_localizer_20260920.md)、
[表示与定位消融](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/counterfactual-reports/background_localizer_attribution_20260920.md)。
Subject 的相关代码与结果保留在[Subject 扩展报告](subject_official_extension_20260920.md)，不与本表混合。
