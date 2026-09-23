# Dynamic Degree：BMC真实录像外部补测（2026-09-23）

状态：**COMPLETED；两后端各336/336、零失败，独立审计PASS**。本轮仅评估冻结的论文模型`aligned-v1`，不训练、不校准、不晋升公开默认，不改写450组主结果。

新增84组使外部实测覆盖LASIESTA与BMC两套数据，共127个片段、16条源录像；不等于127个独立场景。两套协议／标签不同，结果分别报告。本批总体Repair平均涨幅为Origin的1.30%，但仍有37/168个反事实绝对变分>0.1，结论限于平均虚增抑制。

## 数据与适用范围

数据来自[BMC/ACCV 2012作者官网](https://backgroundmodelschallenge.eu/)的全部9条真实评估录像，不采用其合成训练录像。作者论文为A. Vacavant等，*A benchmark dataset for outdoor foreground/background extraction*，BMC/ACCV 2012，[DOI](https://doi.org/10.1007/978-3-642-37410-4_25)。官网只为真实录像提供部分前景标注；它们不能直接当作整段视频的动静真值。本实验也不是官方BMC前景分割排名。作者页面要求引用，尚未核实标准化数据许可证，因此不再分发原始视频。

9份作者ZIP完整下载，共471,466,397字节。7条录像通过输入检查，各固定取12个不重叠的2秒窗口，共84组；不是84条独立录像。时间窗口从起点到终点均匀铺开，不按Origin或Repair分数筛选。新增场景包括室外停车场、工程机械、纸屑运动、铁路道口、站场与道路车流。

| 原始录像 | 容器帧率 | 实际解码帧数 | 纳入片段 | 状态／原因 |
| --- | ---: | ---: | ---: | --- |
| Video_001 | 25 | 32,965 | 12 | 完成输入核验 |
| Video_002 | 10 | 1,498 | 12 | 完成输入核验 |
| Video_003 | 7 | 未全量解码 | 0 | NOT SCORED：不足8fps，不复制帧凑16帧 |
| Video_004 | 10 | 1,895 | 12 | 完成输入核验 |
| Video_005 | 25 | 未全量解码 | 0 | NOT SCORED：解码呈现时间戳不连续，不自动修复时轴 |
| Video_006 | 10 | 1,064 | 12 | 完成输入核验 |
| Video_007 | 10 | 1,726 | 12 | 完成输入核验 |
| Video_008 | 10 | 792 | 12 | 完成输入核验 |
| Video_009 | 25 | 107,817 | 12 | 完成输入核验 |

两条未评分录像及原始文件保留，没有替换样本。原始文件、计数、帧率、排除理由和哈希见[originals.json](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/selection-v5/originals.json)。

## 评分前固定的协议

- [冻结配置](../../configs/dynamic-generalization/bmc-v1.json)：对每个合格源，起始时刻为`linspace(0, decoded_frames / source_fps - 2, 12)`；每窗在`0,1/8,...,15/8`秒选择最近的真实帧，四舍五入半数向上。不静态化、不补帧、不裁剪、不空间缩放原始基底。
- AVI头部帧数可能包含解码缓冲差异；OpenCV在部分尾帧返回0时间戳。因此按FFmpeg完整解码的实际帧数和best-effort呈现时间戳核对CFR，并记录非零起点；不把`CAP_PROP_FRAME_COUNT`当作实际帧数。容器／解码时间得到核验，不等于有外部相机时钟标定。10fps和25fps最近帧采样的时间误差上限分别为50ms和20ms。
- 每窗16帧以8fps RGB无损MP4输入两个后端。原片与编码控制像素相同；另构造种子1701、2904的8px局部纹理坐标往返抖动，空间尺度32px、边界渐消32px。不是RGB噪声、亮度闪烁、镜头平移或自造运动视频。几何警示全部保留，不按其分数删样本。
- 共84原片＋84编码控制＋168反事实＝336个输入／后端，每输入2秒。Origin仍为上游RAFT二值判分；Repair仍为冻结V-JEPA 2.1 ViT-B＋51,393参数连续头，sigmoid输出。不改变原有全帧384²模型预处理。
- 首先对同片两种子取均值，再等权汇总片段。MAE先逐个反事实取绝对差，不是绝对均值差。95%区间按7条原录像聚类bootstrap，20,000次，种子20260923；不把84个短片视为84个独立样本。

构造诊断：168个反事实中24个满足既有几何门槛、144个不满足。警示来自最小Jacobian约0.449887／0.475475／0.481748低于原门槛0.5；全部Jacobian仍为正、坐标在界内、边界位移为零，不是检测到坐标越界或Jacobian≤0。此门槛及样本均未事后删除／放宽；不能将整批描述为“严格几何资格全部通过”，本轮也没有独立真人逐片确认反事实语义有效性。

## 动静标签的证据等级

[评分前锁定的逐片复核](../../configs/dynamic-generalization/review.bmc-v1.json)为**agent视觉复核，不是官方动静标注，也不是独立真人盲审**：59段明显运动、5段视觉静止、20段不确定。所有84段均检查第0/5/10/15个采样帧；5段静止均检查全部16个采样帧，另详细检查纸屑、植被和人物微动等难例。不是原帧率完整视频播放复核。

静止样本仅来自`Video_006:01`与`Video_007:04/06/08/11`，即两个录像场景。固定相机、没有行人车辆或没有前景标签，都不自动等于静止；有树叶／纸屑运动的片段不作为静止负样本。`Video_009`的12个湿地停车场窗口均保守标为不确定。

**主分析包含全部84组，与这些辅助标签无关。** 仅动静区分分析排除20段不确定；它们的抗抖动结果仍单列，不隐去。由于静止仅5段／2源，动静AUROC和静止均值只能作探索性结果，不能充当大型独立标注集的泛化结论。

## 结果

机器权威结果：[summary.json](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/analysis/summary.json)；完整逐片分数：[pairs.jsonl](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/analysis/pairs.jsonl)。下列反事实值均先对每片的两个种子平均。

| 组别 | 片段数 | Origin 原片→反事实 | Repair 原片→反事实 | Repair Δ [95% CI] | Repair MAE |
| --- | ---: | ---: | ---: | --- | ---: |
| 全部（主分析） | 84 | 0.261905→1.000000 | 0.282374→0.292002 | +0.009628 [−0.019859, +0.037950] | 0.060652 |
| 视觉静止（探索性） | 5 | 0.000000→1.000000 | 0.041793→0.014800 | −0.026994 [−0.027817, −0.023700] | 0.030933 |
| 明显运动（探索性） | 59 | 0.372881→1.000000 | 0.362253→0.369636 | +0.007382 [−0.030567, +0.057497] | 0.060621 |
| 不确定（不删除） | 20 | 0.000000→1.000000 | 0.106876→0.132283 | +0.025407 [−0.063324, +0.066902] | 0.068174 |

本批Origin对所有168个抖动版本都输出1；Repair总体平均涨幅为Origin涨幅的**1.30%**。总体MAE为0.060652，相对Origin的0.738095低91.78%，其聚类95%区间为[0.032619, 0.092676]。这是平均虚增得到抑制的证据，但不是逐片不变。

按用户此前提出的10%幅度条件，配对量`ΔRepair − 0.1×ΔOrigin`全体均值为−0.064182，7源聚类95%区间[−0.106124, −0.023628]。本批整体区间位于零以下；但运动子组为−0.055329 [−0.117800, +0.011211]，仍跨零，不能声称所有子组均有95%把握通过。静止子组只来自2源，重采样区间很窄并不消除样本代表性不足；这些是小规模外部描述性检验，不是普适保证。

### 动静区分（辅助标签，非官方BMC准确率）

| 后端 | 原片 AUROC | 抖动1701 AUROC | 抖动2904 AUROC |
| --- | ---: | ---: | ---: |
| Origin | 0.686441 | 0.500000 | 0.500000 |
| Repair | 0.871186 | 0.908475 | 0.874576 |

基于59运动／5视觉静止片段，不拟合分类阈值，不含20个不确定片段。Repair保留了较好的动静区分，而Origin的抖动分数全部饱和。标签由agent评分前复核，静止仅两个场景，且无连续强度真值，因此不据此声称大规模人工验证或物理强度标定已完成。

### 分场景与负结果

各录像都保留12个窗口；下表包括不确定标签，场景描述只是观察，不作为新增真值。

| 录像与主要内容 | Repair 原片→反事实 | MAE |
| --- | ---: | ---: |
| 001：植被与停车场背景车流 | 0.111140→0.052687 | 0.077937 |
| 002：工程机械、车辆及扬尘／杂物 | 0.410781→0.478691 | 0.141748 |
| 004：纸屑／小幅运动 | 0.035243→0.029472 | 0.016476 |
| 006：铁路道口、人车与列车 | 0.424593→0.410479 | 0.038075 |
| 007：站场、工人及列车 | 0.234771→0.251854 | 0.083075 |
| 008：持续道路车流 | 0.729501→0.738824 | 0.015277 |
| 009：湿地停车场（标签不确定） | 0.030592→0.082008 | 0.051977 |

同一道口的静止窗`006:01`为0.036075，两CF为0.011118／0.013632；列车经过窗`006:04`为0.578449→0.592728／0.605478。它说明模型并非对所有视频都压成低分。另一方面，远处列车窗`006:02/03`原分只有0.009866／0.014287，小目标／微小运动响应仍需专门验证；没有物理强度真值，不能从二元运动标签推导“所有真实运动都必须接近1”。

**仍有37/168个反事实绝对变分超过0.1**：23个增分、14个降分。最大增分`007:07`为+0.284431（0.165422→0.449853，标签不确定）；最大降分`002:10`为−0.398449（0.479370→0.080921，标签不确定）。明显运动组也有16/118增分>0.1、10/118降分<−0.1，最大降分−0.367399；不能用总体+0.009628掩盖这些不稳定。静止组无增分>0.1，但有2个降分超过0.1。

## 版本、验证与复现

基础工作树提交`0cd494dc06f36d493147323c44e21abb57c21f65`，分支`winbeau`，含既有未提交模型实现和本轮新增评估脚本。基础提交号**不是完整实验源码版本**；必须使用保存的代码快照、脚本SHA和模型哈希。没有修改已有模型实现、权重、尺度映射及冻结研究目录。

- Head SHA：`6dcfbdce2e604cdd669e54015fee34e034c8004726d06c46611532f451ca8a53`。
- Encoder SHA：`848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`；源码`204698b45b3712590f06245fbfba32d3be539812`。
- VBench上游：`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`；RAFT SHA：`fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`。
- 配置 SHA：`be48f0ede9f8a73d567c74b71ee7c06b3dc875a75fe9a0b486da8be23246605c`；复核 SHA：`f2bdae81a651f467d4a9c955af5f9cdf274b5976ceef39ce017be41d95d898b9`。
- 评估脚本 SHA：`1df4166434e8f0b770187dda4e907a730af8313ee1db2353711368ff57f6a968`；独立审计脚本 SHA：`f9af704583b7515a80a0bc9a4654c12467ae7a13e49a3ccedd128ae01af7247f`。
- 相关56项测试通过，包括真实FFmpeg解码的微型技术夹具、7fps拒收、非CFR子进程清理、模型／干预参数一致性。技术夹具不属于研究数据。`compileall`和`git diff --check`通过。

[独立审计记录](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/analysis/audit.json)：重哈希1,270个原始文件，核对147,757帧的完整呈现时间记录，重读原录像验证1,344个采样帧像素，重新解码336个评分输入；重算336个Origin判据、336个Repair sigmoid、84对精确编码控制及均值／MAE／区间／AUROC，最大算术误差`5.551115123125783e-17`。每分片首项官方`infer`一致；旧TRAIN参考预处理与特征哈希精确一致，头重加载误差≤1e-6。

H200物理5／7卡，各进程仅见`cuda:0`；两个评分分片各168输入，分别233.99／229.01秒。成功输入核验127.58秒，双进程构造73.56秒，双卡评分＋汇总墙钟236.26秒。每输入2秒、每后端共672媒体秒；不是四卡计时，也不把这些阶段之和当作含下载、复核、中止重试与审计的端到端耗时。计时与完整命令见[prepare](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/prepare-v5-controller/completion.json)、[build](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/build-v5-controller/completion.json)、[score](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/score-summarize-v5-controller/completion.json)。

结果JSON SHA：`685d551707de097479bcb6f87028c94148afdef67f0601a5995b5d3c6a79668d`。逐项输入与分数分别见[构造分片0](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/construction/shard-0/inputs.jsonl)、[构造分片1](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/construction/shard-1/inputs.jsonl)、[分数0](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/scores/shard-0/scores.jsonl)、[分数1](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/scores/shard-1/scores.jsonl)，冻结版本与设备证据见[provenance 0](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/scores/shard-0/provenance.json)、[provenance 1](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/bmc-v1/scores/shard-1/provenance.json)。

远端根目录：`H200-target-server:/data/chenjiayu/dynamic-generalization-bmc-20260923/`。成功输入清单位于`selection-v5/`，构造为`construction/shard-{0,1}/`，评分为`scores/shard-{0,1}/`，汇总／审计为`analysis/`。本地紧凑证据在`output/dynamic-generalization/bmc-v1/`；媒体、原始档案、完整时间日志留在远端，不入Git。

复现入口：[下载](../../scripts/counterfactual/download_dynamic_bmc.py)、[选择／构造／评分／汇总](../../scripts/counterfactual/evaluate_dynamic_bmc.py)、[分阶段控制器](../../scripts/counterfactual/launch_dynamic_bmc.py)、[独立审计](../../scripts/counterfactual/audit_dynamic_bmc.py)。在新的隔离目录配置保存的`code/`与相同资产后依次执行：

```bash
python code/scripts/counterfactual/download_dynamic_bmc.py --output /NEW_TASK_ROOT/download
python code/scripts/counterfactual/launch_dynamic_bmc.py --root /NEW_TASK_ROOT --phase prepare
# 复核候选与既有review的逐片身份；不要用分数重新标注。
python code/scripts/counterfactual/launch_dynamic_bmc.py --root /NEW_TASK_ROOT --phase build
python code/scripts/counterfactual/launch_dynamic_bmc.py --root /NEW_TASK_ROOT --phase score-summarize --gpu-uuids GPU_UUID_5,GPU_UUID_7
python code/scripts/counterfactual/audit_dynamic_bmc.py --root /NEW_TASK_ROOT --config code/configs/dynamic-generalization/bmc-v1.json --review code/configs/dynamic-generalization/review.bmc-v1.json --output /NEW_TASK_ROOT/analysis/audit.json
```

控制器使用H200已有模型资产的明确绝对路径；迁机需显式配置这些路径并保持哈希一致。输出必须新建，不能覆盖旧批次。评分前按卡检查显存，一进程仅使用逻辑`cuda:0`。

### 保留的中止与未完成项

输入适配经历5次中止，分别为非零PTS起点、OpenCV尾帧时间戳异常、路径类型错误、7fps整批中止、非CFR拒收时子进程清理阻塞。它们发生在任何GPU评分之前；日志、部分文件与对应代码快照保留于`prepare-controller`、`prepare-v1...v4-controller`、`selection`及`selection-v1...v4`、`failed-prepare*-code/`。最后一次只停止了经PID核实属于本任务的FFmpeg进程；没有终止其他任务。已修复输入适配／异常隔离，未改评分公式。未评分的两条输入不能称为模型评分失败或用合格样本替换。

更大规模独立静止视频、独立真人标签、真实相机运动专门分层、速度／物理强度真值：**NOT RUN**。头训练未使用BMC；编码器预训练是否包含相关视频：**UNKNOWN**。本次不恢复已暂停的Forcing实验，不改变旧DEV失败或45校准预留未读的事实。

与[LASIESTA预实验](dynamic_generalization_lasiesta_20260923.md)分别报告：两者时间协议和标签证据等级不同，不能简单混合成单一准确率。此前450组论文主结果仍见[当前方法说明](../paper/dynamic-degree-current.md)。
