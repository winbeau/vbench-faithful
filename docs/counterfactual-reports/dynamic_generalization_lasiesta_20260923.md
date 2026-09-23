# Dynamic Degree：LASIESTA真实静态／运动片段冻结泛化测试

日期：2026-09-23。状态：**COMPLETED，外部小样本预实验；172/172输入／后端，零失败，独立核验PASS**。

后续另补[BMC真实录像84组](dynamic_generalization_bmc_20260923.md)，沿用冻结模型，独立报告。
下文的NOT RUN为本批完成时的状态快照；不回写本批数字或合并两套标签／时间协议。

## 1. 结论与适用范围

当前`aligned-v1`在这批真实录像中能明显区分运动与静态片段，并显著减轻8px局部纹理抖动的平均虚增。这里“明显减轻”描述观测幅度，不表示通过了预注册显著性检验。静态／运动两组Repair平均涨幅分别只有Origin的**4.25%／5.22%**；没有通过调分、训练或重新选择检查点获得这些结果。

但这不是完全不变：静态组32条反事实中5条增分超过0.1，最大增分0.239900；两组平均涨幅的10%门槛仅点估计满足，不能声称以95%置信度证明通过。样本来自9条室内录像，且部分共用场景，**不是43条独立源视频，更不是128条独立视频或普适泛化证明**。

当前450组VBench主实验、公开CLI默认、模型参数与论文正文均未修改。本报告为新增外部验证，不替换[原主报告](dynamic_static_jitter.md#dynamic-vjepa-aligned450)。

## 2. 为什么选择该数据

[LASIESTA官方数据页](https://www.gti.ssr.upm.es/data/lasiesta_database.html)提供真实室内外帧序列、逐像素运动对象标注，以及对象暂时静止的独立标记／区间。来源论文为Cuevas等，*Computer Vision and Image Understanding*, 152:103–117, 2016，[DOI](https://doi.org/10.1016/j.cviu.2016.08.005)。官方许可CC BY-SA 4.0；本轮不重新发布视频。

用户最新要求是检验“真实运动／真实静态”，而非动作分类。因此没有将UCF101/HMDB51的动作类别冒充运动强度真值，也没有人为把视频冻结成图片来凑静态组。

候选数据核查记录：

- [CDnet](https://www.changedetection.net/)有运动区域标注，但官网访问／证书与原始帧时间信息仍需核查；本轮**NOT RUN**。
- [CameraBench官方任务](https://github.com/linzhiqiu/t2v_metrics/tree/6ecb74f92028f42c7e64546d3a71e98c8c73068f/camerabench/data)的`is_scene_static_or_not`有80对。逐项与`Static.jsonl`交叉后，80个“静态场景”正例全部被标为相机不完全静止；不能用作全画面静态负例。本轮不评分、不将其场景静态标签误用为视频静态。此发现不等于该数据集本身有错。
- 用户不再优先的长生成视频已暂停，[Deep/Causal保留状态](dynamic_generalization_forcing128_20260923.md)另列；其部分结果不混入本报告。

## 3. 数据与评分前冻结协议

[协议配置](../../configs/dynamic-generalization/lasiesta-v1.json) SHA-256：`dcd83b8f54b8af7b5fed98928c7367c22417770eb34bcacacf9a0dd1e199ebd6`。

1. 从作者网站下载预定9个原始RAR，合计325,471,011字节。完整SHA、HTTP来源和原文件清单保留；没有从网络获取新模型。
2. 只选固定相机、无明确动态背景／灯光变化的室内序列；`I_CA_02`因作者明确说明背景植物一直在动而预先不纳入。IL、MC、SM及含风雨雪的室外类别不属于本次静态负例协议，未根据模型分数排除。
3. 逐一检查2800帧官方GT：红／绿／黄是运动对象，白色是暂时静止对象，黑色是背景，灰色是不确定。没有运动像素且仅有无法解释的灰色前景者标为ambiguous，不冒充静态。
4. 对同一状态的最大连续区间，从区间起点每48帧取一个46帧跨度，其中每3帧取1帧，共16帧。**中间未采样帧的GT也必须同状态**；窗口不重叠，全部符合者纳入。36个状态区间中26个产生片段、7个过短、3个ambiguous，完整记录保留。
5. 得到43个原始录像片段：16静态（12空背景、4含静止对象，其中1个站立人物和3个放置后不动的包）及27运动；不是生成视频。全部43片段的首／中／末帧已做无分数视觉复核，[复核范围与限制](../../configs/dynamic-generalization/review.lasiesta-v1.json)明确只看了每片3帧，不冒充独立人类逐帧审核。
6. 每片使用原始352×288空间尺寸，构造原片、零编辑RGB无损编码控制，以及seed1701／2904的8px局部坐标往返干预，共172输入。不是噪声或亮度闪烁。86条CF的几何警示为43 true／43 false，全部保留评分，不按警示或结果删片。
7. **源时间限制：** 官方压缩包为BMP序列，本轮未取得可信原始帧率／时间戳；每3帧采样后统一按8fps封装。每个评分输入16帧／2秒，但不能称原始录像本来就是8fps，或宣称保留了原速、测得物理运动强度。这里评价的是同一标准化时间协议下的动静区分与配对不变性。
8. Origin重新运行锁定官方RAFT及原二值判据，不复用旧450结果。Repair复用冻结V-JEPA 2.1和同一51K连续头，仅沿用已验证的矩形全帧输入适配；无训练、阈值拟合、分数映射或默认变更。

| 官方原序列 | 静态片段 | 运动片段 |
|---|---:|---:|
| I_SI_01 / I_SI_02 | 1 / 1 | 3 / 4 |
| I_CA_01 | 2 | 3 |
| I_OC_01 / I_OC_02 | 1 / 1 | 2 / 2 |
| I_MB_01 / I_MB_02 | 5 / 2 | 3 / 4 |
| I_BS_01 / I_BS_02 | 1 / 2 | 3 / 3 |
| 合计 | **16** | **27** |

## 4. 原片→反事实结果

每片先平均两个种子，再在各类别内等权平均片段；编码控制不混入CF。MAE是逐种子绝对变化的均值，不是均值差的绝对值。95%CI按9条源序列聚类bootstrap 20,000次，seed20260923；部分源共用房间，区间仍可能偏乐观。

| 类别／后端 | 原片→8px局部抖动 | Δ [95%CI] | MAE [95%CI] |
|---|---|---|---|
| 静态16／Origin | **0.000000→0.875000** | +0.875000 [0.642857,1.000000] | 0.875000 [0.642857,1.000000] |
| 静态16／Repair | **0.125877→0.163079** | +0.037202 [−0.020132,0.089064] | 0.070206 [0.040563,0.107311] |
| 运动27／Origin | **0.592593→0.981481** | +0.388889 [0.142857,0.640000] | 0.388889 [0.142857,0.640000] |
| 运动27／Repair | **0.589715→0.610019** | +0.020303 [−0.002802,0.041912] | 0.039774 [0.032434,0.048712] |

按源序列等权的补充均值方向一致：静态Repair 0.125906→0.162106，运动Repair 0.592505→0.616713；完整逐序列值在机器汇总中。

与用户此前“平均增分不超过Origin涨幅10%”的数值要求对照，两组点估计均满足。不过同次聚类bootstrap下，`ΔRepair − 0.1×ΔOrigin`的区间分别为[−0.113519,+0.016117]及[−0.062292,+0.020775]，均跨零。**不把点估计满足升级为统计等效／非劣通过。** 两种分数也不代表同一物理量纲。

### 动静区分：不用测试集调二值阈值

这里AUROC表示随机抽取一个运动片段时，其分数高于静态片段的概率，模型平局计半；它不是新的二值Repair，也不是官方LASIESTA像素分割F-score。

| 后端 | 原片AUROC | CF1701 AUROC | CF2904 AUROC |
|---|---:|---:|---:|
| Origin | 0.796296 | 0.593750 | 0.512731 |
| Repair | **0.988426** | **0.962963** | **0.958333** |

Repair的两类均分间隔为0.463839→0.446939，未通过同时压低运动片段取得不变性；Origin间隔则为0.592593→0.106481。上述为本批观测，未进行两模型AUROC差异的显著性检验。

### 真实站立人物：限定单例诊断

预先复核的`I_CA_01:0200`是真实人物站在门边，不是单帧复制。Repair原片**0.068268**，两CF为0.129760／0.207029；同录像三段行走原片为0.669951、0.617929、0.656714。该例支持模型并非仅凭“有人”给高分，但也暴露静止加抖动仍可能升分，不能只展示有利的一半。

## 5. 失败与尚未验证

- 静态32CF：Repair有5条增分>0.1、2条降分>0.1；最大增分0.239900、最大降分0.255417。运动54CF：无增分>0.1，但2条降分>0.1、最大降分0.135166。均保留，不重跑挑结果。
- 只有9条室内固定相机录像；新的室外、车辆／自然景物运动、真实相机移动、多个独立数据集及128独立源规模仍为**NOT RUN**。
- 官方对象动静标注不是连续运动强度真值，不能据AUROC证明速度排序或绝对强度标定正确。原始帧率仍为**NOT VERIFIED**。
- 评分头训练没有使用LASIESTA；冻结编码器的大规模预训练是否含相关网络视频为**UNKNOWN**，不宣称编码器完全未见过这些内容。
- 图像抽样复核不是独立盲审，静止人物可能有轻微生理运动。部分运动窗口只有小幅肢体变化，如`I_BS_02:0001`；按原标注完整纳入，不用Origin=0反推“静态”。
- 旧开发集失败、450组已暴露提示词及默认未晋升等边界不因本次正向结果而消失。

## 6. 验证、版本与产物

- H200物理5／7卡，各进程仅见逻辑`cuda:0`。构造阶段60.12秒，评分＋汇总双卡墙钟117.91秒；每片评分媒体2秒，共172输入／344媒体秒。不是四卡计时。
- 两后端各172/172，零失败；43编码控制的解码像素、特征哈希和分数精确一致。旧TRAIN参考预处理和特征哈希一致，头重加载误差≤1e-6；每分片首项官方`infer` parity通过。
- 独立审计逐一核对5602个原文件、2800帧GT与43个不重叠窗口；重算172个Origin判据、172个Repair sigmoid、均值／MAE／聚类区间／AUROC，最大算术误差`3.469446951953614e-17`，**PASS**。
- 本地相关测试**44 passed**，新增脚本编译、`git diff --check`通过。未改权重、默认评分、冻结`data/results/splits/runs`或其他任务实现。

远端根：`H200-target-server:/data/chenjiayu/dynamic-generalization-lasiesta-20260923/`。`download/`保留作者压缩包／下载回执，`selection/`保留原帧／标签／全部区间与选择清单，`construction/`保留四视图和构造账本，`scores/`保留逐项原始光流统计／分数／模型身份，`analysis/`保留汇总和审计，`visual-review/`保留抽样原图。

本地紧凑证据：

- [唯一机器结果汇总](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/lasiesta-v1/analysis/summary.json)，SHA `c29ddaa9b75b045ba1be9c869dba0f2d5259aea02c331e9fa5becceff4260040`。
- [逐片配对分数](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/lasiesta-v1/analysis/pairs.jsonl)／[独立审计](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/lasiesta-v1/analysis/audit.json)。
- [选择回执](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/lasiesta-v1/selection/selection.json)／[原片清单](https://huggingface.co/datasets/xju-arlab/vbench-repair/blob/main/dimensions/dynamic_degree/generalization/external-v1-20260923/output/dynamic-generalization/lasiesta-v1/selection/sources.jsonl)。
- [下载脚本](../../scripts/counterfactual/download_lasiesta.py)、[选择／构造／汇总](../../scripts/counterfactual/evaluate_dynamic_lasiesta.py)、[冻结评分实现](../../scripts/counterfactual/evaluate_dynamic_generalization.py)、[两阶段控制器](../../scripts/counterfactual/launch_dynamic_lasiesta.py)、[独立审计器](../../scripts/counterfactual/audit_dynamic_lasiesta.py)。

模型头SHA：`6dcfbdce2e604cdd669e54015fee34e034c8004726d06c46611532f451ca8a53`。V-JEPA权重SHA `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`，源码`204698b45b3712590f06245fbfba32d3be539812`；VBench源码`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。本地Git基底`0cd494dc06f36d493147323c44e21abb57c21f65`加未提交实验代码；完整脚本SHA在远端selection／construction／score provenance记录，不能只用Git基底冒充完整实验版本。

复现使用**新输出目录**，先运行`download_lasiesta.py --output <root>/download`；复制当前代码快照及上述配置到`<root>/code`，随后`launch_dynamic_lasiesta.py --root <root> --phase prepare-build`。核查标签／原图后才运行`--phase score-summarize --gpu-uuids <空闲卡UUID列表>`；最后运行独立审计器。控制器资产路径按已记录H200布局解析，其他机器须显式适配资产路径，不复制私有权重进Git。
