# Dynamic Degree：当前方法、反事实实验与失效机制归类

更新：2026-09-23。本文是便于桌面阅读和论文写作的独立整理稿，采用当前 **aligned-v1** 模型及其450组冻结评估，不用旧版本或开发集分数替代当前结果。数字的权威来源仍为原始报告和机器产物，本文不是新增实验主表。

## 1. 应该放在哪一层？

**建议主归第二类 Nuisance Entanglement（无关因素耦合），论文位置保留方法 §2.2.2。**

理由是：希望评价的是连贯的对象/相机运动；本次反事实加入的是局部纹理的快速往返坐标抖动。原始评分把这种干扰也计入运动证据，导致总体分数明显升高。问题中心是“无关变化污染了目标证据”，而不只是“最后输出了0或1”。

这三类是可以共存的失效机制，不必理解为严格互斥的三个层级。应按当前案例直接揭示的主要异常归类，而不是按图中哪里有空位归类。

| 机制 | Dynamic Degree 中对应什么 | 本轮应如何定位 |
|---|---|---|
| Target Substitution | 把复杂的运动属性操作化为局部光流幅值阈值，可从广义上理解为代理替代 | 不是本轮组图的首选标签；否则容易把几乎所有代理指标都笼统归入这一类 |
| **Nuisance Entanglement** | 局部纹理抖动与连贯运动共同贡献较大光流；加入干扰后分数虚增 | **本次8px反事实的主要机制，建议正式归类于此** |
| Evidence Collapse | 向量光流被压为幅值，时序信息被压为超阈值计数，最终视频分数压为0/1 | 可以作为次要机制讨论，但不能仅凭二值输出就把本次抖动案例主归此类 |

关键判断：即使把二值结果换成连续的光流幅值，评分仍可能对局部抖动敏感。因此，“变成连续值”并不自动等于“排除抖动干扰”。这是一项机制分析；当前450组结果比较完整Origin与完整Repair，**没有在同一协议下单独隔离二值化、方向丢弃、阈值和表征更换的各自因果贡献**。

若另做“真实运动强度明显不同，但Origin同为1或同为0”的案例，那才更直接对应Evidence Collapse；本稿不把它写成已经完成的另一项实验。

论文可使用的归类句：

> We categorize this failure as nuisance entanglement: local texture jitter enters the same motion evidence used to assess coherent object and camera motion, inflating the score without a corresponding increase in the intended motion property.

“稳定性”指的是**评分对这种无关干预的稳定性/不变性**，不意味着Dynamic Degree的评价目标改成了视频稳定、静止或运动平滑度。

## 2. Original Scoring

Dynamic Degree维度旨在衡量视频中的运动程度。VBench首先以约8 fps采样视频，使用RAFT估计相邻采样帧之间的光流。对于每对帧，计算各像素光流的欧氏幅值，并对其中最大的5%取均值。随后依据分辨率缩放阈值和超阈值帧对数量，将整个视频判定为动态或静态：动态记为1，否则记为0。数据集分数为这些视频二值结果的均值，也就是被判定为动态的视频比例，而不是逐视频的连续运动强度。

设采样帧数为 $T$，空间尺寸为 $H\times W$，幅值阈值为 $6\min(H,W)/256$，需要的超阈值帧对数为 $\operatorname{round}(4T/16)$。本实验原生16帧、8 fps，对应15个相邻帧对中至少4个严格超过幅值阈值。

```text
视频 → 约8 fps采样 → 相邻帧RAFT光流
     → 每帧对最大的5%幅值取均值
     → 分辨率阈值 + 超阈值帧对计数
     → 单视频0/1 → 数据集动态视频比例
```

## 3. Mechanism Analysis

较大的光流并不必然来自新的连贯对象或相机运动。画面内部纹理的快速局部往返位移也可能产生较大幅值，并使足够多的帧对跨过动态阈值。原始计算没有显式区分这两类证据，因而会把局部像素抖动计入运动程度。

本轮反事实的目标是：在保留官方原视频及其原有时间进程的基础上，加入指定8px局部纹理抖动，检验评分是否出现不应有的增分；Repair同时应保留对连贯运动的响应。不能简单压低所有变化，也不能将所有往返运动或高频自然运动都视为无效。当前模型训练包含连贯往返运动控制，正是为了避免把“方向反转”本身当作静态判据。

## 4. Repair Strategy

我们将原来的光流阈值判定替换为学习得到的连续运动读出。输入原生16帧、8 fps的RGB视频，全帧缩放至384×384并归一化，经冻结的 **V-JEPA 2.1 ViT-B** 编码为时空tokens，再由一个 **51,393参数**的投影、注意力池化和MLP头输出sigmoid连续分数。

```text
原生16帧 / 8 fps RGB视频
  → 全帧双线性缩放384² + 归一化
  → 冻结V-JEPA 2.1 ViT-B
  → 4608×768时空tokens
  → 无可训练仿射的LayerNorm
  → Linear(768,64) + tanh
  → learned attention + softmax → 加权池化
  → MLP(64,32,1) → sigmoid
  → 单视频连续相对运动分数
```

训练结合以下监督：

- 人类自然运动偏好：保留有序对与人类平局约束。
- 抖动一致性：约束同一原片与其两种局部抖动视图的评分接近。
- 静止锚点：静止片段及静止加局部抖动应得到低分，避免恒定输出约0.5。
- 运动排序控制：平移片段应高于静止，较强平移应高于较弱平移；连贯往返运动不应被一律压成静态。
- 原片下限约束：不凭空给所有原片添加正运动标签，但原片不应低于其自身静止控制。
- TRAIN总体均分监督：仅使训练原片的Repair总体均分接近同一训练集的Origin均分，不逐视频复制Origin的二值标签。

训练210源/14 prompts、开发验证60源/4 prompts，与450评估源按prompt分离。最终阶段从既有静止锚点头继续固定300步AdamW（lr=0.001、weight decay=0.01、seed=20260928），取末步，编码器始终冻结；不能把这300步称作从头训练的全部历史。

**推理只输入视频**，不输入Origin分数、prompt、生成器名称、配对原片或干预身份；不进行推理后平移、缩放或重新二值化。本方法输出连续的相对运动分数，尚未标定为物理运动强度。研发仓库的公开CLI默认没有因此被替换。

## 5. 反事实设置：官方原视频 + 8px局部纹理抖动

| 项目 | 当前协议 |
|---|---|
| 数据来源 | VBench 1.0官方原始MP4；不是新生成视频，也不是把450原片替换成静态图 |
| 基底 | 450源、30 prompts；LaVie / ModelScope / VideoCrafter各150源 |
| 时间 | 原生16帧、8 fps、2秒，帧数和时间轴不变 |
| 干预 | 原图坐标上的有界平滑局部位移场；时间正负交替、边缘衰减，每帧从对应原帧独立重采样，不累计漂移 |
| 幅度 | 最大单帧位移8个原生像素，相邻反向步的局部变化可达16px；不是RGB加噪、亮度闪烁或整幅平移 |
| 反事实数 | 每源seed1701/2904两条，共900条 |
| 编码控制 | 每源1条，共450条；不混入CF均分 |
| 实际评分覆盖 | 450原片 + 900CF + 450编码控制 = 1800/1800，零运行失败 |
| 保留标记 | 1200 qualified / 600历史rejected标记输入均评分保留，不据分数排除 |
| 先前排除 | 30个GIF按既定MP4协议仍未评分，不算作本轮运行失败 |
| Origin | 复用同批锁定的既有评分，并从原始光流统计核对判据；不是本轮重跑RAFT |
| Repair | 当前aligned-v1重新解码/编码全部输入；1800份特征哈希及450编码控制分数一致性核验通过 |

统计方式：先在每个源内平均两个反事实种子，再对450源等权。MAE按900条干预的逐条绝对变化计算，不能用总体均分差的绝对值代替。95% CI按30个prompt聚类bootstrap 20,000次，seed=20260926。

## 6. 450组当前结果

| 评分 | 原片 → 8px反事实 | 有符号变化Δ，95% CI | MAE，95% CI |
|---|---|---|---|
| Origin | **0.680000 → 0.857778** | +0.177778 [0.133306, 0.224444] | 0.177778 [0.133306, 0.224444] |
| 当前aligned-v1 Repair | **0.629251 → 0.623035** | −0.006216 [−0.009879, −0.002862] | 0.021731 [0.018178, 0.025484] |

这一干预期待不变性。结果支持：原始评分在局部抖动下平均虚增，而当前Repair的批量均分基本保持、略微降低。满足用户约定的“平均涨幅不超过Origin涨幅10%”数值检查，但这不等于两种分数已经具有同一物理量纲。

### 人类偏好：单独报告

450个人类配对包含171个有序对和279个人类平局。下面的concordance只针对171个有序对，模型预测平局给半分：

| 后端 | 原片 | CF1701 | CF2904 |
|---|---:|---:|---:|
| Origin | 81.87% | 67.54% | 67.54% |
| 当前Repair | 84.21%（144/171正确） | 84.21%（144/171正确） | 84.21%（144/171正确） |

原片相对Origin改善2.34个百分点，95% CI [−2.87,+7.44]个百分点，跨零，不能说自然原片表现显著优于Origin。两CF分别改善16.67个百分点，CI [5.94,26.92] / [7.35,25.36]个百分点。正确总数一致不等于每一对都不变：两个种子分别7改善/7退步、8改善/8退步。279个人类平局另外分析，不混入有序对分母。

### 静止和真实位移诊断：不是450主实验

模型冻结后，对一个单图来源的控制案例进行诊断：

| 控制输入 | 当前Repair |
|---|---:|
| 静止 | 0.007435 |
| 静止 + 8px局部抖动 | 0.013868 |
| 8px整幅平移 | 0.351916 |
| 32px整幅平移 | 0.554858 |

该单例支持“静止低分、局部抖动不产生类似真实平移的高分、较强平移更高分”的诊断方向，不代表已经覆盖所有真实运动类型，也不能称为450组全部静态化实验。

## 7. 证据支持的结论与限制

**可支持：** 当前450组协议下，Repair抑制了8px局部纹理干预造成的批量虚增；总体人类偏好排序较稳健。单图控制显示模型没有退化为对所有输入输出约0.6。

**仍需保留：**

- 逐视频并非完全不变：900条CF中28条降分>0.1、5条增分>0.1，最大降分0.377932。
- 原DEV60为0.657326→0.651620；自然偏好17/23未达到≥19/23，与Origin均分差0.142674未达到≤0.05。用户接受0.6档是收尾决定，不追溯改写这些门槛。
- 450评估prompt此前已暴露，不能称全新独立留出；45条校准预留仍未读取。
- 绝对运动强度标定、按实际运动类型的人类审核及全新独立留出尚未完成。不能把所有实际高频变化、纹理变化或方向反转都定义为无关干扰。
- 当前比较同时改变了表征和评分头，不把全部收益归因于“连续化”单一因素。历史模型正负结果保留在原始报告，不冒充当前模型。

## 8. 论文组图建议

当前概览图将第二行标为 **Nuisance Entanglement**，第三行标为 **Evidence Collapse**。按当前8px抖动案例的主要机制，Dynamic Degree应放第二类，而不是仅因右下角有空位就放进第三类。

可以后续调整组图分区或案例取舍，正文仍保留§2.2.2。若以第三类作为主标签，则需要补充另一种“运动强度信息被聚合压掉”的证据，而不是改名解释同一个抖动实验。本次按要求暂停图像制作，不自动改图或挪动其他案例。

前面imagegen生成的木屋及时空切片仅用于版式示意，不是实验视频或实测时空图。正式bad case应从真实配对视频抽取同位置、同时间协议的像素切片，注明时间方向并使用该单例实际分数；不能给示意图贴上450组批量均分当作单例结果。

## 9. 代码、报告与版本

以下记录的是本地实验工作区证据路径。本次只提交方法说明与文档入口，不捎带此前未提交的实验代码、报告和配置，也不上传被忽略的output产物或权重；这些本地证据不保证可从本次Git提交直接下载。完整复现仍需相应实验源码和冻结产物，不能只靠本次文档提交。

- [当前450组原始报告](/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit/docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-aligned450)
- [完整统计JSON](/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit/output/dynamic-static-jitter/vjepa-aligned450-v1/analysis/summary.json)
- [冻结450组协议](/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit/configs/dynamic-static-jitter/vjepa-aligned450-v1.json) / [训练配置](/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit/configs/dynamic-static-jitter/vjepa-aligned-v1.json)
- [Origin源码](/home/winbeau/Papers/ICASSP2027-VBench-Audit/VBench/vbench/dynamic_degree.py) / [局部抖动构造](/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit/scripts/counterfactual/local_texture_jitter.py)
- [冻结V-JEPA适配器](/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit/packages/audit-models/src/vbench_audit_models/vjepa.py) / [评分头](/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit/metrics/dynamic-degree/src/dynamic_degree/learned_probe.py)
- [锚点监督损失](/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit/metrics/dynamic-degree/src/dynamic_degree/anchored_probe.py) / [TRAIN均分监督损失](/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit/metrics/dynamic-degree/src/dynamic_degree/aligned_probe.py)
- [复现入口](/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit/configs/dynamic-static-jitter/README.md) / [Overleaf方法说明](/home/winbeau/Papers/ICASSP2027-VBench-Audit/overleaf/docs/dynamic-degree-stability.md)

当前模型头SHA-256：`6dcfbdce2e604cdd669e54015fee34e034c8004726d06c46611532f451ca8a53`。

当前450统计JSON SHA-256：`3325e173ad8cd7cd33938e463177c9f31d03298d6b8ef5f14d385197d60cba16`。

VBench上游：`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`；V-JEPA上游：`204698b45b3712590f06245fbfba32d3be539812`。实验源码基底Git为`fdf4890c67bee5881e53fc43926d555a63f9cef8`，但新增实验脚本尚未提交，完整身份需连同原报告中的脚本哈希核对。

本次仅归档和提交已有方法、结果与分层说明，没有新训练、评分、图像生成或公开默认变更。此前未提交的实验工作保持原状。

评分脚本SHA-256：`fb337ce4ad59a4c8698e7bed2ac38b64a4d1efb20c4a35aebaf728709bc24eca`；汇总脚本SHA-256：`4111d1dfa746b7bb6e21ec20648f1198da2ec009b930efba156f12aa317265e7`。
