# Dynamic Degree：局部纹理抖动稳定性

更新：2026-09-23。论文位置：方法 §2.2.2（原论文 `sections/methodology.tex`），归入 **Nuisance Entanglement**；结果放入 Table 1 不变性表（原论文 `tables/invariance.tex`）。这里的稳定性指运动评分对指定无关干预保持稳定，**不是把 Dynamic Degree 改成运动平滑度或鼓励静态视频得高分**。

## 论文方法表述

**Original Scoring.** Dynamic Degree 旨在衡量视频中的运动程度。VBench 以约8 fps采样视频，用 RAFT 估计相邻帧光流，并对每对帧中最大的5%光流幅值取均值。设采样帧数为 $T$，若超过阈值 $6\min(H,W)/256$ 的帧对数达到 $\operatorname{round}(4T/16)$，该视频记为1，否则记为0；最终对视频取均值得到被判定为动态的视频比例。因此，原始实现并非逐视频连续运动强度评分。

**Mechanism Analysis.** 这一规则依赖局部位移幅值和超阈值次数，却不区分这些位移来自连贯的对象/相机运动，还是画面内部纹理的快速局部往返抖动。即使干预没有引入新的连贯运动，较大的局部光流仍可能使静态或低运动视频被判定为动态。这属于运动证据与无关像素变化的耦合。反事实检验要求评分对指定8px抖动稳定，同时保留对实际运动的响应；不能简单压低所有运动，或将所有往返轨迹都判为伪运动。

**Repair Strategy.** 我们以可学习的连续运动读出替代光流阈值判定。将16帧、8 fps的RGB视频全帧缩放至384×384，经冻结的 V-JEPA 2.1 ViT-B 编码为时空tokens，再通过51,393参数的注意力池化与MLP评分头输出sigmoid连续分数。训练联合使用人类运动偏好、原片与局部抖动的一致性约束、静止及静止加抖动的低分锚点，以及平移和连贯往返运动的排序约束。最后仅以训练集原片的Origin总体均分提供弱尺度监督，不逐视频蒸馏二值标签。推理只输入视频，不输入配对原片、Origin分数、prompt或干预身份，也不进行分数平移、缩放或重新二值化。

```text
原生16帧 / 8 fps RGB
  → 全帧双线性缩放384² + 归一化
  → 冻结 V-JEPA 2.1 ViT-B → 4608×768 时空tokens
  → LayerNorm（无可训练仿射）→ Linear(768,64) + tanh
  → learned attention + softmax → 加权池化
  → MLP(64,32,1) → sigmoid → 连续相对运动分数
```

该分数用于相对运动评价，**尚未标定为物理运动强度**；论文使用的是冻结 `aligned-v1` 实验模型，研发仓库公开默认评分没有被替换。

## 训练、反事实与统计单位

- 训练为官方210源/14 prompts，开发验证60源/4 prompts，与450测试源的30 prompts分离。编码器始终冻结；最终阶段从既有静止锚点头继续固定300步AdamW，lr=0.001、weight decay=0.01、seed=20260928，取末步，不在450上挑检查点或调尺度。原有自然偏好头和静止锚点训练属于前序阶段，不能把最终300步称作从头训练全程。
- 450组基底均为官方 VBench 1.0 原始MP4，不是自行生成或从单帧造出的静态基底。LaVie、ModelScope、VideoCrafter各150源；原生16帧/8 fps/2秒。每源两个固定种子1701/2904，共900条8px反事实，另有450编码控制，共1,800输入，全部评分、零失败。
- 构造在原图坐标上施加有界、边缘衰减的平滑局部位移场，时间上快速正负交替；每帧从对应原帧独立重采样，不累计位移，不替换原有运动或改变帧数/时间轴。8px是单帧最大位移，反向相邻步可达16px；不是RGB加噪、亮度闪烁或整幅平移。
- 1,200条qualified与600条历史rejected标记的输入全部保留评分，后者不是本轮评分失败；先前按MP4协议排除的30个GIF仍未评分。原有构造警示不能抹去，也不能据分数剔除反例。
- Repair重新解码和编码全部1,800输入；Origin复用同一批锁定既有评分并从光流统计复核判据，不能写成本轮重跑RAFT。450编码控制分数精确一致，全部特征哈希与既有缓存一致。
- 主表先在每源平均两个CF种子，再对450源等权。MAE对900条干预逐条计算，不能用总体均分差的绝对值代替。95%区间按30个prompt聚类bootstrap 20,000次，seed=20260926。

## 当前结果及限制

数值的唯一原始报告是研发仓库的 [dynamic_static_jitter.md，aligned450小节](../../docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-aligned450)，下列为论文引用，不建立另一份实验主表。

- Origin：**0.680000 → 0.857778**；Δ=+0.177778，95% CI [0.133306, 0.224444]。
- 当前Repair：**0.629251 → 0.623035**；Δ=−0.006216，95% CI [−0.009879, −0.002862]；MAE=0.021731，95% CI [0.018178, 0.025484]。
- 171个人类有序对上，Repair原片与两CF各144对正确（84.21%）；279个人类平局另报。原片相对Origin的半分平局concordance改善+2.34pp，CI [−2.87,+7.44]pp，不显著；两CF相对Origin均+16.67pp，区间分别[+5.94,+26.92]和[+7.35,+25.36]pp。总正确数相同不等于逐对不变，两个种子分别7改善/7退步和8改善/8退步。
- 模型冻结后的单图诊断：静止0.007435，静止+8px局部抖动0.013868，8px/32px整幅平移0.351916/0.554858。这是独立于450原片构造的单例诊断，不是450源全部静态化或完整运动类型验证。
- 支持结论：本批8px干预的**平均虚增受到抑制**，总体人类偏好排序较稳健。仍有28/900条下降超过0.1、5/900条上升超过0.1，最大下降0.377932；不能称逐片完全不变。
- 原DEV60为0.657326→0.651620、自然偏好17/23（要求≥19/23）和与Origin尺度差0.142674（要求≤0.05）仍未通过。用户接受0.6档是收尾决定，不追溯改写门槛。
- 测试prompt此前已暴露，不称全新独立留出；45条校准预留未读取。绝对强度标定、按实际运动类型的人类审核及全新独立留出尚未完成。本文补写不触发新训练、默认晋升或旧失败清理。

## 证据定位与版本

本页从论文文档复制，源码链接已指向当前私有仓库。大输出的历史路径保留；可携带的发布与复现入口见 [DYNAMIC_ALIGNED.md](DYNAMIC_ALIGNED.md)。

- [构造源码](../../scripts/counterfactual/local_texture_jitter.py)、[冻结编码器](../../packages/audit-models/src/vbench_audit_models/vjepa.py)、[评分头](../../metrics/dynamic-degree/src/dynamic_degree/learned_probe.py)、[锚点损失](../../metrics/dynamic-degree/src/dynamic_degree/anchored_probe.py)、[均分损失](../../metrics/dynamic-degree/src/dynamic_degree/aligned_probe.py)。
- [训练配置](../../configs/dynamic-static-jitter/vjepa-aligned-v1.json)、[450冻结协议](../../configs/dynamic-static-jitter/vjepa-aligned450-v1.json)、[复现命令](../../configs/dynamic-static-jitter/README.md)。
- 实验源码基底Git为 `fdf4890c67bee5881e53fc43926d555a63f9cef8`，实验新增脚本尚未提交，因此该SHA不能单独代表完整实验代码；脚本哈希另见原始报告和原论文仓库 `docs/results-provenance.json`。
- VBench上游 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`；V-JEPA上游 `204698b45b3712590f06245fbfba32d3be539812`。编码器SHA-256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`；当前评分头 `6dcfbdce2e604cdd669e54015fee34e034c8004726d06c46611532f451ca8a53`。
- 汇总 `vbench-audit/output/dynamic-static-jitter/vjepa-aligned450-v1/analysis/summary.json`，SHA-256 `3325e173ad8cd7cd33938e463177c9f31d03298d6b8ef5f14d385197d60cba16`；独立核验 `independent-audit.json`，SHA-256 `d0484d6d77d5959f6a3a4f50238f4e525e1a79f2627c14f63875cab228c42e58`。
- H200产物根 `/data/chenjiayu/dynamic-structural-motion-20260922/vjepa-aligned450-v1/`；视频与模型仍留在实验目录。
- V-JEPA 2.1模型引用来自[官方仓库](https://github.com/facebookresearch/vjepa2)与[原论文](https://arxiv.org/abs/2603.14482)，本项目的小头训练与稳定性结果来自上述本地实验，不是该模型论文的结论。
