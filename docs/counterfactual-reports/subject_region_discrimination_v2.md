# Subject 完整背景糊化 v2：试跑记录（非正式主实验）

2026-09-20。用户在看到样张后指出：7 条中只有演讲者和游泳者的原始图像适合作为候选，其余主体质量较差。因此，本报告保留完整 7 条试跑结果，不据此评价正式方法优劣，也不把选出的 2 条重新包装成验证成功。

用户的图像审核记录见 [`quality_review_v2.jsonl`](../../configs/subject-repair/quality_review_v2.jsonl)。演讲者的建库掩码还包括部分听众，尚未通过指定主体分割审核；游泳者暂留候选。此前的人工 MobileSAM 框只确认评分提示，不认证建库掩码。

## 数据和构造

- 用户要求的主条件：主体掩码内像素原样保留，完整背景补集做高斯模糊，每一帧都处理。镜像盒仅归档；主体糊化是独立敏感性控制。
- 冻结 60 条来源，其中 7 条通过自动定位门槛，53 条拒收。通过面积/GrabCut 门槛不等于图像质量或实例分割质量合格。
- 7 条共 146 原始帧；clean 与 full/start/middle/end 的背景、主体腐化共 63 个 PNG 序列。全文件重放 1,384 个文件逐字节一致，树哈希 `33df42b725cb8f8e74774ae5d268b684ae8a3558762f44cd8aedf8bf4538951f`。
- 重算 1,168 张腐化帧，编辑掩码外变化像素数为 0；背景糊化时对应主体掩码内变化为 0。此结论以掩码正确为条件。
- 建库：SegFormer-B0 + GrabCut；评分定位：独立、人工首帧框驱动的 MobileSAM；编码：DINO ViT-B/16。评分绝不读取建库掩码。

## 全片背景糊化结果（仅 7 条试跑）

分数范围 0–1。每条片段先取绝对分差，再求中位数；有符号变化不能抵消后被称为稳定。95% CI 为按 source prompt 配对聚类 bootstrap，10,000 次。

| 后端 | 干净均分 | 背景糊化均分 | 绝对分差中位数 [95% CI] |
| --- | ---: | ---: | ---: |
| Official / origin | 0.8878 | 0.8202 | 0.0286 [0.0139, 0.0698] |
| 现有聚合 repair | 0.9014 | 0.8271 | 0.0316 [0.0174, 0.0914] |
| masked repair / zero | 0.7424 | 0.7289 | 0.0294 [0.0199, 0.0881] |
| masked repair / exclude | 0.7424 | 0.7289 | 0.0294 [0.0199, 0.0881] |

配对 Official-minus-masked 的绝对变化缩减中位数为 0.0071，95% CI [−0.0232, 0.0404]；预先冻结的背景稳定性判据未通过。不能声称 masked repair 比 origin 更稳定。此处样本的视觉质量又尚未合格，更不能外推到正式实验。

全片主体糊化后，Official 7/7 降分，但没有任何一条降幅达到 0.05；聚合 repair 6/7 降分；masked repair 只有 2/7 降分、1/7 降幅达到 0.05。因此三个后端共同通过 R 门槛的样本数为 0，主条件的 R 和 CI 不可估计。没有用起始/中间/末尾的控制结果替代主条件。

MobileSAM 在 clean/background/subject 三组均返回了非空掩码，帧覆盖率均 100%，因此 zero 与 exclude 分数一致。这只说明输出非空，不说明定位正确。平均掩码面积占比分别为 7.83%、7.72%、8.78%。

53 条拒收仍在固定分母 60 内，完整固定分母均值、逐条排除原因和全部时窗结果见 `statistics.json`；表中清楚标明只用 7 条成功试跑的条件均值。

## 完整产物

- 数据：`output/subject-repair/background-complement-v2-20260920/`
- 全部 7 条原图/背景糊化对照：`output/subject-repair/background-preview-v2-20260920/background_blur_review.html`
- 逐条视觉观察：同目录 `construction_visual_observations.json`，明确标记为助手观察，不冒充人工轮廓审核。
- 最终成功试跑：`output/subject-repair/subject-repair-v2-reviewed-runtime2-20260920/`，包含 `scores.jsonl`、`run.json`、`statistics.json`、63 份独立评分掩码和模型/提示哈希。
- `artifact_verification.json`：统计量本地重放完全一致，63 份评分掩码哈希核对通过，168 条 whole-frame 分数与先前运行精确一致。
- 首次 masked 运行因严格确定性模式拒绝 CUDA float cumsum 而失败，保留在 `subject-repair-v2-reviewed-20260920/`；绝不把失败当测量零分。后续用等价 arange 网格修复，CUDA 完整位置编码逐位对比通过，外部 MobileSAM checkout 未修改。

## 环境与验证

H100 物理卡 1、逻辑 `cuda:0`；Python 3.10.20，PyTorch 2.5.1+cu121。源代码为隔离传输快照，无 `.git`，`run.json` 保存源码逐文件哈希；VBench 固定 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。

DINO 权重 SHA256：`bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e`。人工提示文件 SHA256：`f70edb51e95bb5092486cb689dd680e045feb873746e03942587ef9ea812a0e6`。

CPU 工作区测试：536 passed、3 skipped；8 个 metric 入口 help 通过，`git diff --check` 通过。真实模型和 CUDA 已用于本试跑；冻结 E0 基线 parity、自然偏好验证、马赛克评分均 **NOT RUN**。

## 下一阶段先做图像质量筛选

新筛选规则单独写入 [`quality_screening_protocol.json`](../../configs/subject-repair/quality_screening_protocol.json)：先看未腐化原始帧中主体是否清楚、背景是否有足够结构，再独立审核建库掩码与评分定位。不能按模型分数筛选，也不能把本次两个偏好候选当作独立新验证集。

初筛列表为官方 7 个人物 prompt × 4 生成器、固定 seed 0 的 28 条。它只是人物候选查看集，不是最终实验基底，不宣称覆盖其他主体类别。保留原始 split/source prompt 分组；确认图像和掩码质量后再冻结新的评分队列与分母。

28 条全部完成原始帧导出，查看页为
`output/subject-repair/subject-quality28-r2-20260920/quality_review.html`。
助手初筛建议优先查看 #4 咖啡人物、#16 吉他人物；#12 演讲者作为多人实例分离的备选。
这些是视觉建议，未经用户确认，不是人工金标，不自动进入正式数据集。
全部 28 条的初筛观察保存在 `output/subject-repair/quality28-visual-notes-20260920/assistant_triage.jsonl`；
明确区分只看首帧与检查首/中/末帧的范围。新候选尚未做模型评分。

MovieGen 与既有 spatial/objects/action 的训练 prompt 已用于独立文本银标流程。人工语义审核仍待完成；LLM/人工一致率、subject head 准确率和训练均 **NOT RUN**，不影响这里使用官方 `subject_en` 的试跑。
