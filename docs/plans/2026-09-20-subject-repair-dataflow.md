# Subject 维度 repair：数据流架构与数据规格

状态：2026-09-20 已实现；真实建库模型已运行，60 条冻结输入有 7 条通过定位门槛。
用户已确认：**主体清晰，主体以外的全部背景糊化**。当前协议是
[`protocol.json`](../../configs/subject-repair/protocol.json) v2；镜像盒仅归档供历史重放。
完整数据格式、隔离规则和复现命令见 [`subject-repair.md`](../subject-repair.md)。
通过自动定位门槛不等于通过人工分割质量审核；多人、多个同类对象的实例范围仍需检查。
用户随后明确：7 条中只保留演讲者、游泳者为原图候选，其余 5 条主体质量不适合主实验。
7 条评分完整保留为[试跑记录](../counterfactual-reports/subject_region_discrimination_v2.md)，
不对选出的 2 条重新宣称方法有效。正式评分前须先审核新原图与建库掩码。

参考实现（重要）：`C:\Users\genev\Documents\Codex\2026-09-19\referenced-chatgpt-conversation-this-is-an`
已用 SegFormer-B0 + GrabCut（城堡）与 SegFormer-B0 + MobileSAM vit_t + 5 正 2 负人工点（女孩）
产出过一套结果，并逐像素验证「二值掩码之外改变像素数 = 0」。本架构沿用其算子与数值，
但那些速度/内存数字属于那次 CPU 单机运行，不是本仓库的测量。

---

## 1. 三层职责与"不同系列"隔离

**硬约束：抽离（出掩码）与编码（出向量）不得由同一系列模型承担。** 本设计用三个互相独立的组件：

| 角色 | 模型族 | 输入 | 输出 | 何时跑 |
|---|---|---|---|---|
| 建族定位 | **SegFormer-B0**（ADE20K 150 类）+ GrabCut | 帧 + `subject_en` | 逐帧主体掩码 | 离线一次，冻结 |
| 评分定位 | **MobileSAM（自动）** | 每片段冻结的一份提示（框，人工确认一次） | 评分时用的掩码 | 评分时，逐帧 |
| 编码 | **DINO ViT-B/16** | 掩码内的 patch token | 主体向量 | 评分时 |

三条隔离同时成立：建族族 ≠ 评分族（SegFormer vs MobileSAM），定位族 ≠ 编码族
（SegFormer/MobileSAM vs DINO），编码不参与任何定位决策。任何一条被打破，实验就变成自证。
本次建库不使用参考样图中可选的 MobileSAM 精修。人工评分框只来自干净首帧，
不得拿去修改建库掩码；每个腐化版本重新运行 MobileSAM，不能复用干净版本的预测。

## 2. 数据流（四层）

```text
L0 输入（冻结，只读）
   counterfactual bases 的 subject 行（base_id / video_uid / prompt_en / subject_en / split）
   + 解码帧（上游 load_video，全部帧，原生分辨率）

L1 主体定位 → masks/<video_uid>.npz
   SegFormer-B0(ADE20K,512×512 输入) → 逐帧标签图
   → 类别映射表（configs/subject-repair/class_map.yaml）合并标签
   → 形态学种子 + GrabCut 精修 → 连通域清理
   → [T,H,W] uint8 掩码 + 每帧面积 + 拒收原因 + 权重哈希

L2 确定性改造 → clips/<base_id>/<level>.*  +  manifests/<base_id>.json
   同算子、同窗口，三个 level（背景与主体面积不同）：
     clean               不修改
     subject_corrupt     掩码内施加算子
     background_corrupt  主体掩码的完整补集内施加同一算子，保留主体原像素
   → 无损 PNG 序列 + manifest（参数、sha256、构造自证）

L3 评分 → scores/
   official / 聚合 repair（已发布，直接复用）
   masked repair：独立 MobileSAM 逐帧定位 → DINO patch token 在评分掩码内池化 → all-pairs
   同时报 zero 与 exclude；绝不读取 L1 掩码进行评分

L4 统计 → 报告
   敏感性前提 clean > subject_corrupt（每基底）
   主统计：背景腐化的绝对分数变化，以及 Official-minus-masked 配对变化缩减
   次统计：R = median_bases [ drop(background) / drop(subject) ]，按 source prompt 聚类自助 CI
```

## 3. 确定性改造契约

L2 的每个算子必须满足"同样输入 → 逐字节同样输出"，并逐片段自证：

| 项 | 取值 | 来源 |
|---|---|---|
| 高斯模糊 sigma | 12（非 person）/ 18（person），按冻结类别设置 | 参考实现 |
| 马赛克块宽 | 16 / 22 px（先 INTER_AREA 缩小，再 INTER_NEAREST 放大） | 参考实现 |
| 羽化 | **只向内**羽化 1.5 px（`distanceTransform/1.5`） | 参考实现 |
| GrabCut | 11×11 椭圆核 erode/dilate 做种子，`cv2.setRNGSeed(0)`，4 次迭代 | 参考实现 |
| 连通域清理 | 去 <20 px 碎片；person 取最大连通域并填 <200 px 内洞 | 参考实现 |
| 时间窗口 | 全片 full 为主条件；`w = round(0.25·T)` 的 start/middle/end 分开报告为控制 | v2 协议 |
| 保存 | 无损 PNG；数值用 `np.rint` 后转 uint8，禁止浮点累积 | 参考实现 |
| 自证 | `result[edit_mask==0] == image[edit_mask==0]`；背景腐化时主体改变像素数为 0 | 参考实现 + v2 |

拒收（逐条计数）：类别不在映射表；掩码面积比 <1% 或 >50%；GrabCut 种子不可用；
可用帧数不足以放下窗口；`subject_en` 缺失。主体居中和背景中存在其他物体不再导致拒收。

## 4. 数据格式

```text
configs/subject-repair/
  class_map.yaml      subject_en -> ADE20K 标签集合（person→person；castle→building/wall/tower/...）
  localizer.jsonl     评分定位的冻结提示：video_uid, box, points, point_labels, sha256
  bases.jsonl         基底清单（复用 counterfactual 的 subject 行 + 拒收原因）
output/subject-repair/            # 不入库
  masks/<video_uid>.npz           masks[T,H,W] uint8, area[T], model, weights_sha256
  clips/<base_id>/<level>.png|mp4 + manifest
  manifests/<base_id>.json        参数、逐文件 sha256、拒收原因、构造自证结果
  scores/<dimension>__<backend>.jsonl
```

## 5. 已确定的运行口径

1. **评分提示**：用户已提供 7 条人工首帧框，校验并冻结在 `configs/subject-repair/localizer.jsonl`。
2. **基底规模**：从原 25 条扩至 60 条，依据 prompt 元数据排序，不使用指标分数挑样本。
3. **载体**：无损 PNG 序列；评分各列读取同一 RGB 张量。
4. **算子集**：高斯模糊为主实验；马赛克代码可用，实际马赛克评分保持 NOT RUN。
5. **语义训练数据**：复用 MovieGen 和已有 spatial/objects/action 的训练 prompt；
   人工语义审核与 7 条像素定位框是不同的审核，缺少前者不报告语义 head 准确率。

## 6. 明确不做

本次基准评分不训练、不微调；如泛化需要 subject head，优先复用已有 Qwen 8B/LoRA 基础设施。
不手绘主体轮廓；不用 LLM 猜基准内主体类别（类别来自官方 `subject_en` +
固定映射表）；不让同一系列模型同时承担定位与编码；不用 Official/Repair 的分数反向
挑选基底。
