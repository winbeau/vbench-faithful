# Goal 提示词：Subject Consistency 的表示层修复与背景糊化数据集

> 历史任务提示词。当前工作目标已转为
> [Background Consistency 修复与主体糊化实验](2026-09-20-background-repair-goal-prompt.md)。
> 下文的镜像盒、逐条人工确认等旧约束已被用户后续指令覆盖；subject 的现行协议见
> [数据流说明](2026-09-20-subject-repair-dataflow.md)。本文件保留历史，不表示旧目标已完成。

> 用途：把本文件整段交给 GPT/Codex 作为任务目标。仓库工作区为
> `/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit`，动手前必须先读
> `AGENTS.md`、`README.md`、`docs/plans/2026-09-20-subject-repair-dataflow.md`。

## 0. 你的角色

你在这个仓库里做研究工程：为 VBench 1.0 的 `subject_consistency` 维度做**表示层修复**，
并构造一个**背景糊化数据集**来检验它。你要写代码、写测试、写可复现文档，不要只给建议。

## 1. 问题（为什么做这件事）

官方 `subject_consistency`（`VBench@fd18b3d`，`vbench/subject_consistency.py`）的计算是：

```text
全部解码帧 → dino_transform(224) 短边缩放（无裁剪）→ DINO ViT-B/16 逐帧 → L2 归一化
每帧 c_t = (max(0,cos(f_{t-1},f_t)) + max(0,cos(f_0,f_t))) / 2
视频分 = mean_t c_t
```

三个缺陷：

1. **首帧锚定**：第 0 帧进入 2(T−1) 个余弦项中的 T−1 个，同一改动放在开头和放在中段惩罚不同。
2. **数据集聚合有两种律**：单卡按帧转移数加权（`:68`），多卡按逐视频等权（`:81`），长度不齐时结果不同。
3. **整帧表示**：`dino_transform(224)` 把整帧送进 DINO，背景、光照、相机运动都会改变分数，
   "主体"从未被解离出来。现有修复只改了聚合（全对 all-pairs），**没有**解决第 3 条。

**本任务的核心命题（可证伪）**：
如果这个指标真的在测主体，那么**只把背景糊化**不应该显著改变分数。若背景糊化与主体糊化造成的
降分相当，该指标就只是一个"帧稳定性指标"，不配叫主体一致性。修复的目标是让背景不敏感，同时
**保住**主体被破坏时的敏感性。

## 2. Repair 链路

```text
帧 → 主体定位（文本/类别条件的分割）→ 掩码重采样到 patch 网格
   → DINO patch token 在掩码内池化 → 每帧主体向量（L2 归一化）
   → 所有帧对余弦（all-pairs），固定分母 C(T,2)
```

**三条硬隔离，任何一条被打破实验即自证，必须写进代码注释和文档：**

| 角色 | 用什么 | 不用什么 |
|---|---|---|
| 建族定位（决定腐化哪块像素） | SegFormer-B0（ADE20K 150 类）+ GrabCut 精修；person 类可加 MobileSAM vit_t + 少量人工点 | 不得用于评分 |
| 评分定位（repair 自己找主体） | **MobileSAM**（每片段冻结一份提示，人工确认一次即可，参数写进 `localizer.jsonl` 并哈希） | 不得复用建族掩码 |
| 编码（出主体向量） | **DINO ViT-B/16** patch token | 不得参与任何定位决策 |

即：**定位族 ≠ 编码族，建族族 ≠ 评分族。**

已实现、可直接复用（不要重写）：`metrics/subject-consistency/src/subject_consistency/subject_evidence.py`
（掩码池化、union/mean 实例模式、zero/exclude/carry 缺失策略）、`models.py`
（`SubjectMaskProvider` 协议、`NpzSubjectMaskProvider`）、`metric.py::evaluate_masked_batch`、
CLI `--audit-variant subject_masked`。

## 3. LLM 在链路里的位置

LLM **只做语义归一化，不碰像素**：

- 输入：原始 prompt（**不得**看视频、不得看检测结果）。
- 输出 schema（固定，不要加字段）：

```json
{"subject": "person", "phrase": "person in red jacket", "count": 1, "status": "ok"}
```

  - `subject`：冻结词表（官方 `subject_en` 全量 ∪ 经核实的分割类别）∪ `null`；
  - `phrase`：**抽取式**名词短语（只能由 prompt 里出现过的词组成），≤6 token，喂给定位器；
  - `count`：`1` / `2` / `null`；`status`：`ok` / `none` / `ambiguous`。
  - 非 `ok` 的视频记 `unsupported`，分母固定规则先定死，**不得删样本抬分**。
- **基准内优先用官方 `subject_en`，不要用 LLM 猜**；LLM 只用于离开基准的泛化。

## 4. 数据怎么做

### 4.1 定位数据（离线、确定性、冻结）

- 逐帧掩码，输出 `[T,H,W] uint8` 存 `npz`，附每帧面积、拒收原因、模型权重哈希。
- 确定性契约（沿用已验证的参考实现）：

| 项 | 取值 |
|---|---|
| 语义模型 | `nvidia/segformer-b0-finetuned-ade-512-512`（ADE20K，512×512 输入，~3.75M 参数） |
| 类别映射 | `class_map.yaml`：`person→person`；`castle→building/wall/tower/windowpane/door/house/column/skyscraper` |
| GrabCut | 11×11 椭圆核 erode/dilate 做种子，`cv2.setRNGSeed(0)`，4 次迭代 |
| 连通域清理 | 去 <20 px 碎片；person 取最大连通域并填 <200 px 内洞 |
| 羽化 | **只向内** 1.5 px（`distanceTransform/1.5`），保证掩码外像素逐位不变 |
| 保存 | 无损 PNG；`np.rint` 后转 uint8，禁止浮点累积 |

- 拒收（逐条计数，先出门不硬造）：类别不在映射表；掩码面积比 <1% 或 >50%；`subject_en` 缺失；
  可用帧数放不下窗口。

### 4.2 背景糊化数据集（本任务的主交付）

族名 `region_discrimination`，每一基底三个 level，**同算子、同窗口、等面积**：

| level | 编辑 | 声明 |
|---|---|---|
| `clean` | 不修改 | 参照 |
| `background_corrupt` | 主体盒关于画面中轴的**镜像盒**内做高斯糊化 | ≈ clean |
| `subject_corrupt` | 主体掩码内做同一算子 | 最低 |

- 算子：高斯模糊 sigma 12（大主体）/ 18（人像），可选马赛克块 16/22 px。
- 窗口：`w = round(0.25·T)`，位置复用现有族的 start/middle/end。
- 构造自证（必须断言并写进 manifest）：
  `result[mask==0] == image[mask==0]`；镜像盒面积 == 主体盒面积；镜像盒与主体盒交集为空；
  镜像盒内没有其他实例。
- 产物：无损 PNG 序列 + `manifests/<base_id>.json`（逐文件 sha256、参数、自证结果）。

### 4.3 DeepSeek 作为数据增强大模型（替代人工标注）

用于生成 prompt 改写与主体短语标注，作为**银标训练数据**，不是金标准：

- 必须锁定模型版本号与日期，固定 prompt 模板与输出 schema，产物写进 manifest。
- 抽取式约束：`phrase` 只能由原 prompt 词组成；不得生成式扩写。
- **必须保留一个 150–200 条的人工审核子集**，用它报告"LLM 与人工的一致率"作为标签噪声上界；
  head 的准确率只报在这个子集上，不得只在 LLM 全集上自证。
- 划分：先按 source prompt 分组再扩写，同源改写/同视频帧不得跨 train/test。
- DS 不参与定位、不生成掩码、不看视频。

## 5. 最终实验

- 基底：现有 25 条 subject 基底起步，可扩到 40–60 条（prompt 池 71 条）。
- 三列后端，同一批基底：**Official** / **聚合 repair（已发布）** / **masked repair（新）**。
- 敏感性前提（每基底，先过）：`clean > subject_corrupt`；不过的基底作废并计数。
- 主统计量：

```text
R = median_over_bases [ drop(background) / drop(subject) ]
drop(x) = score(clean) - score(x)
```

  分母下限 `drop(subject) ≥ 0.05`，否则该基底不进 R。

- 预注册判据：`R_masked` 的 95% CI 上界 < 0.5，且 `R_official` 的 CI 含 1 → 命题成立。
- 必须同时报：`missing_policy=zero`（端到端）与 `exclude`（检测条件化）两列；掩码覆盖率与拒收计数。
- 两张样图（城堡 474×355、女孩 720×450）只作**构造可行性样例**，不作统计证据。

## 6. 交付物与验收

- 代码：`scripts/counterfactual/` 下的 `region_discrimination` 变换、类别映射、掩码生成脚本、manifest 与评分脚本。
- 测试：纯算法测试（掩码池化、等面积镜像盒、确定性重放哈希一致、掩码外零改变）。
- 文档：数据格式、确定性契约、复现命令、以及**非声明**（没有自然集测量、没有真实模型 parity 时必须写明 NOT RUN）。
- 验收：`uv run --no-sync --group test pytest tests metrics` 全绿；确定性重放两次产物哈希一致；
  背景改变像素数 = 0 的断言成立。

## 7. 禁止事项

不训练、不微调；不手绘主体轮廓；不用同一系列模型同时做定位与编码；不把权重下载进仓库；
不修改 `data/`、`results/`、`splits/`、`runs/`；不修改上游 checkout；不用 Official/Repair
的分数反向挑选基底；不得在无人工审核子集的情况下宣称 LLM 标注精度。
