# 给 Astra 的开工计划提示词（新增 4 维）

用途：把下面 `=== 提示词开始 ===` 与 `=== 提示词结束 ===` 之间的全文复制给 Astra。
Astra 不共享本次会话上下文，因此事实、路径、行号与约束都写在提示词里。
提示词只要求 Astra 产出**计划文档**，不写实现代码。

=== 提示词开始 ===

你是 Astra。工作区：`/home/winbeau/Papers/ICASSP2027-VBench-Audit/vbench-audit`
（git 工作树，含未提交改动，禁止 commit/push）。锁定上游 checkout 在同级 `../VBench`。

## 0. 任务

为 VBench Audit 仓库新增的 4 个候选维度——`background_consistency`、`temporal_style`、
`object_class`、`color`——产出一份**可执行的开工计划**，覆盖：范围与命题、每维契约与反事实族、
实现拆分、数据与模型、H100 计算与调度、验收与报告口径、风险与待拍板项、里程碑顺序、交付物清单。

产出：`docs/plans/<当天日期>-four-dimension-kickoff-plan.md`（必须放在 `docs/plans/` 下）。
本次**不要写实现代码**：不改 `metrics/`、`packages/`、`scripts/`、`configs/`，
不下载权重，不修改 `../VBench`，不 commit/push，不动 `data/ results/ splits/ runs/ figures/`。

## 1. 必须先读（按顺序）

1. `AGENTS.md` —— 研发约定：目录边界、冻结内容、验证协议、禁止事项。
2. `docs/plans/2026-09-15-dimension-scope-11d.md` —— 当前 11 维范围、4 个候选的源码 pin。
3. `docs/paper/unaudited-dimensions-triage.md` —— 这 4 维的源码定位、字符串匹配/命名错配证据、
   repair 方向与推荐星级（本提示词 §2 是它的摘要）。
4. `docs/plans/2026-09-14-experiment-plan-8d.md` —— 既有审计协议（契约、反事实族、CPA 报告口径）；
   其"八维"范围表述已被第 2 条覆盖，其余内容仍适用于 7 个在办维度。
5. `docs/counterfactual-dataset.md`、`docs/counterfactual-reports/CONSOLIDATED.md`、
   `docs/semantic-adapter-constraints.md`、`docs/supplementary-experiments.md`。
6. `configs/upstream.toml`（12 条 `[dimensions.*]` pin）与上游源码 `../VBench`（`Vchitect/VBench@fd18b3d`）。
7. 任选一个现有 metric 包看结构（建议 `metrics/subject-consistency/` 与 `metrics/scene/`）：
   `src/<snake>/cli.py`、`metric.py`、`models.py`、`backends/`、独立 `pyproject.toml`、`tests/`。

## 2. 已确认事实（可直接采用，但**行号必须自己复核**）

范围与处置：

- 本轮范围 11 维 = 在办 7 维（`dynamic_degree`、`motion_smoothness`、`subject_consistency`、
  `scene`、`human_action`、`spatial_relationship`、`multiple_objects`）+ 候选 4 维
  （`background_consistency`、`temporal_style`、`object_class`、`color`）。
- `overall_consistency` 已退出本轮（估计器与 `temporal_style` 逐行相同）；其包与 CLI
  **暂时保留为 legacy**，等用户确认后再删。计划里不得安排删除它。
- 语义适配器（`docs/semantic-adapter-constraints.md`）范围已从四维扩为**六维**，
  新增 `object_class` 与 `color`。
- `configs/upstream.toml` 已为 4 个候选补上 pin，并已用 `verify_upstream()` 对
  `../VBench@fd18b3d` 校验通过（remote/SHA/clean/12 个源文件哈希）。

四维的源码事实与 repair 方向（行号针对 `fd18b3d`，请复核）：

| 维度 | 源码事实 | 问题归类 | repair 方向 |
|---|---|---|---|
| `background_consistency` | `vbench/background_consistency.py:39` 取**全部帧**（`utils.py:154-160`）、`:42-43` 整帧 CLIP 编码、`:49-51` `(cos(prev,cur)+cos(first,cur))/2`、`:64` 全局帧均值；与 `vbench/subject_consistency.py:58-60` 逐条同构；`evaluate.sh:7-10` 与 `dimension_to_folder.json` 显示它读的就是 `scene/` 视频、复用同一 86 条 prompt | 命名≠实现（无背景分解） | 背景掩码/前景抑制后再编码；锚点改 prev-only 或滑动窗口；复用 subject 的 `temporal_relocation`+median-box 契约 |
| `temporal_style` | `vbench/temporal_style.py` 与 `vbench/overall_consistency.py` 只差函数名与 dimension 串；`:48` `query = info['prompt']`（整条 prompt，风格子句是尾部）；该维度在 `VBench_full_info.json` 里**没有 auxiliary_info**；ViCLIP `max_txt_l = 32`（`viclip.py:25,152`）+ `truncate=True` 截尾（`viclip_text.py:151-152`）；suite = 10 base × 10 风格子句；最长的 prompt 恰好是长风格子句（31 词 `featuring a steady and smooth perspective`、30 词 `with an intense shaking effect`、29 词 `in super slow motion`） | 命名≠实现（最硬）+ 文本截断 | 切出风格子句单独编码（学 `appearance_style` 的 aux 口径），内容项与风格项分开报；同 base 的 10 个风格改做相对判别（softmax/ranking）；保证风格子句落在 32 token 内并记录被截断的 prompt |
| `object_class` | `vbench/object_class.py:32` 每帧 `set(run_caption_tensor(...)[0][0][2])`、`:40` `if key_info in pred`（集合精确成员）；目标串是 COCO 风格标签（79 条，含 `tv`、`couch`、`potted plant`、`hair drier`），另一侧是 GRiT ObjectDet 文本解码器生成后 decode 的自由文本（`grit_src/grit/modeling/roi_heads/grit_roi_heads.py:299-301`），检测阈值 0.5（`image_dense_captions.py:88`） | 严格字符串匹配 | 保留全部实例与 per-instance 分数；标签语义对齐（同义/复数/词表 + 不支持显式标注）；分母固定全帧，"未检出"与"叫法不同"分开报 |
| `color` | `vbench/color.py:37` `[cap_det[0], cap_det[2][0]]` 把每段 caption 都贴到 **top-1 检测**的类名上（两路 head 同序，正确索引是 `[2][i]`）；`:46` 对象名精确相等；`:50` 颜色子串（`red` ⊂ `colored`/`hundred`）；`:47-49` 分母用硬编码 12 色白名单；`:65-67` 用 `replace('a ','').replace('an ','').replace(color,'')` 从 prompt 切对象名；`:81-89` 未匹配到对象的视频整条丢弃 | 严格字符串匹配 + 实例错位 + 条件分母 | 修 `[2][0]→[2][i]`（或按 box/IoU 绑定 caption↔实例）；颜色拼写/同义归一；全帧为主分母、条件率降为诊断；未检出不再整条丢视频 |

环境与验证事实：

- 本机（开发机）没有 VBench 权重、没有可用 CUDA：任何真实模型结果都不得在计划里承诺本机产出。
- 正式评分在 H100 上，位于 `/root/wenbiao_zhao/`：解释器 `venvs/vbench/bin/python`；上游 checkout
  `VBench`（`fd18b3d`）；权重 `models/raft/` 与 `~/.cache/vbench/`；scene Repair 用
  `/root/.cache/clip/ViT-B-32.pt`。**物理卡 6 对 nvidia-smi 可见但对 CUDA 不可用，可用范围 1–5。**
- H100 上一个维度完成、校验并合并后才开始下一个维度；单维度可拆四个隔离 shard，
  每进程只见一张卡并使用逻辑 `cuda:0`；跨卡恢复必须按 `video_uid` 去重并检查完整覆盖。
- 本机测试现状（2026-09-15 实测）：`PYTHONPATH=. uv run --no-sync --group test pytest tests`
  → 117 passed；不加 `PYTHONPATH` 时 4 个 `tests/test_counterfactual_*.py` 因 `import scripts...`
  收集失败（**预先存在**，与本次范围变更无关）。是否用 pytest `pythonpath` 修
  `pyproject.toml:58-60` 由用户拍板，计划里可列为待拍板项。

## 3. 硬约束（违反即返工）

- 不下载权重；不修改 `../VBench`；不 commit/push；不改 `data/ results/ splits/ runs/ figures/`。
- metric 之间不得互相 import；共用基础设施放 `packages/audit-core/`，共用模型适配放
  `packages/audit-models/`，评分公式不进这两个包；`configs/` 只放可审阅配置。
- 每个新维度是独立包：`metrics/<kebab>/` + `src/<snake>/` + 独立 `pyproject.toml` + `tests/`；
  入口先实现 `--vbench/--audit/--both`、`--video/--video-dir`、元数据与默认输出，再谈算法。
- 官方后端必须调用锁定上游的 `compute_*`，不得改上游；provenance 记录上游 SHA 与源文件哈希。
- 报告口径：不得单独引用 pooled CPA；按族、按契约半边分开报；不得无保留宣称 "Repair 更好"。
- 真实模型/CUDA/权重 parity 未验证时必须写明验证方式与未验证范围。

## 4. 计划必须包含的章节

A. 目标与命题（每维要证什么、可证伪的表述）
B. 每维契约与反事实族（族名、变换、期望关系、base 数、检测器资格）
C. 实现拆分（metric 包、Official 适配、审计后端、repair 后端、共享设施改动）
D. 数据与模型（视频来源与共享关系、prompt/元数据、权重清单、无权重降级方案）
E. 计算与调度（H100 卡数/时长/四卡墙钟、分片与恢复、输出路径与 provenance）
F. 验收与报告（单测/合约/parity、CPA 报告口径、失败与不支持率）
G. 风险与待拍板项
H. 里程碑与顺序（含"零成本离线证据优先"的前置里程碑）
I. 交付物清单（文件路径 + 生成命令）

## 5. 每一维必须回答的问题（缺一不可）

1. 命题：要证的是"命名≠实现"还是"严格字符串匹配"？给出可证伪表述与预期方向。
2. 反事实族：族名、变换、期望关系（ordered / same-rank / invariance）、用现成官方视频还是
   必须重新生成、每族需要多少 base、需要什么检测器资格（例如 GRiT 能否在同一帧定位两个目标）。
3. 零成本可判定证据 vs 完整 GPU 测量：哪些结论只用元数据/prompt 改写就能判定
   （`object_class` 的目标串改写、`temporal_style` 的风格子句交换与 32-token 截断、
   `color` 的 `[2][0]→[2][i]` 诊断、`background_consistency` 与 `scene` 共用视频的耦合），
   哪些必须跑模型；两级各给命令与判据。
4. repair 算法：具体改动点（上游 `文件:行` → 修复后行为）、要保留的中间证据、
   与 Official 的对照口径、以及是否属于"评分公式修复"还是"适配器职责"。
5. 模型与权重：需要 GRiT / ViCLIP / CLIP ViT-B/32 / MUSIQ 中的哪些、H100 上是否已存在、
   缺失时的降级路径（mock/合约测试）是什么。
6. 共享关系：`background_consistency ↔ scene`（同一批视频与 prompt suite）、
   `temporal_style ↔ overall_consistency`（同一估计器、不同 prompt suite）——
   必须在结论里声明哪些引用会被同时影响。
7. 失败与不支持率：检测失败、不支持标签、被丢弃样本如何计入分母；明确禁止用删除样本抬分。

## 6. 输出格式要求

- Markdown，中文为主；代码标识符、路径、命令用英文。
- 源码引用写成 `vbench/<file>.py:<line>`，并在计划里注明"行号已于 <日期> 复核"；
  若与 §2 表格不一致，以你复核的源码为准并列出更正。
- 每个里程碑给出：输入、可直接执行的命令、预期产物路径、验收判据、预计机时。
- 末尾给"需要用户拍板的问题"清单（≤8 条，每条附推荐选项）。

## 7. 自检（提交计划前逐条确认）

- [ ] 4 维都覆盖了 §5 的 7 个问题
- [ ] 没有任何"下载权重 / 改上游 / 动冻结目录 / commit"的步骤
- [ ] 每个反事实族都有期望关系、判据与 base 数
- [ ] 无权重环境下不会承诺 GPU 数值结果
- [ ] 明确写出未验证范围、以及每一步由谁（本机 / H100）执行
- [ ] 明确说明哪些结论引用时必须与 review/契约半边一起读

=== 提示词结束 ===
