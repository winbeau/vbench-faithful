# 十一维审计范围决定（2026-09-15）

状态：用户确认的范围决定。本文是**当前审计范围的单一来源**，替代
[`2026-09-14-experiment-plan-8d.md`](2026-09-14-experiment-plan-8d.md) 的"八维"范围表述；
该文的协议内容（契约、反事实族、统计口径、交付物）仍适用于其中的 7 个在办维度。

## 1. 决定

本轮范围 = **11 维**：

**A. 继续执行（7 维）**——已有 metric 包、Official 适配、审计/Repair 后端，并已进入
`counterfactual-vbench` 数据集：

`dynamic_degree`、`motion_smoothness`、`subject_consistency`、`scene`、`human_action`、
`spatial_relationship`、`multiple_objects`

**B. 新增候选（4 维）**——尚无 metric 包；先钉源码身份、保留源码定位与 repair 方向：

`background_consistency`、`temporal_style`、`object_class`、`color`

**C. 退出（1 维）**：`overall_consistency`。

- 理由一：它与 `temporal_style` 是**逐行同一份 ViCLIP 估计器**
  （`diff vbench/temporal_style.py vbench/overall_consistency.py` 只差函数名与 dimension 串），
  取 `temporal_style` 即保留了该估计器的审计覆盖。
- 理由二：它的 prompt 条件需要人工撰写（八维计划 §12.2），本轮无法产出可比条件。
- 理由三：自然集与反事实集都没有它的数据，`figures/supplementary_main_table.csv` 中
  `overall_consistency` 两行一直是 `--`。
- 处置：`metrics/overall-consistency/`、其 CLI 入口、测试与
  `configs/upstream.toml` 里的 pin **保留为 legacy，不删除**（见 §4 未决项）。

## 2. 四个候选的问题与 repair 方向（摘要）

完整源码定位见 [`../paper/unaudited-dimensions-triage.md`](../paper/unaudited-dimensions-triage.md)。

| 维度 | 问题 | repair 方向 |
|---|---|---|
| `background_consistency` | 命名≠实现：整帧 CLIP 相似度，无背景分解；`(prev,first)` 锚点 + 全局帧均值与 `subject_consistency` 逐条同构 | 背景掩码/前景抑制后再编码；锚点改 prev-only 或滑动窗口；复用 subject 的 `temporal_relocation`+median-box 契约 |
| `temporal_style` | 命名≠实现（最硬）：与 `overall_consistency` 同一份代码；只读整条 prompt；风格子句被 ViCLIP 32-token 截断 | 单独编码风格子句（学 `appearance_style` 的 aux 口径）；同 base 的 10 个风格改做相对判别；风格子句前置/分段并记录被截断的 prompt |
| `object_class` | 严格字符串：GRiT 生成类名与目标串做 `set` 精确成员，无归一，per-instance 分数被丢 | 保留全部实例与分数；标签语义对齐（同义/复数/词表 + 不支持显式标注）；分母固定全帧，"未检出"与"叫法不同"分开报 |
| `color` | 严格字符串 + 实例错位：`[2][0]` 只比 top-1 类名、颜色子串、12 色白名单、条件分母 | 修 `[2][0]→[2][i]`（或按 box/IoU 绑定 caption↔实例）；颜色拼写/同义归一；全帧为主分母、条件率降为诊断；未检出不再整条丢视频 |

## 3. 上游映射与 pin

`configs/upstream.toml` 现有 12 条 `[dimensions.*]`（11 维范围 + 退役的
`overall_consistency`），已用 `verify_upstream()` 对 `VBench@fd18b3d` 校验通过
（remote/SHA/clean/12 个源文件哈希全部匹配）。4 条新 pin：

| dimension | module | entrypoint | source | source_sha256 |
|---|---|---|---|---|
| `background_consistency` | `vbench.background_consistency` | `compute_background_consistency` | `vbench/background_consistency.py` | `7def92057819df0dad01c857eecf3b39b8b555c8cdfe881a672cc2b19dc226b4` |
| `temporal_style` | `vbench.temporal_style` | `compute_temporal_style` | `vbench/temporal_style.py` | `9f9e950efca4673721d0a07ea83137be9ebdc9aef2364f412300ba979beef57a` |
| `object_class` | `vbench.object_class` | `compute_object_class` | `vbench/object_class.py` | `8d4c4a7feee52503e3d27f8aba0ed467cbc356ed75da2598367b572237e05db2` |
| `color` | `vbench.color` | `compute_color` | `vbench/color.py` | `03baffc12508fd7ce6c70a227d6794bbc7152c24ba2d2c447a816974466db3d7` |

pin 只声明源码身份，**不代表**已有适配器或 `metrics/` 包。

## 4. 边界与未决项

- 冻结内容不动：`data/`、`results/`、`splits/`、`runs/`、
  `figures/supplementary_main_table.*` 保持原样；该主表仍是历史 8 维交付物，
  `overall_consistency` 两行为 `--`。
- `background_consistency` 读的就是 `scene/` 的视频，并复用同一 86 条 prompt
  （`evaluate.sh:7-10`、`dimension_to_folder.json`）：任何基于 scene 视频的反事实族都会
  同时扰动它，scene 结论必须声明这一耦合。
- `object_class`/`color` 属 GRiT 严格字符串族，而
  [`../semantic-adapter-constraints.md`](../semantic-adapter-constraints.md) 原来只覆盖
  Spatial Relationship / Human Action / Scene / Multiple Objects。**已决定（2026-09-15）**：
  并入该语义适配器范围，扩为**六维**；该文件顶部已加范围更新说明，分维细则待正式动工补齐。
- **未决**：是否立刻为 4 个候选建 metric 包骨架（独立 `pyproject.toml`、
  `--vbench/--audit/--both`、`tests/`），以及先做哪一维。用户 2026-09-15 指示：先暂停代码工作，
  由 Astra 先出开工计划。
- **已决定（2026-09-15）**：`metrics/overall-consistency/` 暂时保留，等用户确认后再删除。
- **未决**：若日后要删除 `metrics/overall-consistency/`，需同步
  `tests/test_workspace_layout.py:6-15,49`、`tests/test_cli_contracts.py:17`、`uv.lock`、
  `README.md` 的目录树，并作为独立变更单独授权。

## 5. 本次改动的文件

- `README.md`：范围段与目录树改为 11 维（7 在办 + 4 候选），说明 `overall_consistency` 退出。
- `configs/upstream.toml`：新增 4 条候选维度 pin。
- `docs/upstream-mapping.md`：入口表补 4 个候选与 `overall_consistency` 的退役标注。
- `docs/paper/unaudited-dimensions-triage.md`：标题与范围段更新，标注入选/未入选。
- `docs/plans/2026-09-14-experiment-plan-8d.md`：顶部加范围横幅，指向本文。
