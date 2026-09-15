# VBench Audit 补充实验执行计划

日期：2026-09-14。目标是利用已生成的 LTX-2 视频和冻结的 VBench 人类偏好标注，补齐八维 Official/Repair 对照、主表和补充敏感性实验。源码、脚本和新结果留在当前工作树；不改写 `data/`、`results/`、`splits/`、`runs/` 中已经冻结的输入或历史结果，不下载权重，不修改上游 checkout。

## 1. 交付物与固定口径

1. 新运行目录使用 `output/supplementary_20260914/`，保存 manifest、逐视频分数、pairwise 统计、coverage、失败原因、远端命令和 provenance。已有的 `results/e0/` 与 `runs/` 内容只读作为历史/冻结基线。
2. 主表 CSV 固定为 16 行、4 列（不计表头）：8 个维度各两行 `official`/`repair`，列为 `dimension,variant,vbench_score,human_preference`。`vbench_score` 是该维度自然集的分数均值（同一支持集优先，同时报告 coverage）；`human_preference` 是冻结 E0 test split 的 pair accuracy，另存 tie-aware accuracy、Kendall tau-b、Pearson 和 CI，避免把一个数字误解成完整统计结果。
3. 论文 TeX 固定为 8 行、4 列（不计表头）：`Dimension | VBench score | Human preference | Coverage`。每个数值单元格写成 `official→repair`，后接方向箭头；这轮所有主指标都采用“越高越好”并标 `↑`。使用 `booktabs` 三线表、Times New Roman 字体配置，宽度 7 inch；同时编译 PDF 并导出同宽 PNG 到 `figures/`。
4. 表格生成必须从 CSV/JSON 统计结果读取，禁止手填数字；缺失或未运行结果写 `--`，不把 failed/unsupported 当作 0。表格脚本同时校验 16 行/4 列、8 行/4 列和 Official/Repair 配对完整性。

## 2. 数据与远端资源

- Natural Set：读取 `data/processed/pairwise_master_split.csv` 和已有 `results/e0/raw_official_scores/`；四个已缓存维度先直接计算，后四维只有在视频、metadata、模型和结果完整时才进入最终汇总。VBench 1.0 的 Dynamic Degree 与 Motion Smoothness 按官方协议复用四个生成器的 `subject_consistency/` 视频，在数据根目录建立可审计软链接；两维仍使用各自的人类标注和评分公式。
- Repair：复用 `scripts/run_official_dataset_compare_dimension.py` 与现有 audit CLI，统一输出 `repair_status`、`repair_score`、失败原因和运行环境。执行严格按维度串行：一个维度完成并校验后才开始下一个；单个维度内部固定使用 H200 物理 4–7 卡四张卡并行，每个 worker 用单卡 `CUDA_VISIBLE_DEVICES` 隔离并在进程内使用逻辑 `cuda:0`，使用独立输出目录和可恢复分维运行。
- LTX-2：只读审计 H200 侧 `/data/chenjiayu/wenbiao_zhao/LTX-2/tasks` 的文件、metadata、hash 和 Hugging Face 任务清单；将能与八维 metadata 对齐的生成视频复制/软链接到新运行目录的输入索引，不改 LTX-2 checkout 和原始视频。
- H200 在执行前记录 `nvidia-smi`, 当前用户和 PID、显卡 UUID、代码/环境版本。下载/评分只使用物理 4–7 卡；若卡被其他用户占用，停止在这些卡上提交任务，保留可复现命令和阻塞记录；不得终止或发送信号给他人进程。

## 3. 八维自然集 Official/Repair

按 TODO 中的冻结协议执行：同一视频、同一帧采样、同一 metadata 和 checkpoint 对比 Official 与 Repair；逐视频记录 success/unsupported/failed。对每个维度计算：自然集 test zero-margin pair accuracy、dev 校准 tie-aware accuracy、Kendall tau-b、model-level Pearson（描述性）、coverage、cluster bootstrap 95% CI，以及 Repair−Official 差值。动态、空间、主体和动作先覆盖已有 E0 资源；Scene、Multiple Objects、Overall Consistency、Motion Smoothness 逐项检查 manifest 和依赖，满足条件才运行，不用伪造的空结果填表。

## 4. 补充敏感性实验

新增一个确定性实验 manifest/generator，所有派生视频记录 `base_id`、`intervention_family`、`level`、`expected_relation`、参数、输入/输出 hash、FPS/帧数/时长和代码 SHA。每个维度至少包含正向增强、反向减弱和不敏感稳定对照：

- Dynamic：全局相机运动强度增大与减小、FPS 重采样稳定性；分别看 CAMERA 单调上升和 SUBJECT 不变。
- Subject：主体 corruption 强度增大/减小，位置（首/中/尾）交换；报告 corruption 敏感性和位置不变性。
- Human Action：正确/相关错误/无关 target，及 action 覆盖比例正向增大和反向缩小；文件名改名保持字节不变。
- Spatial：水平/垂直翻转与反向恢复、role swap；同时保留 reciprocal 稳定对照。
- Scene：目标环境 0/25/50/75/100% 的网格覆盖和反向缩小；孤立局部 cue 作为不敏感控制。
- Multiple Objects：弱目标遮挡/模糊强度增大与减小；只在同一帧共现的版本和时间错开的版本作为 conjunction 稳定/失败控制。
- Overall Consistency：替换 0/1/2 个条件；正向恢复条件和反向破坏条件成对运行，报告 violation count 单调性。
- Motion Smoothness：时间扰动等级增大与减小；均匀 FPS 和平滑加速作为不敏感控制，另测单个尾部 jerk。

统计使用每个 `base_id` 的 macro mean、Spearman/strict-order、CPA、相对范围或 CV；正向和反向分别报告，稳定控制报告绝对差、CV 和非劣性区间。开发集用于调阈值，测试集只使用冻结参数。没有真实模型/输入的维度只生成 manifest 和 blocked 记录，不把 model-free 合约测试当作经验结果。

## 5. 实现和验证顺序

1. 先落盘本计划；由 Luna 按本计划实现可恢复的统计/表格/敏感性脚本，保持现有包边界。
2. 在 H200 的 `wenbiao_zhao/vbench-audit` 完成 `uv sync --locked` 和模型运行所需的 `uv sync --locked --extra models`；只读核对 LTX-2 tasks 和 HF 元数据，按远端 `<model-pack>/<dimension>` 的精确目录下载官方视频。Dynamic Degree 与 Motion Smoothness 是已确认的官方例外，使用 `scripts/link_shared_official_dimensions.sh` 链接到同生成器的 `subject_consistency/`。核对 H200 4–7 卡状态，空闲后按“单维度完成→校验→下一维度”的顺序启动四卡 worker。
3. 先用已有 Official 缓存生成主表 smoke；再接入 Repair 结果和八维自然集统计；最后生成补充 manifest、可运行命令和已完成维度的统计。
4. 运行受影响的纯算法/合约测试、`git diff --check`、表格 schema 校验；若改动工作区接口，再按 AGENTS.md 执行 `uv lock --check`、`uv sync --locked`、测试和八入口 `--help`。
5. 使用 XeLaTeX 编译 7 inch TeX；若系统没有 Times New Roman，记录字体 fallback，不把 Nimbus/Liberation 的渲染冒充 Times New Roman。生成的 PDF/PNG、CSV、TeX、JSON 和 README 放入 `figures/` 或新运行目录并互相记录来源。

H200 计时 smoke 已完成：`group_id=310` 的四生成器视频各 1 个，共 4 个视频，在物理 GPU 4--7 并行；三个 MP4 为 2.0 s，CogVideo GIF 为 33 帧且没有 duration 元数据。端到端（含模型加载）四 worker 用时为 16.731/16.828/16.840/16.842 s，墙钟总耗时 16.842 s；完整 JSON 在 `output/supplementary_20260914/timing/group_310/timing.json`。

## 7. 已完成 checkpoint（2026-09-15）

Dynamic Degree、Subject Consistency 的四卡 repair 已完成，Motion Smoothness
先完成 4 卡首批 722 条，随后在用户明确要求下使用 H200 0--7 卡并行完成剩余
718 条；后者与已有 `sglang` 进程共存，没有终止其他用户进程。Motion 合并覆盖
1440/1440，全部 `succeeded_scalar`。当前自然集表格数值为：

| Dimension | Official VBench | Repair VBench | Official human | Repair human |
|---|---:|---:|---:|---:|
| Dynamic Degree | 0.624 | 0.255 | 0.684 | 0.590 |
| Subject Consistency | 0.898 | 0.909 | 0.585 | 0.596 |
| Motion Smoothness | 0.952 | 0.753 | 0.636 | 0.395 |

`figures/supplementary_main_table.csv`、`.tex`、`.pdf` 和 `.png` 是由统计
输出自动生成的当前快照；其余维度仍按 blocked/未接入状态保留 `--`，不填充
伪造分数。跨分片恢复使用 `scripts/combine_partial_repair_shards.py`，通用
启动使用 `scripts/launch_dimension_shards.sh`。

## 6. 结果边界

最终报告明确区分已完成、blocked、unsupported、failed 和未验证的真实模型 parity。H200 上其他用户的进程不会被 kill；若资源持续被占用，交付脚本、冻结命令、输入审计和已有缓存统计，并标记剩余 GPU 运行未完成。
