# VBench Repair

从 `vbench-audit` 的 `winbeau` 分支迁入，保留全部历史与当前研究实现。
代码仓库为私有的 [winbeau/vbench-repair](https://github.com/winbeau/vbench-repair)。
16 维原始数据、独立人类偏好标注及版本化反事实归档发布到
[xju-arlab/vbench-repair](https://huggingface.co/datasets/xju-arlab/vbench-repair)；
选定训练模型按维度发布到 [xju-arlab/vbench-model](https://huggingface.co/xju-arlab/vbench-model)。
整理协议、来源问题和验收见[迁移计划](docs/plans/2026-09-23-vbench-repair-migration.md)。
数据覆盖 16 维不表示本仓库为 16 维都实现了 Repair。

H100 可用 `scripts/h100_python.sh -m dynamic_degree.cli --help` 调用新 checkout，
复用既有 CUDA 依赖，固定上游及已有权重。Python 包和 CLI 名称保持兼容。
原有审计范围、算法状态和历史证据如下。

VBench 1.0 的可复现审计工作区。本轮范围 **11 维** = 已实现的 7 维 + 4 个候选维度
（范围决定见 [`docs/plans/2026-09-15-dimension-scope-11d.md`](docs/plans/2026-09-15-dimension-scope-11d.md)）。
2026-09-20 已另接通 background consistency 的官方与候选修复后端，验证进展见下文；
原范围决定时的七维实现清单保留如下。
Object Class / Color 也已完成独立包后端、共享 Qwen3-8B 的独立 LoRA 与四级消融，
见[实现与复现](docs/object-color-repair.md)及[两维实测报告](docs/counterfactual-reports/object_color_repair_20260920.md)。
Object 的 metadata 改写与 Color 的可见性响应分别报告；Color test 仅 5 个合格基底，
Official 同义控制缺失，人工语义审核尚未完成，不据此宣称总体修复优越。
每个维度都是独立包，Python import 名使用下划线，发行包和命令使用 kebab-case：

```text
metrics/                     # 已实现的 7 维
├── dynamic-degree/          → dynamic_degree / dynamic-degree
├── motion-smoothness/       → motion_smoothness / motion-smoothness
├── subject-consistency/     → subject_consistency / subject-consistency
├── scene/                   → scene / scene
├── human-action/            → human_action / human-action
├── spatial-relationship/    → spatial_relationship / spatial-relationship
└── multiple-objects/        → multiple_objects / multiple-objects
packages/audit-core/         → vbench_audit_core
packages/audit-models/       → vbench_audit_models

后续 4 维（已在 configs/upstream.toml 钉住源码，均已有独立包）：
background_consistency、temporal_style、object_class、color
```

`overall_consistency` 已退出本轮范围：它与 `temporal_style` 是逐行同一份 ViCLIP 估计器，
且其 prompt 条件需要人工撰写（八维计划 §12.2）。`metrics/overall-consistency/` 及其 CLI、
测试暂时保留为 legacy，未删除。
未入选的另外 4 维（`temporal_flickering`、`appearance_style`、`aesthetic_quality`、
`imaging_quality`）的源码定位与不入选理由见
[`docs/paper/unaudited-dimensions-triage.md`](docs/paper/unaudited-dimensions-triage.md)。

## 环境

仓库固定 Python 3.11.14（`.python-version`），uv 0.9.17 在 CI 固定，依赖由已提交的 `uv.lock` 管理：

```bash
# Development (model-free)
uv sync --locked

# CPU scientific/contract tests (separate overlay)
uv sync --locked --group test
uv pip install --python .venv --index https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
uv run --no-sync --group test pytest tests metrics
```

模型运行另起一个同步环境执行：

```bash
uv sync --locked --extra models
```

不要在 CPU 测试 overlay 之后再次同步 models extra，以免替换 CPU wheel。

模型权重、CUDA 驱动和 Detectron2/GRiT 等外部构建不由 uv.lock 提供。本机 CPU 测试不等于真实模型 parity；Subject 的 H100 全量 parity 已在下述扩展报告中逐条核验，其余维度以各自运行报告为准。某些第三方模型构造器在传入默认 pretrained 配置时可能联网下载权重；正式运行应预置本地权重并使用对应的本地路径参数。纯算法、输入输出合约及 CLI help 不需要权重。

## VBench 1.0 官方视频目录

VBench 1.0 的 Dynamic Degree 与 Motion Smoothness 复用 Subject Consistency 的 prompt suite 和 sampled videos。官方人类标注中的视频路径也指向各生成器的 `subject_consistency/`；不能据 `question_en` 的残留文本重新判定维度。远端数据根目录使用以下布局：

```text
<data-root>/<generator>/subject_consistency/
<data-root>/<generator>/dynamics_degree -> subject_consistency
<data-root>/<generator>/dynamic_degree  -> subject_consistency
<data-root>/<generator>/motion_smoothness -> subject_consistency
```

其中 `<generator>` 为 `lavie`、`modelscope`、`cogvideo`、`videocrafter`。`dynamics_degree` 是 VBench 官方维度名，`dynamic_degree` 是本仓库 CLI/包别名；两者均链接到同一只读视频目录。可用 `scripts/link_shared_official_dimensions.sh <data-root>` 建立并校验软链接。此共享关系只适用于视频输入，Dynamic Degree、Motion Smoothness 和 Subject Consistency 仍分别读取各自的人类偏好标注并运行各自评分公式。

## CLI

所有入口都必须显式选择 `--vbench`、`--audit` 或 `--both`，并在 `--video` 与 `--video-dir` 中二选一。示例：

```bash
uv run dynamic-degree --vbench --video /data/video_000.mp4
uv run dynamic-degree --audit --video-dir /data/videos --gpu
uv run dynamic-degree --both --video-dir /data/videos --gpu 0,2,4
```

Dynamic Degree 的新[官方原视频局部纹理抖动实验](docs/counterfactual-reports/dynamic_static_jitter.md)
已接通 CoTracker2 轨迹候选（显式 `--audit-variant trajectory`），不更改旧默认。
按用户最新纠正，使用官方完整原视频，对内部纹理施加小范围快速往返的局部空间位移，
不是 RGB 噪声、亮度频闪或整帧平移。先做固定 5 条开发视频；原始 MP4 直接评分。
五视频 pilot 已完成 40 条：Origin 40/40 有效，Repair 32/40 有效、8 条证据不足。
4 px 长颈鹿双种子出现 Origin 0→1，但 Repair 仍有列车误增与覆盖问题，尚未修复成功。
8 px 同五源 Origin 40%→80%，旧 dev-v4 Repair 仅 4/10 有效配对，尚未修复成功。
用户随后视觉确认这批 8 px 适合作为反事实：后续固定该强度，十条全部纳入开发验收。
旧自动几何筛查的 4 合格/6 拒收仍保留，但不再以 Jacobian <0.5 否定本批人工确认；
确认范围与身份见[复核清单](configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json)。
新的[结构支持运动 Repair 目标](docs/plans/2026-09-22-dynamic-structural-motion-repair-goal.md)
已按最新要求明确为[抖动不敏感与真实运动敏感的联合目标](docs/plans/2026-09-22-dynamic-structural-motion-repair-goal.md#joint-motion-goal)：
固定官方原片与 8px，保留 32 组平均涨幅 ≤Origin 的 10%，同时验证真实运动
识别、强弱排序及运动视频加抖动后不被误压低。Origin 不改，Repair 连续；
仅原片分数不变或 CF 小幅下降都不能单独算完成。短窗跟踪、旧 RAFT 稠密局部对应各测
20 条，CF 有效为 5/10、6/10；已有 torchvision RAFT-large 的 1×/2× 对照各
20 条，均为 8/10，雪地仍不足。时间/结构分解、共同模态/空间保护两候选已在
旧 CoTracker 和新稠密随动轨迹上分别完成消融，仍有误增；全轨迹诊断不冒充
有效 Repair 分数。
最新[机制复核](docs/counterfactual-reports/dynamic_static_jitter.md#机制复核先确认真实运动再设计-repair2026-09-22)
已查看五原片全帧并重放二十缓存：列车原片实际在行驶，但稠密光流也漏掉明显
实体位移；不能以 Origin=0 定义静止，或把 CF 压回错误低值称为修复。时间反向
与局部一致性都不足以单独区分抖动。后续已有 DINO keys/tokens 对照完成同二十
条，直接最近邻位移仍使十对误增，仅为诊断。随后 SAM 自动区域提取已完成
同二十条/320 帧；区域多点联合平移、多个匹配假设及交叉核验已完成二十条/1080
帧对诊断，四分片完整取回并合并核验，运行失败为零、分数均为 null。
列车仍有描述子内部自洽但方向错误的反例，不能晋升。
它还不是完整轨迹/评分，区域覆盖不能替代运动可靠性；完整新 Repair/独立验证未完成。
随后原生 RGB 模板、SAM 目标区域及独立 SIFT/三帧对应诊断完成二十条/1080 帧对，
已合并核验，五组编码对照完全一致，分数仍为 null；列车长间隔错配未解决。
源/目标共同可见性修正减少了自行车案例的背景假运动，全组 v2 已完成并合并
20 条/1080 帧对，0 运行失败；37,653 个区域帧对仍有 3,570 个位移缺失。
列车首帧对恢复正确方向及上述遮挡修正只是局部证据，尚未证明整段运动或不变性。
区域平移/相似/仿射及稳健拟合对照也已完成 20 条，但半数区域帧对缺少稀疏对应，
不能用新增几何模型弥补缺失证据。自然开发 32 条均已作全帧代理复核，8 条 GIF
缺时长；人类审核/尺度校准未完成。列车单帧对的同视频 SAM 提示重定位已实际
恢复车体，但最低 RGB 误差仍选近零运动；独立几何检查只有一个空间见证点，
尚不足以证明恢复。共同 SIFT 排序全组也已完成，23,790/37,653 区域帧对无法
排序，全部视频 score=null；这些诊断不增加有效 Repair 分数或完成验收。
随后三种稀疏对应/三帧身份检查已完成同二十条；自动结构点 CoTracker 与各 SAM
区域放大跟踪各完成列车四版本，仍未恢复连接处的真实位移，不能晋升为 Repair。
原生局部外观及平移/仿射细化各完成列车四版本的一个帧对：能恢复个别连接处，
但全部点在 8px 下仍大量错配；全部仍为 null 分数，完整 Repair 未通过。
跨区域空间预测和区域×时间交叉预测又各完成同二十条：列车/马/长颈鹿干预有
可预测的共同往返分量，但自行车不一致、雪地证据严重不足；尚未据此抑制运动
或输出新 Repair 分数，自然周期运动保护未验证。
后续边界检查和区域运动保护分解均完成二十条；分解 v1 的小区域数值保护问题
已修正并以 v2 全组重跑、独立核验。六条干预的条件运动量下降约 29–48%，但
仍明显高于原片，全部 score=null，不是成功 Repair。数学部件/镜头保护成立
不等于自然运动验证，列车真实对应与雪地证据仍需解决；详见报告最新小节。
再后续独立外观峰/源自相似与区域联合候选已完成列车四版本的全部 60 个相邻
帧对；正确位移得到更多查询位置支持，但重复车窗的近零错配仍可获较高裕量，
两种权重/位置折仍分歧。全部 null 分数，不升级为真实运动恢复或 Repair 成功。
后续三帧重检关联和固定源模板各完成同四版本的全部 56 窗，并独立核验全部
路径/排名及每组首选原图证据；编码对照精确一致。取消中间特征重检依赖后仍
会跳到相似纹理，三帧 NCC 不能认证身份，完整 Repair 仍未通过；详见报告
[三帧路径诊断](docs/counterfactual-reports/dynamic_static_jitter.md#三帧关联与固定模板路径重检断链不是唯一问题)。
再后续源模板之外的同区域不相交环带检查完成四版本/全部 60 帧对；完整几何/
排名及所有组首选原图检查通过，但扩大平移支持会压过连接处移动候选，并有
出画缺失，不能晋升为抖动删除门槛。全部仍为 null，见
[空间证据与负例](docs/counterfactual-reports/dynamic_static_jitter.md#不相交空间证据检查扩大平移支持仍会丢掉运动)。
最新线段/交点对应完成全部二十条、320 帧、1080 帧对，像素/几何/身份及五组
编码对照核验通过；但干预下匹配大量丢失，七条 CF 的相邻交点没有第三帧支持，
列车已检查的移动连接处仍漏检。全部 null 分数，不以缺失/剩余点制造低分，见
[线段与交点负结果](docs/counterfactual-reports/dynamic_static_jitter.md#线段与交点对应完整五源的稀疏几何仍不足)。
CoTracker3 随后经用户明确授权下载并完成实测：固定权重/源码核验及真实加载
parity 通过；同二十视频全部十六起点的 SIFT/固定网格两臂共 640 组，0 失败，
160 组编码对照精确一致。既有保护分解的条件诊断 MAE 为 0.044164→0.034739，
但十条 CF 仍全抬高，列车已检查连接处仍漏检，不能称为成功 Repair。默认评分
未变，正式留出未打开。结果、计时及局限见
[CoTracker3 实测](docs/counterfactual-reports/dynamic_static_jitter.md#cotracker3-实测双查询臂完成但仍未修复)。
用户随后明确按 **32 组批量平均分**验收，并纠正为 **Origin 保持原版，只有
Repair 使用连续运动强度**。已按授权将缺时长 GIF 替换成官方开发 MP4，完成
32 源/128 输入及完整缓存重放：Origin 0.593750→0.781250（动态比例），Repair
0.255291→0.365000（短边/秒）；32 组编码控制完全一致，Origin 记录未改变。
Repair 仍明显误增，相对同单位无抑制消融仅减少 8.06% 的平均误增，未修复成功。
旧二值 Repair 0.6875 不再是当前主结果。两种原始指标量纲不同，不能直接相除。
后续已用 prompt/UID 不重叠的 63 个官方自然开发 MP4 校准连续 0–1 映射
`I/(I+0.1268495331)`，原片、CF 同尺度、无拟合偏置。32 组校准后首版 Repair 为
**0.560659→0.683035**，Base 比 Origin 低 0.033091，未达到“略高”；涨幅为
Origin 的 **65.27%**，仍未通过 10% 目标。63/63 校准评分零失败，未读取正式留出，
原强度与 128 条 Origin 保留。见[独立尺度校准](docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-calibrated-scale)及
[32 组连续强度结果](docs/counterfactual-reports/dynamic_static_jitter.md#最新32-组官方-mp4origin-原版repair-连续运动强度)。
随后按用户要求进一步降低抖动误增，新增“先分离受保护运动、再检查残余往返”
候选，在同尺度下完成全部 128 输入和 63 自然控制的缓存重放。CF 均分从
0.683035 小降到 **0.681470**，32 个 Base 仍为 **0.560659**；95 条原片分数
逐条不变，64 CF 中 7 条下降、57 条不变。涨幅仍为 Origin 的 **64.43%**，
并未达到 10%；仅开发候选、不改公开默认。见[残余分解结果与保护检查](docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-residual-reversal)。
联合目标恢复后，又完成最大SAM区域支持分区的191条消融：
**0.560659→0.678265**，涨幅仍是Origin的**62.72%**，95原片不变但5条CF相对
上一版升高，仍未通过。自然偏好覆盖仅23对（4有序/19平局），有序为Origin3/4、
候选4/4，不能据此宣称自然运动验证完成；本轮9条全帧代理复核另记，短时车辆
通过不能当成静态。见[实测、覆盖限制与下一步](docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-outer-support)。
随后按用户授权完成[冻结V-JEPA 2.1＋小头有限开发试验](docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-probe)：
210训练/60开发验证原片、810特征、零失败；自然监督与联合一致性两个51K小头
各固定100步。联合臂验证相对分数0.566153→0.564112，绝对变化比自然臂小33.31%；
原片运动偏好20/23，干预19/23。仅预定开发可行性通过；未经绝对强度校准，
当时未测主32组及正式留出，不能由开发结果宣称通过；默认评分不变。
最新按用户要求完成[真实VBench 1.0冻结反事实验证](docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-validation)：
实际新构造90原片的90编码控制＋180条8px局部坐标干预；同时复测已有32组，
并评价450条自然测试视频。Origin、V-JEPA各848/848输入，0失败，122编码控制精确一致。
32组Origin/联合Repair分别为**0.593750→0.781250 / 0.515726→0.508504**；
留出90组为**0.700000→0.900000 / 0.517112→0.514804**。冻结相对分数上的10%涨幅
数值检查通过，没有再训练或缩分。自然171个有序对的半平局计分concordance为
81.87%→85.96%，改善+4.09个百分点的CI为[−0.97,+9.12]，未证明显著优于Origin。
30个预留GIF因固定原生MP4协议NOT SCORED；有CF的有序偏好仅6对。
绝对运动强度标定及运动类型人工审核仍未完成，45校准预留未读取，完整goal未完成。
测试prompt现已打开，不能再作为未见留出循环调参；旧合成/噪声/轨迹负结果仅作历史。
后续按用户要求完成[450源/900CF覆盖扩展](docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-expansion450)：
两后端各1800/1800，Origin **0.680000→0.857778**，旧联合Repair
**0.518089→0.516319**；450个人类配对的两种CF均已覆盖，不变性数值结论保持。
但[静态图片诊断](docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-static-frame)
显示旧头静止0.456649、轻微平移8px反降至0.442835，不能称为合格运动强度评分。
用户明确要求调整模型，现正执行[静止/位移锚点监督的小头重训练](docs/counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-anchored)，
仍冻结编码器，DEV210/60、不用该单例或TEST450训练、不作分数平移/缩放。
首次大张量训练发生非有限loss，失败记录保留；以相同数据/目标/300步、等价分块
执行恢复训练，尚不预先声明新模型通过。默认不变。
构造、配置和命令见[复现入口](configs/dynamic-static-jitter/README.md)。

裸 `--gpu` 和不带值的默认选择是当前 CUDA 可见逻辑设备 0；显式列表拒绝重复或越界编号，不会静默切换 CPU。默认结果写入仓库根 `output/<metric>/<backend>/<run-id>/`，每次运行使用新 run-id；`--output DIR` 可替换输出基目录。输入 `data/`、`results/`、`splits/`、`runs/` 为冻结研究内容，不修改、不重算、不删除。

## 规范与文档

开发边界和测试要求见 [`AGENTS.md`](AGENTS.md)，贡献、uv extras 与提交规则见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。架构、CLI、依赖和上游映射分别见 [`docs/architecture.md`](docs/architecture.md)、[`docs/cli.md`](docs/cli.md)、[`docs/dependency-compatibility.md`](docs/dependency-compatibility.md) 与 [`docs/upstream-mapping.md`](docs/upstream-mapping.md)。实施计划在 [`docs/plans/2026-09-14-workspace-refactor.md`](docs/plans/2026-09-14-workspace-refactor.md)。

全部维度、历轮实验、分母/拒收、Origin/Repair、置信区间、版本、输出路径和
未完成项的统一入口是[实验结果总览与复现索引](docs/EXPERIMENT_INDEX.md)。

本轮实际通过的检查及未验证范围见[重构验收记录](docs/plans/2026-09-14-workspace-refactor-verification.md)。补充实验计划与 H200 执行口径见[补充实验计划](docs/plans/2026-09-14-supplementary-experiments.md)。当前范围（11 维）见[十一维范围决定](docs/plans/2026-09-15-dimension-scope-11d.md)；其中 7 在办维度的总体实验协议见[八维实验协议](docs/plans/2026-09-14-experiment-plan-8d.md)；反事实数据集（VBench-CF）的构造口径见[反事实数据集](docs/counterfactual-dataset.md)。

补充实验的可直接执行命令、H200 四卡计时、8 卡恢复记录和当前主表结果见
[补充实验运行手册](docs/supplementary-experiments.md)。主表交付物位于
`figures/supplementary_main_table.{csv,tex,pdf,png}`；逐视频研究输出仍写入
被 Git 忽略的 `output/supplementary_20260914/`，避免把大量中间缓存提交到仓库。

Subject 的表示层修复与完整背景糊化实验见 [数据、协议与复现说明](docs/subject-repair.md)。
[Subject 背景干预稳定性修复](docs/plans/2026-09-20-subject-stability-goal.md)已按用户最新验收要求收尾：
开头背景糊化时，Origin 明显变化、repair 变化约 0.01，且保留主体变化响应；人类偏好为辅助诊断。
[本轮真实运行报告](docs/counterfactual-reports/subject_stability_20260920.md)：官方 **720 条、全部 72 个 prompt、每生成器 180 条**的主实验已完成，241 条数值构造全部评分、0 运行失败。预定 240 条主分析的 Origin/repair 平均绝对变化为 **0.120035/0.010596**，主体平均降分 **0.099319**；背景联合成功 74/240，加主体降分 ≥0.05 为 70/240。用户看到结果后接受“0.01 左右”的均值表现，原严格 ≤0.01 统计仍保留为未通过。479 条构造拒收、最坏 repair 分差 0.149715 均保留。用户随后要求快速收尾，补充单帧评分在 204/241 条完成时停止，部分输出完整保留，不作为全量结果。旧 60 条单首帧的 0.102377/0.001275 与偏弱主体响应均保留，不能替代新主实验。
此前[官方扩展实验](docs/counterfactual-reports/subject_official_extension_20260920.md)已完成 1440 条自然视频、2160 对人工偏好：
test 1290 对上 origin 为 58.53%，原有聚合 repair 为 59.61%，自动定位＋隔离裁剪为 48.76%。
初轮候选整体退化；不能由两条背景示例宣称修复成功。所有官方/原聚合分数分别与历史值在 1e-6 内对齐。
后续保持粗掩码、改用 CLS 并在主体证据不足时回退 origin：统一零平局容差下，test 为 57.05%，origin 为 58.45%。
34 条完整背景干预的平均分差降到 0.053575（origin 0.060218），但配对改善区间跨零，主体响应仍有不足；尚未达到优秀修复。
背景构造也已扩到 288 条候选，其中 34 条通过自动门槛；类别不支持和构造门槛限制了覆盖。
后续接受大体正确的主体粗定位，不因少量边缘误差反复暂停实验；重点检查表示、缺失帧处理和整体效果。
当前 v2 保留主体像素、模糊主体掩码的全部补集，独立 MobileSAM 评分定位与建库掩码隔离；
历史镜像盒协议只归档重放。真实运行结果与质量限制单独报告，未验证项目保留 NOT RUN。
当前 [7 条试跑](docs/counterfactual-reports/subject_region_discrimination_v2.md) 因图像与建库掩码质量不足，
不作为正式主实验；用户只保留演讲者和游泳者为候选，后续先做原图质量筛选。
编码前主体隔离与规范化裁剪已作为 `subject_isolated` 候选变体实现，
[开发消融](docs/counterfactual-reports/subject_isolation_development.md) 显示中位背景扰动减小，
但均值、最坏情况和独立验证仍未达标，尚不宣称修复效果优秀。
后续 [咖啡、吉他两条新候选](docs/counterfactual-reports/subject_isolation_quality2.md) 已完成真实评分：
背景平均绝对分差从 origin 的 0.01449 降至 0.00591，局部时窗主体干预 6/6 降分；
咖啡评分掩码仍漏掉部分身体，两条样本不足以支持总体修复成功的结论。

此前的 [Background Consistency 修复与主体糊化实验](docs/plans/2026-09-20-background-repair-goal-prompt.md)：
糊化主体、保留背景，对比 origin 与 repair 的干预前后分数，同时验证背景变化响应与自然偏好表现。
background 已完成真实 CLIP 后端与独立验证，[正式报告](docs/counterfactual-reports/background_holdout_20260920.md)
覆盖 1,040 条自然测试视频、1,560 对偏好及 188 条基底的 1,504 个干预版本。
完整主体糊化平均绝对分差从 origin 0.016240 降至 repair 0.006733（降低 58.54%）；
自然偏好准确率从 52.12% 提高到 61.22%，三个时窗平均的背景响应保留 88.80%。
18 项冻结门槛全部通过，默认 `repair` 已指向背景 patch 池化与全帧对方案；真实 CLI parity 误差为 0。
仍有单片段失败与粗定位局限，所有旧候选负结果保留在[开发报告](docs/counterfactual-reports/background_development_20260920.md)。

后续构造语义检查发现旧干预存在误选和漏分，以上数值通过不能证明所有编辑正确。
按用户要求，background 原生开发集 680 条已用同一 SegFormer + GrabCut 流水线扩大到 104 类重构：
全空从 450 降至 89，数值构造通过从 7 增至 49，待复核项目显式保留。
评分端的 COCO80 与既有 MobileSAM 自动补充分支仍属开发候选；完整结果、逐步复核范围和未达标项见
[空掩码审计与类别扩展](docs/counterfactual-reports/background_empty_masks_20260920.md)。
后续复用 GRiT 区域描述与 MobileSAM 的评分定位候选，已完成 680 条自然开发视频和 1,020 对偏好：
origin 52.35%、现有 repair 57.84%、COCO80 57.94%、Caption v2 56.67%。
Caption v2 的检出覆盖改善，但未证明对现有 repair 非劣，因此保持开发候选；
详见[定位修复与全量自然评估](docs/counterfactual-reports/background_caption_localizer_20260920.md)。
104 类重构后的 49 条数值候选在暂停时已逐次复核 30 条：12 条可用、18 条拒收、19 条待复核。
十条已完成干预对照，origin / 现有 repair / Caption v2 的全片主体糊化 MAE 为
0.011578 / 0.003054 / 0.002704；origin 达到 0.10 或 0.20 的比例均为 0/10。
仍有非空但错位的评分掩码，类别扩展改善了构造覆盖，尚未达到新的联合修复目标。
按用户要求暂停实验，新增两条接受项尚未评分；[阶段总结与紧凑证据](docs/counterfactual-reports/background_repair_checkpoint_20260920.md)
记录暂停状态、全部已完成结果和剩余工作。多物体 union 仅完成输入探测，未晋升默认。
