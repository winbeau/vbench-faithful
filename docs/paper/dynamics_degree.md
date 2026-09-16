# dynamics_degree —— Origin vs Ours（论文素材）

**范围。** 官方实现读自锁定上游 `VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`，
文件 `vbench/dynamic_degree.py`（164 行，sha256 `2e93d3a7…`，见 `configs/upstream.toml`），
行号均指该修订。我们的实现读自本仓库 HEAD `3677dbf`：
`metrics/dynamic-degree/src/dynamic_degree/`、`scripts/counterfactual/`。
实验数字标 `[报告]` 者来自 `docs/counterfactual-reports/`（报告代码 SHA `5c4a1309`），
标 `[复核]` 者由本轮在 h100-server 上对冻结分数树重算
（`scripts/counterfactual/verify_dynamics_review.py`，CPA 的 margin/CPA/配对 bootstrap
为不 import `scripts.counterfactual` 的独立实现，与仓库实现逐位一致）。
本维 nuisance factor：**时间采样率**。

## 1. Origin —— 完整 reduction（video → scalar，含伪代码与公式）

```text
# 抽帧  vbench/dynamic_degree.py:96-120
fps      = CAP_PROP_FPS(video)                    # :99
k        = max(1, round(fps / 8))                 # :100
frames   = decode_all_frames(video)               # :101-109（先全解码）
frames   = frames[0::k]                           # :112 -> :116-120
# 参数  :59-61（count 为抽帧后的帧数）
thres    = 6.0 * min(H, W) / 256
count_num= round(4 * len(frames) / 16)
# 逐对 RAFT  vbench/third_party/RAFT, 20 iters  :74-79
for (a, b) in zip(frames[:-1], frames[1:]):
    flow        = RAFT(a, b)                      # :77
    rad         = sqrt(u^2 + v^2)                 # :46-48
    cut         = int(H*W*0.05)                   # :52
    max_rad[i]  = mean(top-5% of rad)             # :54  （最大的 5% 的均值）
# 视频级判定（布尔）  :84-93
moving = False; c = 0
for s in max_rad:
    c += (s > thres)                              # :89
    if c >= count_num: moving = True; break       # :91-92  （提前退出）
# 数据集分数  :141-149
score = mean([moving(v) for v in videos])          # :148  （动态视频比例）
```

形式化。设原生帧率 `f`，抽帧间隔 `k = max(1, round(f/8))`，则相邻被抽取帧的时间步为

```text
Δt = k / f   （秒）                                          (1)
m_i = top5%mean(‖F(p_i, p_{i+1})‖)   （像素）                 (2)
b_i = 1[m_i > τ],  τ = 6·min(H,W)/256                        (3)
S   = (1/N) Σ_videos 1[ #{i : b_i = 1} ≥ round(4n/16) ]      (4)
```

**Δt 的两段行为（代码事实）。** `k = 1 ⟺ round(f/8) = 1 ⟺ f ∈ [4, 12)`；`f < 4` 时
`round(f/8) = 0` 被 `max(1, ·)` 夹到 1。因此：

| 原生帧率 `f` | `k` | `Δt = k/f` | 备注 |
|---|---:|---|---|
| `f ≥ 12` | `round(f/8)` | `≈ 1/8 s`，仅在 `f` 是 8 的倍数时精确等于 1/8 | `f=12 → k=2 → Δt=1/6`（+33%）；`f=20 → k=2 → Δt=0.1`（−20%） |
| `4 ≤ f < 12` | 1 | `1/f`（**不再有下界**，`f→0` 时无界） | `f=8 → 0.125`；`f=10 → 0.100` |
| `f < 4` | 1 | `1/f` | `f=2 → 0.5 s` |

即官方所谓的"归一到 8 fps"只在原生帧率为 8 的倍数时成立；**`f < 12` 时根本没有
重采样**，步长就是 `1/f`。这不是推断而是 `:100` 的分段性质。实测（h100 上对冻结源
视频用 `cv2` 读取，与 `:99` 同一 API）：CogVideo GIF 报 `10.000 fps`、33 帧 → `k=1`、
`Δt = 0.100 s`；LaVie/ModelScope mp4 报 `8.000 fps`、16 帧 → `k=1`、`Δt = 0.125 s`。
**同一份 VBench 1.0 包内，"8 fps 归一化"给出两个不同的时间步（0.100 s 与 0.125 s，
相差 25%）。**

**输出类型（代码事实）。** 单视频输出是布尔（`:80-81`、`:84-93`，且一旦累计到
`count_num` 就 `return True` 提前退出），数据集分数是这些布尔的均值（`:148`），即
"被判为动态的视频比例"；幅度信息在 `:89` 的阈值比较后全部丢弃。附带边界事实：
`count_num = round(4n/16)` 在 `n ≤ 2` 时为 0（`round(0.5)=0`，银行家舍入），此时
`:91` 的 `c >= 0` 首次迭代即真 → **两帧及以下的视频无条件判为动态**。

## 2. Origin —— 坏在哪里，以及为什么它就是本维的 nuisance factor

**(a) 代码位置（代码事实）。** 量纲与时间无关的阈值 `τ`：`:61`（纯像素）；从不除以 `Δt`：`:74-79`
直接对相邻抽取帧取 `max_rad`；布尔化丢幅度：`:84-93`；抽帧分段：`:100`。

**(b) 形式化（由 (a) 推出的推断）。** 设轨迹在间隔 `Δt` 上的位移服从幂律 `E[m] = A·Δt^p`（`p=0` 才是不变
量，匀速直线运动 `p=1`）。代入 (2)–(4)：`S` 通过三处依赖 `Δt`——(i) `m` 随 `Δt^p` 缩放；
(ii) `τ` 是固定像素数，跨阈值概率随 `Δt` 单调变化（`Δt` 越大越容易判为 moving）；
(iii) 判据是**采样帧数的比例** `round(n/4)`，与 `n` 无关，但每个 `b_i` 的跨越概率
依赖 `Δt`。因此 `S` 是 `Δt` 的函数，而 `Δt = k/f` 在 `f < 12` 时随 `f` 无界变化。
本维反事实阶梯恰好落在 `k=1` 段：`Δt = 1/8, 1/6, 1/4, 1/2 s`（比值 1 : 1.33 : 2 : 4）。

**(c) nuisance factor（命名）。** 时间采样率。官方 reduction 把"运动有多快"与"我们隔多久看
一眼"纠缠在一起，且（由实测的生成器差异）连 `Δt` 本身都不是固定常数。

**(d) 可测预测。** 代码事实推出的三条预测，全部可检验：
1. *逐档膨胀*：同一轨迹在低帧率档应得更高分，且档均值应满足 `S(Δt) ∝ Δt^p`（`p>0`）。
   匀速合成内容应给 `p≈1`，不变内容 `p=0`。
2. *阈值翻转*：随 `Δt` 增大，部分视频应从"static"翻到"moving"。
3. *控制*：若把 `Δt` 固定、只换编码（同一 rung 内重编码），分数不应有系统差异。

已核实：预测 1 成立——官方档均值（test）`18.42 → 22.04 → 26.47 → 35.20`，拟合
**`p = +0.458`**（pooled `+0.491`）`[复核]`；预测 3 成立——同一 0.5 s 轨迹跨度上，2 fps
档的 `lag-1` 位移与 8 fps 档的 `lag-4` 位移相差 ≤3.5%，即档间差异来自采样间隔而非
重编码 `[报告 §5]`。预测 2 **未测量**：冻结的反事实分数只保存 `mean(raw_flow_top5_mean)`（连续代理，`score.py` 的 `_first()` 丢弃了逐 transition 的 `official_moving_flags`），不重跑 RAFT 无法复原布尔输出；自然偏好集的官方分数虽是布尔，但其 10 fps 子集（CogVideo GIF）与 8 fps 子集同时对应不同生成器，Δt 与内容不可分离。故 (d)2 属**未验证推断**。

**必须区分的推断。** `Δt = 0.100 s`（CogVideo）与"Boolean 翻转"的因果链含两个推断：
OpenCV 对 GIF 报的 10 fps 是否等于真实回放节奏（我们自己的 GIF 探测用 100 ms 约定，
两者一致，但这是约定而非容器内元数据）；以及 `round(n/4)` 判据下翻转的具体比例。
`f < 12 ⇒ k=1 ⇒ Δt = 1/f` 则是纯代码事实。

## 3. Ours —— 合约、修复算子，以及它没有修好的部分

**合约。** `fps_resampling` 把同一条轨迹重采样到 8/6/4/2 fps（时长固定），声明
**不变性**：`expected_rank = 1` 对所有档相同（`scripts/counterfactual/transforms.py:31`
`FPS_LADDER`、`:34` `fps_resample`、`:46` `expected_rank=1`；
`docs/counterfactual-dataset.md:15`）。保留帧是源轨迹的严格递增子集，只降采样
（`scripts/counterfactual/common.py:316` `resample_indices`）。合约的可判定形式就是
**档间 profile 的带符号 log-log 斜率 `p = 0`**。

**v1（归档）算子。** `d / dt`（`metrics/dynamic-degree/.../backends/audit.py:181`
`normalized_speed`），即弹道假设 `α = 1`。它把 `S ∝ Δt^p` 变成 `Δt^{p−1}`：当
`p ≈ +0.5` 时得到 `Δt^{−0.5}`，**把违约镜像成反方向**，实测 `p = −0.511`。

**shipped v2 算子。** `d / dt**α`，固定默认 `α = 0.5`（`audit.py:49-60`
`LagExponentMode.FIXED`、`:82-85` `default_lag_exponent = 0.5` 及其 provenance 字符串、
`:295-321` `resolve_time_normalization_exponent`；`motion.py:42` `normalized_intensity`、
`:68` `scale_intensity`）。同一归一化域还用于静态/运动阈值与 coverage
（`audit.py:119` `derive_motion_threshold(..., exponent)`、`:511` 传入解析后的指数），
使 (3) 的比较变成 `official_px/diag/0.125**α` 对 `d/dt**α`，从而与该域的分数同量纲。
此外每档仍测量 clip 自身的 lag 指数作为**诊断**（`audit.py:192` `_measure_lag_scaling`、
`motion.py:77` `resolve_lags`、`:93` `fit_power_law_exponent`、
`schemas.py:67` `LagScalingEvidence`，含 `straightness = chord/path`）。

**为什么 α=0.5 能消掉第 2 节的问题。** 由 `E[m] = A·Δt^p`，
`m / Δt^α = A·Δt^{p−α}`，当 `α = p` 时该量与 `Δt` 无关，于是 `S` 的档间 profile 斜率为
0。实测内容律 `p ≈ +0.5`（标签均值 +0.49；逐 clip 均值 +0.51 `[复核]`），故取 `α = 0.5`。
对照：`α = 0`（不归一化，等价于官方对 `m` 的使用）给出 `p = +0.458`；`α = 1` 给出
`−0.511`；`α = 0.5` 给出 **`−0.011 [−0.146, +0.133]`**（test，base 自举 `[复核]`），
区间含 0 且排除前两者的点估计。

**它没有解决什么（必须与结论同时陈述）。**
1. **逐 clip 不变性没有解决，且任何单一 α 都无法解决**：`α` 对每一档只是一个常数因子，
   它只能平移分布、不能压缩分布。实测逐 base `fps2/fps8` 中位 `1.163`、IQR
   `[0.731, 1.520]`、±20% 内仅 6/30，`α=1` 与 `α=0.5` 的**相对 IQR 完全相同**（2.079）
   `[复核]`。
2. **α=0.5 不是独立校准**：它取自同一批 40 个 base 的自举 CI
   （`exponent_source = diffusive_sqrt_lag_default_bootstrap_ci_0p39_0p65_not_independently_calibrated`），
   P1.3 的 holdout 只在**聚合中心**上支持它（见 §4）。
3. **语义改变**：v2 输出连续分数并对阈值域做了统一，因此它**不是**官方布尔判定的等价
   复现，这一改动与 α 本身无关（P1.1 的归因实验正说明这一点）。
4. 阶梯构造的两个缺陷仍在：低档覆盖的轨迹跨度更短（16 帧/8 fps 源在 2 fps 档只覆盖
   80%），10 fps GIF 档容器时长漂移最多 +6.1%。

## 4. Experiment

**S1 反事实 CPA（同 rank 族，纪律：不可单独引用）。** `[复核]` + `[报告]`
`docs/counterfactual-reports/dynamics_degree.md`：

| backend | test CPA（tie-aware，95% CI） | 对 Official 的配对 Δ（95% CI） |
|---|---|---|
| Official | 0.8333 [0.7444, 0.9111] | — |
| v1 归档 `d/dt` | 0.8444 [0.7388, 0.9333] | **+0.0111 [−0.0889, +0.1167]** |
| v2 shipped `d/dt**0.5` | 0.7722 [0.6665, 0.8667] | **−0.0611 [−0.1500, +0.0167]** |

两者配对区间都含 0（parity）。**这张表的作用是反证而不是支持**：满足合约的 v2 反而
比"镜像违约"的 v1 低 0.0722，说明该族的 tie-margin CPA 与合约反相关，因此 §3 的任何
论断都只能由 S2/S3 支撑。（v1 的配对区间**从未被正确计算过**：旧报告把 zero-margin 的
`[0,0]` 印在 tie-aware 的 Δ 旁边。）

**S2 带符号 log-log 指数（主证据）。** `[复核]`，档均值的 log-log 斜率，base 自举 1000 次：

| backend | pooled `p`（95% CI） | test `p`（95% CI） | 逐 clip `p` 均值 ± sd（test） |
|---|---|---|---|
| Official | +0.491 [+0.371, +0.637] | **+0.458 [+0.334, +0.632]** | +0.506 ± 0.372 |
| v1 归档 | −0.481 [−0.598, −0.360] | **−0.511 [−0.646, −0.367]** | −0.501 ± 0.392 |
| v2 shipped | +0.019 [−0.098, +0.140] | **−0.011 [−0.146, +0.133]** | −0.001 ± 0.392 |

支持 §3 的核心论断：Official 显著非 0（`P(p<0)=0.000`）→ 官方确实把采样率编码进分数；
v2 区间含 0（`P(p<0)=0.525`）→ 聚合层不变性达成；v1 是镜像而非修复。同时暴露 §3 的
第 1 条局限：逐 clip 指数 sd ≈ 0.39，与档均值无关地保留了大幅离散。

**S3 level profile（test）。** `[复核]`：
Official `18.42 → 22.04 → 26.47 → 35.20`（比值 1 / 1.196 / 1.437 / 1.911）；
v1 `0.2632 → 0.2375 → 0.1936 → 0.1311`（1 / 0.902 / 0.736 / 0.498）；
v2 `0.0930 → 0.0970 → 0.0968 → 0.0927`（1 / 1.042 / 1.040 / 0.996）。
v2 的 profile 在 ±4.2% 内持平，是"聚合层已修好"的最直观证据。

**S4 独立 holdout（P1.3）。** `[报告 P1.3]`，30 个 prompt 与视频都与反事实集不重叠的 base
（`build_dynamic_validation.py` 断言；候选池经复核为 72 个 base、72 个互不相同的 prompt 与 UID）：

| method | median `fps2/fps8` | IQR | ±20% 内 |
|---|---:|---|---:|
| Official | 1.8474 | [1.253, 3.052] | 0.133 |
| α = 0 | 1.9352 | [1.273, 3.037] | 0.133 |
| **α = 0.5（shipped）** | **0.9676** | [0.637, 1.518] | **0.200** |
| α = 1 | 0.4838 | [0.318, 0.759] | 0.167 |

支持 §3："α=0.5 把**聚合中心**拉回 1"（中位 0.968，自举 CI [0.685, 1.411] `[复核]`），
且明显优于 α=1（配对 `Δ|log ratio|` −0.317 [−0.490, −0.129]）。**反驳 §3 中任何更强的
说法**：三行 α 是同一分布被精确缩放（`ratio(α)=ratio(0)·4^{−α}`，误差 1.1e−15），
**相对 IQR 三行全等 2.385**，±20% 的 4/6/5 只是同一分布的滑窗；即 holdout 只含**一个**
独立测量（raw ratio 分布），不能证明 α=0.5 "已校准"，也不能给出任何离散度主张。

**S5 自然偏好集（P1.1）与 α 归因。** `[报告 P1.1]` + `[复核]`
（`scripts/counterfactual/natural_alpha_attribution.py`，由已存 per-transition 证据重算，
**保真度：1440 个视频 max|shipped−reconstructed| = 0.0**）：
Official tie-aware `0.6845 [0.6597, 0.7093]`，v2 `0.5690 [0.5426, 0.5953]`，
配对 Δ **−0.1155 [−0.1496, −0.0806]**（1290 对）。按 α 重新打分：

| α | 0 | 0.25 | **0.5（shipped）** | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| 配对 Δ vs Official | −0.1178 | −0.1178 | **−0.1155** | −0.0930 | −0.0930 |

**这反驳"不变性修复付出了精度代价"的读法**：缺口在每个 α 上都存在，`α=0`（完全不归一化）
甚至更大；指数对自然集精度的影响 ≤0.025，而缺口是 0.093–0.118。因此 S5 支持的是
"audit, not replacement"——v2 修好了**测量合约**，但作为**人类偏好预测器**它整体劣于
Official 的布尔判据，这个劣势来自该分数族（residual 通道 + 对角归一化 + 连续分 +
dev 校准 margin），不来自 `dt**0.5`。

## 5. Consistency with this dimension's review

`docs/counterfactual-reports/dynamics_degree.review.md` 的核心 verdict 在新数据下的状态：

| review 结论 | 状态 |
|---|---|
| §3 "报告自己的 level table 证明 repair 什么都没修" | **对 v1 仍成立**（`p=−0.511` 是镜像，且它的 `+0.0111` 配对 CI [−0.0889,+0.1167] 本就含 0，从来不是赢）；**对 v2 被推翻**（`−0.011 [−0.146,+0.133]`，聚合层达成不变性） |
| §2 CPAs 在同 rank 族退化、符号不敏感 | **强化**：v2(0.7722) < v1(0.8444) 是同一事实的最终形态 |
| §2.4 "离散度改善与符号翻转不可区分" | **对 v1→v2 被推翻**（CV 0.2360→0.1822 是真实下降），但作为判据仍降级：无符号、且 α 家族无法改变相对 IQR |
| §5 片段不是原因（重编码、双估计器、straightness） | **不变**，本轮未重跑 |
| §8 三处报告缺陷（配对区间张冠李戴、Frame evidence 过期、`mode="measured"` 与 `exponent=0.5` 矛盾） | **仍未修**，写进论文前须先修 `run_dimension.py` 与冻结行的 provenance |

**写作红线。** (i) 不引用本维的 pooled CPA 作为任何胜/负证据；(ii) "v2 satisfies the contract"
必须带 *aggregate*，逐视频不可比；(iii) α=0.5 只能说"由 40 base 自举 CI 支持、并由独立
holdout 在聚合层验证"，不能说"独立校准"；(iv) 自然集的 `−0.1155` 不能读作不变性修复的
代价；(v) 官方 `Δt` 的两段性（`f<12 ⇒ Δt=1/f`）与生成器间差异（0.100 vs 0.125 s）是本维
最硬的代码事实，应作为 Origin 批评的主证据，而非"CogVideo 翻转"这类未验证推断。
