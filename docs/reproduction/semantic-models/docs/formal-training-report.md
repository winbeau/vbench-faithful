# 正式训练报告（8B 四维模型，弱监督基线）

本轮按用户指令"正式开始训练并交付四个维度的模型"执行，全部四条训练在远端 `rtx4090` 上完成。
**数据是弱监督 + 工程 fixture，没有人工金标准**，因此本报告的指标是同分布工程数字，不是泛化或最终 VBench 收益结论。

## 1. 交付物

| 维度 | 产物 | 底座 | 训练步数 |
| --- | --- | --- | --- |
| Spatial | `runs/formal/8b/spatial/`（LoRA adapter） | Qwen3-8B `b968826d…` | 600 |
| Human Action | `runs/formal/8b/action/` | 同上（同一冻结底座） | 300 |
| Multiple Objects | `runs/formal/8b/objects/` | 同上 | 900 |
| Scene | `runs/formal/8b/scene/`（独立模型） | 同上 | 200 |

- 权重：`Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`，
  5 个分片全部通过 `ALL_VERIFIED`（逐分片 SHA256 与 HF LFS 元数据一致）。
- 推理：`scripts/predict.py`，外部指定 `--task` 硬路由到对应 adapter；Scene 用 `--scene-model` 并加 `--scene-stop-at-newline`。
- 曲线：`docs/curves/8b/<task>-{loss,accuracy}.svg`（含 HTML 与 CSV），由 `scripts/plot_training_curves.py` 生成。

## 2. 数据（`data/formal/v1/`，按 `group_id` 家族切分，train/dev 不共享家族）

| 任务 | train | dev | 来源构成 | 长度桶（train） |
| --- | ---: | ---: | --- | --- |
| spatial | 2,330 | 267 | VG 白名单方向边模板（left 376/right 220/above 1000/below 1000）+ fixture + teacher | 1–14: 2,292；15–49: 38 |
| action | 371 | 41 | K400 400 类模板 + fixture + 1 teacher | 全部 1–14 词 |
| objects | 7,368 | 844 | VG 关系端点实体（4,000+200）+ Flickr30k Entities 候选（4,000）+ fixture + teacher | 1–14: 6,169；15–49: 1,198；50–99: 1 |
| scene | 81 | 9 | 30 个手写场景家族（24 家族训练 / 3 家族 dev）+ 5 teacher | 全部 1–14 词 |

许可：VG 为 CC-BY-4.0；**Flickr30k Entities 许可未澄清**（内部研究用，不得再分发）；MovieGen 未用于这四个任务。

## 3. 训练配置与资源

| 任务 | 物理卡 | 步数 | 墙钟 | samples/s | 峰值显存 | train loss | eval loss | token acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial | 4 | 600 | 744 s | 3.22 | ≈19.3 GB* | 0.0000 | 0.0040 | 1.000 |
| action | 6 | 300 | 371 s | 3.23 | ≈20.5 GB* | 0.0002 | 0.0036 | 1.000 |
| objects | 3 | 900 | 1033 s | 3.48 | ≈22.2 GB* | 0.0325 | 0.0329 | 0.992 |
| scene | 1 | 200 | 210 s | 3.81 | ≈25.4 GB* | 0.0000 | 0.0132 | 1.000 |

\* `nvidia-smi` 整卡占用（含其他用户进程），不是本进程净占用；四任务并行在不同卡上，进程内只使用逻辑 `cuda:0`。

共同配置：LoRA r16/α32 all-linear（可训练 43,646,976 参数，底座 8B 冻结）、BF16、梯度检查点、
microbatch 1 × 累积 4、`max_length=2048`、`completion_only_loss=true`、无 packing、seed 20260919。
四个任务全部 `base_frozen=true`、`adapter_changed=true`，监督片段经真实 batch 校验为目标本身、prompt 全 `-100`。

## 4. dev 结果（同分布弱监督，含未微调底座对照）

| 任务 | 样本 | adapter F1 | adapter exact | 底座 F1 | 底座 exact | 解析失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial | 100/267 | 1.000 | 1.000 | 0.990 | 0.990 | 0 |
| action | 41/41 | 0.976 | 0.976 | 0.313 | 0.317 | 0 |
| objects | 200/844 | 0.978 | 0.920 | 0.495 | 0.455 | 0 |
| scene | 9/9 | macro-F1 1.000 | 1.000 | — | — | 0（加/不加停止约束均为 0） |

读法：
- **Action 与 Objects 的收益最大**（F1 +0.66 / +0.48），说明微调把"自由发挥"约束到了目标格式与词表。
- **Spatial 底座本身已 0.99**，模板句任务对 8B 太简单，adapter 提升有限；这不代表自然复杂空间关系的表现。
- **Scene 的 8B 版本不再出现"标签后续写"**（0.6B 版本需要停止约束），加不加约束都 9/9 可解析。

## 5. 自然 prompt 抽样（不在训练集内，人写）

| 输入要点 | 输出 | 判定 |
| --- | --- | --- |
| "small red car … right of a blue van, cyclist waits behind both vehicles" | `car right van`；`cyclist behind both vehicles` | 正确 |
| "children run … kite flying high above them" | `kite above green field` | 关系对象选错（代词 them → 应指 children） |
| "woman plays the piano and sings into a microphone" | `["playing piano","singing"]` | 正确且均为 K400 |
| "chef flips a pancake … waiter pours coffee" | `["flips pancake","pours coffee"]` | **词表外动作**：K400 无此二类，需词表校验/拒收 |
| "cat … windowsill next to a potted plant and a stack of books" | 4 个实体 | 正确（带冠词，需归一化） |
| "wooden bench and a tall oak tree … gravel path" | 2 个实体 | 正确 |
| scene：prompt 海洋帆船 / caption 森林河流独木舟 | `contradicted` | 正确 |
| scene：prompt 现代办公室 / caption 开放式办公室 | `supported` | 正确 |

## 6. 关键发现

1. **弱监督足以教会格式与词表，但不足以证明语义泛化**：dev 与训练同源，F1 高是预期结果。
2. **Action 必须加 K400 校验**：自然 prompt 上仍会输出 `flips pancake` 这类词表外短语，下游需按协议拒收，否则会把不存在的动作送进评分。
3. **Objects 输出需归一化**：模型倾向保留冠词与修饰语（`a cat`、`A young woman`），下游需做冠词剥离与词表映射。
4. **Spatial 的瓶颈在自然语言歧义**（代词、复合关系），不在格式；模板监督无法覆盖这些情况。
5. **Scene 在 8B 上格式自然收敛**，但训练数据仍是手写工程 caption，真实 Tag2Text 场景数据仍缺。

## 7. 未完成 / 未验证

- 无人工金标准、无第二标注者；未做 VBench 端到端接入与最终评分收益验证。
- 201+ 词长度桶为空；未做长度外推测试。
- Action 的自然动作覆盖、Objects 的摄影词汇排除表、空结果/未知类别协议仍未冻结。
- Flickr 许可未澄清，Objects adapter 不得随数据一起再分发。
- 四任务各自独立训练，未做多任务联合或 adapter 合并实验。

## 8. 复现命令

```bash
# 数据
uv run --no-sync python scripts/prepare_data.py --source all --build-id formal-0001 --limit 4000
uv run --no-sync python scripts/build_formal_sets.py --name v1 --build-id formal-0001

# 训练（远端，四卡并行；weight 校验见 docs/decisions.md）
bash scripts/run_formal_training.sh 8b 4 6 3 1

# 曲线
.venv/bin/python scripts/plot_training_curves.py --run runs/formal/8b/spatial \
    --run runs/formal/8b/action --run runs/formal/8b/objects --run runs/formal/8b/scene \
    --output-dir runs/curves/8b --summary runs/curves/8b/summary.json

# 评测（dev，含未微调底座对照）
.venv/bin/python scripts/evaluate_smoke.py --task objects --data data/formal/v1/objects/dev.jsonl \
    --base-model /data1/wenbiao_zhao/models/Qwen3-8B --adapter objects=runs/formal/8b/objects \
    --output-dir runs/formal/8b/eval-objects --limit 200 --compare-base

# 单条推理
.venv/bin/python scripts/predict.py --base-model /data1/wenbiao_zhao/models/Qwen3-8B \
    --adapter spatial=runs/formal/8b/spatial --task spatial --prompt "..."
```

---

## 9. v4 重训（按 B1–B5 决策补齐数据后）

数据（`data/formal/v4/`，合并全部 LLM gold，objects 无 Flickr）：

| 任务 | train | dev | 构成要点 |
| --- | ---: | ---: | --- |
| spatial | 2,679 | 312 | VG 弱监督 2,311 + MovieGen LLM gold 241 + VG prompt gold 118 |
| action | 740 | 81 | K400 模板 361 + LLM gold 自然 prompt 369（K400∪`other`） |
| objects | 7,688 | 811 | VG 关系端点 3,826 + VG 区域短语 3,588 + LLM gold 264，**无 Flickr** |
| scene | 154 | 16 | **真实 Tag2Text caption**（VBench human-preference 真实帧）+ 手写 fixture |

训练（tmux 会话 `vpc-train`，卡 0/5/7/2，8B 底座，v4 配置）：四任务全部 exit=0。

| 任务 | 步数 | train loss | eval loss | token acc | dev exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| spatial | 600 | 0.0038 | 0.0093 | 0.994 | 1.000 |
| action | 300 | 0.0468 | 0.0702 | 0.980 | 0.875 |
| objects | 900 | 0.0106 | 0.0420 | 0.998 | 0.917 |
| scene | 200 | 0.0006 | 0.1175 | 1.000 | 0.812 |

dev 评测（含未微调底座对照）：

| 任务 | 样本 | adapter F1 | adapter exact | 底座 F1 | 底座 exact | 解析失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial | 100/312 | 0.974 | 0.960 | 0.907 | 0.910 | 0 |
| action | 81/81 | 0.876 | 0.864 | 0.185 | 0.259 | 0 |
| objects | 150/811 | 0.957 | 0.933 | 0.634 | 0.453 | 0 |
| scene | 16/16 | macro-F1 0.810 | 0.812 | — | — | 0 |

**与 v1 的差别**：v1 的 dev 全是同源模板（数字接近饱和）；v4 的 dev 含自然 prompt 与真实 caption，因此指标下降但更真实
（action 0.976→0.876、objects 0.978→0.957、scene 1.000→0.812），底座对照差距仍然显著（action +0.69 F1、objects +0.32 F1）。
scene 首次在**真实 Tag2Text caption** 上评测（16 条，3 类，macro-F1 0.81）。

曲线：`docs/curves/v4-8b/<task>-{loss,accuracy}.svg`；指标：`docs/eval/v4-8b/<task>-metrics.json`。

仍未完成：100–200 与 201–400 词桶的家族数尚未到每任务 ≥100（当前 5–9 条），两个标注任务仍在跑；
test-only 家族的真实场景评测集尚未标注（当前 scene dev 为 dev-only 家族 + 工程 fixture）。
