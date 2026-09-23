# Action 同义接口修复 v2

已实现并完成全部缓存复算：**同义表达下 Origin 0.88917→0，Repair 0.88917→0.88917**。60/60 对动作表达的目标类别正确且一致，1,200/1,200 视频的分数逐项相同。这里的 Repair-model 是固定 **Action v9 final＋确定性归一化接口**；没有重训，也不把已声明词典覆盖域内的结果当成未见同义泛化。

## 原因与实际修改

旧链路 35 个同义输入成为 other，其中 **31 个原始模型输出为非 other 动作短语**，被旧词表归一化拒绝；另 4 个模型本身输出 other。前者也包含省略上下文的短语，不能简单说 31 个全是完整、无歧义的模型正确答案。只换输出词典仍有剩余错误。

现有 60 对同义契约从实验脚本移至生产模块，已与修复前提交 `ea9d500f21d800507ef220b6e95fee672d0e70f1` 做逐项相等校验；没有按本次失败增加别名。接口只读取当前文本、模型输出与 K400 词表：

1. 保留模型原始 JSON，将已声明同义短语序列化为 K400 名称；未覆盖部分保留旧模型输出解析能力。
2. 当前输入若完整匹配类别原名或已声明同义表达，用该文本契约确定类别，修复模型遗漏/other。逐项记录 `complete_prompt_contract`，不冒充模型原生预测。
3. 完整类别优先于并列拆分，保留已知动作及未知 `other`；歧义、否定、仅观看或计划等语境不因含关键词被强配。非法/缺失模型输出保留失败。

没有读取视频分数、视觉类别、变换类型、原始配对预测或测试参考目标。类别替换由替换后的文本重新解析，不能通过“复制原句分数”满足不变性。

代码：[Action 接口](../../../src/vbench_prompts_compile/action_repair.py)、[冻结同义契约](../../../src/vbench_prompts_compile/action_lexicon.py)、[推理](../../../src/vbench_prompts_compile/inference.py)、[评分](../../../scripts/score_matrix.py)。`predict.py`、`batch_predict.py`、`score_matrix.py` 默认使用 `repair-v2`，可显式选择 `--action-interface legacy-model`。模型能力评测脚本 `evaluate_smoke.py` 仍默认旧模型路径。

## 同义主结果与消融

60 家族，每家族 4 个生成器×5 seeds，共 1,200 视频。同一 UMT 缓存、top-5、置信度四位小数舍入与 0.85 阈值；原始/同义两次评分分别从各自解析的类别计算。

| 方案 | 原始均分 | 同义均分 | 配对 Δ（95% 家族 CI） | 正确且一致的类别对 |
| --- | ---: | ---: | --- | ---: |
| Origin | 0.88917 | 0 | −0.88917 [−0.92833,−0.84333] | 0/60 |
| 旧 Repair-model v9 链路 | 0.86500 | 0.37833 | −0.48667 [−0.59750,−0.37583] | 25/60 |
| v9＋仅输出短语归一化 | 0.86500 | 0.77000 | −0.09500 [−0.16667,−0.03583] | 51/60 |
| **v9＋完整接口 v2** | **0.88917** | **0.88917** | **0 [0,0]** | **60/60** |
| Repair-rule | 0.88917 | 0.88917 | 0 [0,0] | 60/60 |

v2 原始类别匹配也从 58/60 修复到 60/60，故原始均分与旧模型不同。1,200/1,200 视频精确不变，超过原有 |Δ|≤0.03 门槛；不是只看两组均值相等。一直输出 other 的常数策略不能通过“正确且一致”或非弃权不变性要求。旧模型原始/同义目标相同率 45%，正确且一致只有 41.67%，两者分开报告。

完整表：[全部条件](scores.md)、[类别解析](parsing.md)、[评分消融](ablations.md)、[目标一致性](consistency.md)。CSV 保存精度，CI 为 2,000 次来源家族 bootstrap、seed=20260919。

## 控制项与边界

| 控制项 | v2 结果 | 解释 |
| --- | --- | --- |
| 换为另一 K400 类 | 60/60 目标集合正确；0.88917→0.00167 | 条件确实变化；没有独立核验每条新动作都不在画面中，故不称人工确认敏感性 |
| 词表外 assembling furniture | other，得分 0 | 只有 1 个唯一输入，不能用重复的 1,200 视频夸大 OOV 样本数；不报 CI |
| 已知＋未知动作 | 60/60 集合正确；总均分 0.44458 | 已知动作仍保留；other 计零使两个分量均值减半，已知项均分仍 0.88917 |
| K400 全部原名 | 400/400 往返一致 | 原始标签和 ID 不变 |
| 工程反例 | 否定/观看/意图、歧义、混合动作、含 and 类别、非法/缺失输出通过测试 | 这些是工程回归，不能当作独立人工盲测 |

234 个唯一预测中，173 个由完整文本契约覆盖，60 个为已知＋未知的部分覆盖，1 个纯 OOV 保留模型路径。本轮域内大部分确定性由已有词典提供；规则与 v2 系统最终同分，**不能据此声称 LoRA 获得了新的语义泛化能力**。词典外自由表达仍依赖模型，可能失败；本轮没有新增开放同义盲测或多样 OOV 评测。

## 验证与产物

- 全四维 29,660 清单项 / 88,980 评分记录重新计算。Action Repair-model 6,000 条使用新接口；其余 **82,980 条**与 Spatial v2 版本逐字段相同，包括所有 Origin、Spatial v2、Scene、Objects 和 Action 规则结果。
- 234 条原始模型生成字符串保持不变；从 raw 重新后处理得到的目标与评分入口逐条一致，对应 **6,000 个 Action 条件记录**。
- **195 passed**；依赖锁、主 CLI 和 7 个相关脚本 help、格式检查通过。只有已有 Trainer 文件 mtime 警告。
- 0 新训练步、0 新模型生成、0 API 调用；本轮是 CPU 后处理与缓存评分。固定模型身份仍由旧预测 plan 绑定，新后处理 plan 绑定旧 plan 哈希。

[report.json](report.json) 记录旧错误归因、统计、全矩阵回归、词表/代码/输入哈希；[acceptance.json](acceptance.json) 保存验收。原始/修复后逐项预测与 trace 在忽略目录 `output/deterministic/action-repair-v2/`，不提交 Git。四维新表在 [matrix-action-v2](../../../output/deterministic/matrix-action-v2/main-table.md)。首轮 matrix-v1 和 Spatial v2 报告均保留。

## 复现

缓存后处理与正式推理共用同一模块：

```bash
uv run --no-sync python scripts/repair_action_predictions.py \
  --predictions data/repair/matrix-v1/action-v9.jsonl \
  --out output/deterministic/action-repair-v2/predictions.jsonl
```

完整四维评分继续使用冻结矩阵、原始缓存和同一预测文件，设置默认 `--spatial-backend repair-v2 --action-interface repair-v2`，写入新目录 `output/deterministic/matrix-action-v2`。随后核验与生成 Action 表：

```bash
uv run --no-sync python scripts/evaluate_action_repair.py \
  --matrix data/deterministic/matrix-v1/matrix.jsonl \
  --previous-paired output/deterministic/matrix-spatial-v2/paired-rows.jsonl \
  --paired output/deterministic/matrix-action-v2/paired-rows.jsonl \
  --score-report output/deterministic/matrix-action-v2/report.json \
  --predictions data/repair/matrix-v1/action-v9.jsonl \
  --repaired-predictions output/deterministic/action-repair-v2/predictions.jsonl \
  --cache data/backend-cache/matrix-v1/action-test.jsonl \
  --out output/deterministic/action-repair-v2
```
