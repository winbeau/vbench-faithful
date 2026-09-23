# teacher 试标报告（pilot-01）

本轮按 [S1.5 teacher 数据构造子计划](plans/10-teacher-data.md) 执行了授权的 19 次 `deepseek-flash` 请求。**teacher 输出只是候选，不是金标准**；本报告不构成泛化、准确率或收益结论。

## 1. 授权与预算

| 项 | 值 |
| --- | --- |
| 端点 / 模型 | `https://api.deepseek.com` / `deepseek-flash`（返回 `system_fingerprint=aeb56401ca74e127821c4f9126dcb669`） |
| 授权总量 | 20 次（含 1 次连通性检查） |
| 本次请求 | 19 次，全部 HTTP 200、`finish_reason=stop` |
| 重试 | 0 次 |
| 预算账本 | `output/teacher/BUDGET.json` → `used_requests=20`（已用满，后续调用需重新授权） |
| 单请求上限 | `max_tokens=2048`（实际 completion 最大 66 tokens） |
| 输出 token | prompt 3,596 + completion 333 |
| 延迟 | p50 817 ms，均值 981 ms，最大 2,002 ms |

计划先于执行落盘：`output/teacher/20260918T193148Z/plan.json`；每次 POST 先记账后发送（`ledger.jsonl` 中 charge=19、result=19）。

## 2. 逐条结果

| item | 任务 | 词数桶 | 状态 | 结果摘要 |
| --- | --- | --- | --- | --- |
| parse-spatial-b1 | spatial | 1–14 | candidate | `between(astronaut, stone buildings)` |
| parse-action-b1 | action | 1–14 | candidate | `["roller skating"]`（K400 合法） |
| parse-objects-b1 | objects | 1–14 | candidate | `["Blooming Flowers"]` |
| parse-spatial-b2 | spatial | 15–49 | candidate | `behind(cape, person)` |
| parse-action-b2 | action | 15–49 | **quarantine** | `["pushing aside vines"]` 不在 K400 |
| parse-objects-b2 | objects | 15–49 | candidate | person / motorbike / quiet street / engine |
| parse-spatial-b3 | spatial | 50–99 | candidate | `{"relationships":[]}` |
| parse-action-b3 | action | 50–99 | **quarantine** | `sits` / `reads a picture book aloud` / `using expressive voices` |
| parse-objects-b3 | objects | 50–99 | candidate | 6 个实体，含长修饰名词短语 |
| parse-spatial-b4 | spatial | 100–200 | candidate | 2 个关系（`between`, `above`） |
| parse-action-b4 | action | 100–200 | **quarantine** | 5 个自然动作，均非 K400 |
| parse-objects-b4 | objects | 100–200 | candidate | 6 个实体，含 `Camera`（见 §4.4） |
| scene-supported-1/2 | scene | 1–14 | candidate | 与工程 fixture 标签一致（supported） |
| scene-contradicted-1 | scene | 1–14 | candidate | 一致（contradicted） |
| scene-insufficient-1 | scene | 1–14 | candidate | 一致（insufficient） |
| edge-spatial-none | spatial | 1–14 | candidate | 显式输出空数组，符合协议 |
| edge-action-unknown | action | 15–49 | **quarantine** | `using a screwdriver` / `assembling furniture` / `tightening screws` |
| edge-scene-none | scene | 1–14 | candidate | `insufficient` |

汇总：candidate 15 / quarantine 4（全部为 `action_not_in_k400`）；解析任务 9/12 通过 schema+词表+跨度校验，Scene 5/5 通过；解析任务中空间 5/5、Objects 4/4 合法。

## 3. 校验门禁的实际作用

- JSON 合法率 19/19（无 `invalid_json`、无 `schema_keys`/`schema_types` 失败）。
- 跨度校验拦下 0 条（teacher 未在已通过 schema 的样本中凭空造对象），但把 `Camera` 这类**输入支持的伪对象**留给人工审核，说明 span 校验只能防幻觉、不能判语义角色。
- 4 条 quarantine 全部来自 K400 词表门禁：teacher 输出的是自然动作短语，不是锁定类别名。**这是本轮的结论性发现，而不是代码缺陷。**

## 4. 关键结论

1. **Action 不能直接靠 teacher 生成 K400 标签。** 5 条 action 请求中 4 条被词表拒绝（12 个动作短语）。后续必须改为：提示词内嵌/约束 K400 类别 + 确定性别名表 + 人工复核，或只把自然动作作为"未映射候选"，不进入 Action 训练。禁止在推理端加 LLM match 阶段（与既有约束一致）。
2. **词汇线索 ≠ 显式关系。** `parse-spatial-b3`（69 词）包含 `floats above`，但那是"云自己飘在上方"，没有第二个具名实体，teacher 返回空数组。说明自然 prompt 不能仅凭正则提取空间监督；Spatial 的弱监督仍应以 VG 结构化边为主，自然 prompt 需逐条审核。
3. **空结果协议可用。** 无关系 prompt 得到显式 `{"relationships":[]}`，未出现编造关系；但空数组是否入训仍取决于尚未冻结的协议。
4. **需要一份"非场景对象"排除表。** `Camera tracking shot.` 让 Objects 目标出现 `Camera`。摄影/后期词汇（camera、close-up、3D animation、shallow depth of field 等）应作为协议性排除项，而不是靠模型自觉。
5. **Scene 方向初步可行但证据很弱。** teacher 与 4 条工程 fixture 标签及 1 条"无场景要求"探针 5/5 一致；但这些 caption 是我们手写的工程文本，不是真实 Tag2Text 输出，**不能当作真实管线准确率**，`research_data_missing` 仍然成立。
6. **长度桶全覆盖 1–200 词**，无一截断（`finish_reason` 全为 stop）；201+ 词与 401+ 压力桶仍未试标。
7. **成本极低**：19 次请求合计约 3.9k tokens、平均 1 秒延迟。但预算已用满，任何扩标都需要用户重新授权。

## 5. 产物与复现

| 路径 | 内容 |
| --- | --- |
| `configs/teacher/pilot-01.json` | 选拔规则（入库，不含上游 prompt 原文） |
| `output/teacher/20260918T193148Z/plan.json` | 解析后的 19 条请求（含 prompt 文本，ignored） |
| `output/teacher/20260918T193148Z/ledger.jsonl` | charge/result 逐条记录与响应哈希（ignored） |
| `output/teacher/20260918T193148Z/candidates.jsonl` | 15 条 `teacher_candidate_unreviewed`（ignored） |
| `output/teacher/20260918T193148Z/quarantine.jsonl` | 4 条拒绝样本与原因、原始响应（ignored） |
| `data/processed/local-0001/teacher/` | 候选与 quarantine 的 build 内副本（ignored） |

```bash
uv run --no-sync python scripts/run_teacher_pilot.py --config configs/teacher/pilot-01.json --dry-run
uv run --no-sync python scripts/run_teacher_pilot.py --config configs/teacher/pilot-01.json --confirm-budget 19
uv run --no-sync pytest tests/test_teacher.py tests/test_pilot_sources.py
```

## 6. 未解决 / 未验证

- 15 条候选**尚未人工审核**，也没有第二标注者；一致率与分歧类型未统计。
- 未使用真实 Tag2Text caption，Scene 仍缺研究级数据。
- Action 的 K400 约束方案未定；Objects 的摄影词汇排除表未落盘。
- 201+ 词、跨任务一次多标签、非英文 prompt 均未试标。
- MovieGen 为 CC-BY-NC 4.0：用过其 prompt 后不得再把同一批 prompt 当独立 MovieGen 测试集。
- 这些候选不足以训练正式适配器；当前只用于 S2/S3 的工程冒烟。
