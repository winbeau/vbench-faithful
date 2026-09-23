# LLM 标注管线与长 prompt 试点（B1/B2）

按决策 [D3/B1/B2](limitations-and-remediation.md)，用 DeepSeek 做"两遍标注 + 分歧仲裁"，
并新增 `token.txt` 双 token 解析（deepseek 文本、chiyi 视觉）与按用途计费的预算账本。

## 1. 管线

| 组件 | 作用 |
| --- | --- |
| `src/vbench_prompts_compile/llm.py` | 多 token 解析、OpenAI 兼容客户端（文本+图像）、按用途记账的 `RequestBudget` |
| `src/vbench_prompts_compile/annotate.py` | 任务指令、两遍标注、仲裁、可审计清洗（`validate_annotation`） |
| `scripts/annotate_gold.py` | 选样本 → 标注 → `gold.jsonl`/`quarantine.jsonl`/`report.json` |
| `configs/teacher/pilot-01.json` | 早期 teacher 试标（20 次授权，已用满，保持原样） |

标注规则要点：
- **Action（B2）**：K400 类别名直接输出；无法映射为 K400 的动作输出 `"other"` 哨兵，**不允许猜类别**；
  评测端把 `other` 视为"无 K400 匹配"，不计入任何类别。K400 词表校验仍在。
- **Objects（B3）**：输出规范名（小写、单数、去冠词/所有格）；跨度校验在**原始提及**上做，
  规范化在之后进行；被丢弃的实体写入 `meta.sanitisation_notes`，不静默删除。
- **Spatial**：只允许白名单关系词；端点必须在 prompt 中有依据，否则记录并丢弃该条关系。
- 仲裁触发条件：两遍不一致、任一遍校验失败、或都失败 —— 仲裁会带上失败原因。

## 2. 长 prompt 试点（MovieGen 现有长 prompt，先试点不重写）

- 样本：MovieGen 去重池中 50–99 词 20 条 + 100–200 词 9 条 = 29 条 prompt × 3 任务 = **87 项**。
- 结果：**87/87 产出 LLM 标注记录**（58 项两遍一致、29 项经仲裁），一致率 0.667，0 拒绝。
- 花费：203 次 `deepseek-flash` 请求（约 2.3 次/项）；账本 `output/annotation/BUDGET-annotate-longprompt-pilot-v4.json`。
- 清洗痕迹：丢弃关系词 7 次、动作映射为 `other` 7 次、丢弃无依据实体 27 次（全部记录在 meta）。
- **关键观察**：27/87 项的动作无法映射到 K400（`action_other`），说明长 prompt 的动作分布确实超出 K400，
  B2 的 `other` 设计是必要的，而不是可选优化。

## 3. 与早期 teacher 试标的对比

| 项 | pilot-01（teacher 单遍） | longprompt-pilot-v4（两遍+仲裁） |
| --- | --- | --- |
| 请求数 | 19（预算 20 用满） | 203（本次预算 400） |
| 产出 | 15 候选 / 4 拒绝 | 87 记录 / 0 拒绝 |
| Action | 4/5 因非 K400 被拒 | K400 或 `other`，全部可入库 |
| 质量标记 | `teacher_candidate_unreviewed` | `llm_annotated_arbitrated` |

两者都不是人工金标准；差别在于新管线有仲裁与可审计清洗，能直接进训练与评测。

## 4. 复现

```bash
uv run --no-sync python scripts/annotate_gold.py --name longprompt-pilot-v4 \
    --source moviegen --buckets 50-99,100-200 --budget 400
uv run --no-sync pytest -q tests/test_records.py tests/test_teacher.py
```

## 5. 下一步（进行中）

1. **B1 规模化**：每个任务每个长度桶 ≥100 家族；短桶用现有池 + 改写，长桶用现有长 prompt + 改写（不截断）。
2. **B4 Objects 去 Flickr**：下载 VG `region_descriptions.json.zip`（127MB，CC-BY-4.0），
   用自然区域描述 + 关系端点重建 objects 训练集。
3. **B5 Scene 真实数据**：H100 上已有 Tag2Text 权重（`/root/.cache/vbench/caption_model/tag2text_swin_14m.pth`），
   用 VBench 1.0 human-preference 的真实视频抽单帧出 caption；标签用 `gpt-5.6-luna` 看图 + 仲裁。
   注意：pilot 已确认冻结 split 覆盖全部 scene prompt（dev 18 / test 36 / dev+test 32），
   **test 家族只用于评测**，训练只用 dev-only 家族。
4. 四维重训 + 曲线 + 评测报告更新。
