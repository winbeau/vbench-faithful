# S1.5：teacher 数据构造子计划（试标）

状态：pilot-01 已执行（19 次请求，预算 20/20 用满），结果见[试标报告](../teacher-pilot-report.md)。本子计划细化 S1（[数据加工](06-smoke-data.md)）与 P1（[数据与标注](02-data.md)）之间的缺口：现有原料没有足够的自然 prompt→四维目标标签，需要受控调用外部 teacher 生成**候选**标签。

teacher 只产生候选，不是金标准，也不是自动真值。工程 fixture、teacher 候选、人工金标准三者物理隔离并分别计数。

## 1. 授权与预算（硬约束）

| 项 | 值 |
| --- | --- |
| 端点 | `https://api.deepseek.com`（`POST /chat/completions`） |
| 模型 | `deepseek-flash`（官方映射 DeepSeek-V4.1-Flash） |
| 已授权总请求 | 20 次（含已用 1 次连通性检查） |
| 本轮剩余 | 最多 19 次 HTTP POST |
| 单请求输出上限 | `max_tokens <= 2048` |
| 超出预算 | 停止并先向用户确认，不自动扩额 |

- 每一次 HTTP POST 都计入预算，包括超时、限流、重试和响应不可解析的请求。不自动重试超过 1 次。
- 密钥只从 `token.txt` 或环境变量读入，进程内使用；不打印、不写日志、不入 Git、不写进报告。请求体与响应体只记录去掉 `Authorization` 后的哈希与结构化元数据。
- 预算账本 `output/teacher/BUDGET.json` 为累计计数，只增不减；运行前后都要读。

## 2. 试标目标（不是正式造数据）

用 19 次请求同时验证四件事：

1. teacher 在该端点上能否稳定输出符合最小 schema 的 JSON（三解析任务）与三标签（Scene）。
2. 长度分桶（1–14、15–49、50–99、100–200 词）下输出是否仍然合法、是否出现截断或漏字段。
3. 边界输入（无显式关系、词表外动作、无场景要求、多实例歧义）会得到什么输出，用于冻结"空结果/不支持"协议。
4. 产出少量带 provenance 的 `teacher_candidate_unreviewed` 记录，供 S2 训练编码与 S3 冒烟使用（只做工程用途，不作泛化证据）。

不承诺：候选标签正确率、teacher 与人工一致率、覆盖四维全部语义、替代真实 Tag2Text 场景数据。

## 3. 请求矩阵（19 次）

| 组 | 次数 | 任务 | 输入 | 长度桶 |
| --- | ---: | --- | --- | --- |
| 解析-短 | 3 | spatial / action / objects（各 1 次） | 仅 prompt | 1–14 词 |
| 解析-常规 | 3 | 同上 | 仅 prompt | 15–49 词 |
| 解析-中长 | 3 | 同上 | 仅 prompt | 50–99 词 |
| 解析-长 | 3 | 同上 | 仅 prompt | 100–200 词 |
| Scene | 4 | scene | prompt + caption | 与最接近的场景 fixture 家族同桶 |
| 边界 | 3 | spatial / action / scene | 仅 prompt 或 prompt+caption | 不限 |

- 三解析任务**不合并成一次请求**：合并会让单任务缺失无法归因，也不匹配"外部硬路由、每 adapter 一个输入分布"。
- 每次请求一个 prompt，一次只出一个任务的 target；解析任务要求只输出该任务的 JSON。
- prompt 来源：MovieGen 去重后的自然 prompt 池（长短桶）与 K400/Flickr 原料；Scene 使用工程场景 fixture 的 prompt+caption（自写，非真实 Tag2Text 输出）。
- 具体条目由 `configs/teacher/pilot-01.json` 显式列出，只记录 `source`、`source_file`、`line`/`id`、`sha256(prompt)` 与选择理由，**不把上游 prompt 原文写进 Git**（MovieGen 为 CC-BY-NC 4.0，Flickr 许可未定）。运行脚本从 `data/raw/` 解析原文并校验哈希。
- 选择规则确定且可复现：固定 seed、按来源分组、每组内按 (word_count, sha256) 排序取前 N，禁止手工挑选"好样本"后不记录。

## 4. 请求与输出契约

- `thinking: {"type":"disabled"}`；`response_format: {"type":"json_object"}`；`temperature=0`；`max_tokens<=2048`。
- 系统提示固定：只做语义接口，不解释、不补充视觉推测、不输出 JSON 以外的内容。
- 三解析任务的输出必须只有目标字段：

```json
{"relationships": [{"subject": "cat", "relation": "left", "object": "dog"}]}
{"actions": ["playing guitar"]}
{"entities": ["cat", "dog"]}
```

- Scene 输出单标签裸字符串（`supported` / `contradicted` / `insufficient`）。若使用 JSON 包装以便解析，只接受 `{"label": "..."}`，落盘 target 仍为裸标签。
- 提示词明确要求：无法从输入确定时输出空数组（解析任务）或 `insufficient`（Scene），不得凭常识补造；不得输出解释性文字。
- 完整模板、期望 schema 与拒绝规则写在代码 `teacher.py` 中并逐条单元测试。

## 5. 校验门禁（teacher 输出先过校验才入库）

| 检查 | 规则 | 失败处理 |
| --- | --- | --- |
| JSON | 可解析且为对象 | quarantine：`invalid_json` |
| 字段 | 只允许该任务的目标键；缺键或多键拒绝 | `schema_keys` |
| 类型 | 数组元素为字符串；三元组为恰好三键的对象 | `schema_types` |
| 词表 | action 必须在锁定 K400（含别名表，别名需人工确认）；spatial relation 必须在方向白名单 `left/right/above/below`，其他关系词只进 `weak` 队列 | `vocab` |
| 非空与去重 | 空字符串、自关系（subject==object）、重复项去除并计数 | `empty_or_self` |
| 输入支持 | entity/action 短语必须在 prompt 中有依据（原文子串或已登记别名）；否则标 `unsupported_span` | `unsupported_span` |
| Scene | 标签必须三选一；caption 与 prompt 都非空 | `scene_label` |
| 长度 | 记录 word_count 与 token 数；超 token 预算拒绝 | `overflow` |
| 截断 | `finish_reason != "stop"` 视为不完整，不得当成功 | `truncated` |

- 校验通过 → `data/processed/<build-id>/teacher/candidates.jsonl`，`quality="teacher_candidate_unreviewed"`。
- 校验失败 → 同 build 的 `quarantine.jsonl`，保留原始响应文本（ignored 目录）与原因；报告只出计数。
- 计数恒等式：`requests = candidates + quarantined`，逐任务闭合。

## 6. 人工审核与升级路径

- teacher 候选永不直接进正式 train split；先 `candidate`，人工审核后升 `reviewed`，双人仲裁后才可能为 `gold`。
- 每任务先审核约 50 条（本轮若候选不足则全审），记录一致率与分歧类型；不一致样本写回协议文档。
- 冲突不静默丢弃：`disputed` 队列单独计数并说明。
- Scene 候选只能用工程 fixture 评估；明确标记 `research_data_missing`，不用 teacher 冒充真实 pipeline 标签。

## 7. 产物与路径

| 路径 | 内容 | Git |
| --- | --- | --- |
| `output/teacher/BUDGET.json` | 累计请求/输出 token 账本 | ignored |
| `output/teacher/<run-id>/plan.json` | 本次解析后的 19 条请求计划（含哈希，不含密钥） | ignored |
| `output/teacher/<run-id>/ledger.jsonl` | 每次 POST 追加：时间、task、bucket、模型、tokens、finish_reason、原始响应 | ignored |
| `output/teacher/<run-id>/summary.json` | 去密钥的结构化汇总 | 摘要进 docs |
| `data/processed/<build-id>/teacher/` | candidates / quarantine / counts / manifest | ignored |
| `configs/teacher/pilot-01.json` | 计划（locator + 哈希 + 理由） | 入库 |
| `docs/teacher-pilot-report.md` | 试标报告：预算、合法率、失败原因、长度分布、结论与未解决项 | 入库 |

## 8. 失败与停止条件

- 预算耗尽、非法 JSON 率 > 1/3、出现系统性截断、或 K400/关系白名单反复失败 → 停止该轮，写报告，不自动补测。
- 网络/HTTP 错误：记录状态码（不回显响应体），最多重试 1 次，计入预算。
- 不使用第二个付费 API，不切换模型，不在未授权时提高 `max_tokens`。

## 9. 与长度泛化（S4）的衔接

- 试标只需覆盖 1–200 词四个桶；201 词以上、401+ 压力桶留给后续正式造数据。
- 每条候选记录 word_count、字符数与固定 tokenizer 的 token 数（tokenizer 在 S2 冻结后回填），用于验证"完整序列 token 预算"而非只统计 prompt。
- 训练用候选禁止用截断 prompt 制造短样本；长短一致性样本必须同 group。

## 10. 验收

- [ ] 计划先于执行落盘，索引更新。
- [ ] 预算账本与实际 POST 次数一致，未超授权。
- [ ] 每次请求有 task/bucket/模型/tokens/finish_reason/provenance。
- [ ] 三解析任务 JSON 合法率与 Scene 标签合法率分别报告，失败原因分类计数。
- [ ] 候选与 quarantine 计数闭合；teacher 未被写成金标准。
- [x] 计划先于执行落盘，索引更新。
- [x] 预算账本与实际 POST 次数一致（charge=19、result=19、无重试，累计 20/20）。
- [x] 每次请求有 task/bucket/模型/tokens/finish_reason/provenance。
- [x] 三解析任务与 Scene 合法率分别报告；失败原因分类计数（4 条全部为 `action_not_in_k400`）。
- [x] 候选与 quarantine 计数闭合（15+4=19）；teacher 未被写成金标准。
- [x] 报告明确：这 19 条不能证明泛化，不能替代真实 Tag2Text 场景数据，也不能替代人工审核。
- [ ] 15 条候选的人工审核（待办，需新授权或人工标注轮次）。
