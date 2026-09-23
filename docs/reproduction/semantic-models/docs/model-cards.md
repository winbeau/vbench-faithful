# 四维模型卡（当前交付状态）

后续新增[同数据 0.6B/8B 消融与架构报告](deterministic/ablation-v1/README.md)。四个新 0.6B adapter 在远端隔离目录 `vbench-ablation-v1/runs/ablation-v1/0.6b/`，与当前 8B 同数据、同固定步数；不能与下文历史冒烟产物混用。Action 默认接口已更新为带范围保护的 `repair-v2.1`，`repair-v2` 保留冻结旧行为。新同义表达、Scene 长文本与视觉正例保留的限制见新增报告。

**状态（2026-09-19）**：四个 8B 模型已训练完成并可用，训练/评测数字见[正式训练报告](formal-training-report.md)；
曲线在 `docs/curves/8b/`。下面每张卡片给出可用范围与不能声称的内容。

四个可加载模型：三个解析 LoRA（共享同一冻结底座、按任务硬路由）+ 一个独立 Scene 模型。
每个模型都附数据来源、训练配置、可用范围与明确不能声称的东西。

## 0. 底座

| 用途 | 底座 | revision | 说明 |
| --- | --- | --- | --- |
| 正式（8B） | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | Apache-2.0，5 分片 16.4GB，逐分片 SHA256 固定 |
| 快速回归（0.6B） | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 用于冒烟与对照，不作为正式结论来源 |

推理时不合并 adapter；三个 adapter 同时挂在同一底座上，由外部任务名硬路由（`scripts/predict.py --task`）。

## 1. Spatial adapter

| 项 | 内容 |
| --- | --- |
| 输入 / 输出 | prompt → `{"relationships":[{"subject","relation","object"}]}` |
| 训练数据 | Visual Genome 白名单方向边（left 376 / right 220 / above 1000 / below 1000 采样）生成的模板句 + 12 条手写 fixture + 5 条 teacher 候选 |
| 训练产物 | `runs/formal/8b/spatial/`（600 步；dev F1 1.000，底座 0.990） |
| 质量等级 | `synthetic_weak`（模板句）+ `engineering_only` + `teacher_candidate_unreviewed` |
| 长度覆盖 | 以 1–14 词为主（2,292/2,330），15–49 词 38 条；**无 50 词以上训练样本** |
| 明确不能声称 | 不是人写自然 prompt 的监督；左右关系样本稀缺；`on/under` 等弱关系不保证满足几何判据；不能声称已解决遮挡与多实例身份 |

## 2. Action adapter

| 项 | 内容 |
| --- | --- |
| 输入 / 输出 | prompt → `{"actions":[...]}`，名称对齐锁定 K400 |
| 训练数据 | 400 个 K400 类别各一条模板句（动名词用 "A person is {label}."，其余 "The video shows {label}."，70 条标注 `grammar_review=required`）+ 12 条 fixture + 1 条 teacher 候选 |
| 训练产物 | `runs/formal/8b/action/`（300 步；dev F1 0.976，底座 0.313） |
| 长度覆盖 | 全部 1–14 词 |
| 已知失败模式 | 推理端未做 K400 约束解码，模型可能输出词表外动作（实测出现 `sleeping cat`）；下游必须用 K400 词表校验，未知动作按协议拒收 |
| 明确不能声称 | 不代表自然动作表达覆盖；不代表 K400 全类都能被正确识别；teacher 不能直接产出 K400 标签（试标 4/5 被拒） |

## 3. Objects adapter

| 项 | 内容 |
| --- | --- |
| 输入 / 输出 | prompt → `{"entities":[...]}`，不限于两个对象 |
| 训练数据 | Visual Genome 关系端点派生实体对（4,000 + 200 多实体）+ Flickr30k Entities 候选（4,000）+ 12 条 fixture + 4 条 teacher 候选 |
| 训练产物 | `runs/formal/8b/objects/`（900 步；dev F1 0.978，底座 0.495） |
| 许可 | VG 为 CC-BY-4.0；**Flickr30k Entities 许可未澄清，当前只作内部研究用，不得再分发**；MovieGen 未用于本任务 |
| 长度覆盖 | 1–14 词 6,169、15–49 词 1,198、50–99 词 1 |
| 已知失败模式 | 输出可能带冠词或修饰语（`a cat` / `A young woman`），下游需做词表归一；摄影/后期词汇（camera 等）尚无排除表 |
| 明确不能声称 | Flickr 部分不得作为可发布数据；不统计实例数量；不代表 GRiT 标签空间已对齐 |

## 4. Scene 模型（独立）

| 项 | 内容 |
| --- | --- |
| 输入 / 输出 | prompt + caption → `supported` / `contradicted` / `insufficient` |
| 训练数据 | 30 个手写场景家族 × 3 标签（90 条，engineering_only）+ 5 条 teacher 候选 |
| 训练产物 | `runs/formal/8b/scene/`（200 步；dev 9/9 exact，严格与停止约束下都可解析） |
| 决策（D2） | 保持生成式三标签 SFT，推理端固定标签边界约束（`--scene-stop-at-newline`） |
| 已知失败模式 | 不加约束时会先给正确标签再续写解释，严格解析判失败；600 步训练仍如此 |
| 明确不能声称 | **没有真实 Tag2Text caption 与人工金标准**，因此不能声称真实管线准确率；工程 caption 上的 100% 只说明流程可用 |

## 5. 使用方式

```bash
uv run --no-sync --extra train python scripts/predict.py \
  --base-model /data1/wenbiao_zhao/models/Qwen3-8B \
  --adapter spatial=runs/formal/8b/spatial \
  --adapter action=runs/formal/8b/action \
  --adapter objects=runs/formal/8b/objects \
  --scene-model runs/formal/8b/scene \
  --task spatial --prompt "A cat sleeps to the left of a dog."
```

下游必须做的事：Action 用 K400 词表校验并拒收词表外动作；Objects 做冠词/别名归一化；
Spatial 的弱关系（`on/under/beside` 等）不保证满足几何判据，需按协议处理。

## 6. 共同限制

> 每条限制的弥补路径、验收标准与执行顺序见[边界与弥补方案](limitations-and-remediation.md)。

- 训练/验证都来自同一批弱监督模板，dev 指标是**同分布工程数字**，不是泛化证据。
- 201+ 词长度桶为空；401+ 压力测试未做。
- 未做双人标注仲裁、未做多卡吞吐与更长的显存测量。
- 训练产物为 LoRA adapter（约几十 MB），底座权重不随仓库分发。
