# Objects Repair v2：相邻帧确认及其代价

本轮实现了 **同名检测框的相邻帧确认**，抑制孤立检测产生的正分。在 275 个模型复核为目标全帧不可见的视频中，仍有正分的视频从 **29 降至 15**，减少 14 个（48.3%）。原始视频分数及可见帧检出也有损失；持续误报仍然存在。该结果支持保守过滤孤立检测，不支持“视觉识别已全面修好”。

Origin 保留原有单帧标签合取。Repair-rule 与 Repair-model 都使用同一新后端，因此二者本轮完全相同；模型仍为固定 Objects v6 final，未重训。已有检测缓存、输入目标和可见性标签未改。本轮为看过错误后的工程修复，不是独立盲测；固定方案见[计划](../../plans/15-objects-temporal-repair.md)。

## 方法与接口

生产实现为 `src/vbench_prompts_compile/objects_repair.py`，正式评分默认 `--objects-backend repair-v2`；`legacy-official` 可完整重现上一版 Objects。

每个当前采样帧必须同时检测到所有要求的实体。每个实体还需至少一个当前框在前一帧或后一帧找到同名框，IoU≥0.5。边界只检查存在的相邻帧；不跨缺帧，不把末帧当成首帧邻居，不用相邻检测补出当前帧不存在的物体。阈值在新方法复算前固定，没有搜索测试最优值。16 帧分母和每帧硬合取保留。

评分函数只接收实体和该视频的检测帧，不接收遮挡等级、遮挡框、背景/目标变换名、原视频配对分数、可见性标签或参考答案。原始视频和所有变换使用同一实现。

每帧单列四种状态：`supported`（合取且获确认，记 1）、`absent`（原检测未满足合取，记 0）、`unconfirmed`（检测存在但确认不足，记 0 并计弃权）、`missing`（坏框/缺帧，记 0 并计缺失）。`absent` 是检测判据，不是人工真值。**过滤为 unconfirmed 不等于确认物体不存在。**

## 不可见端点与消融

完整保留 980 个视频 / 49 家族、12,740 个 Objects 条件。980 个重度目标遮挡端点中，505 个完成视觉模型复核，475 个几何不完整；275 个目标全帧不可见来自 48 家族，均为同一视觉模型多遍共识，`human_reviewed=false`。

| 方法 | 275 视频中仍有正分 | 没有正分 | 全部明确阴性 | 不可见端点正分帧 |
| --- | ---: | ---: | ---: | ---: |
| Origin / 旧 Repair | 29 | 246 | 246 | 158/4,400（3.5909%） |
| 仅相邻同名标签 | 15 | 260 | 246 | 124/4,400（2.8182%） |
| 相邻同名框，IoU≥0.5 | 15 | 260 | 246 | 118/4,400（2.6818%） |

新后端的端点正分率降低 **5.0909 个百分点**，配对家族 bootstrap 95% CI 为 **[−8.3916, −2.2436]** 个百分点。正分帧率降低 0.9091 个百分点，CI=[−1.5513, −0.4337]。多出的 14 个零分端点含未经确认的检测，因此“全部明确阴性”仍为 246，不能改写成 260 次已证实正确拒绝。

全 980 计划分母上的“模型复核不可见且没有正分”下界为 260/980=0.26531；要求全部明确阴性时仍为 246/980=0.25102。475 个几何不完整端点没有被删除或当作成功。完整 CI、弃权及固定隔离子集见 [endpoints.md](endpoints.md)。

仅标签与框确认在视频层面都减少 14 个残留；框确认额外过滤 6 个不可见正分帧，同时也增加可见目标损失。没有把这个消融描述成无代价收益。

## 原始与可见目标的代价

| 指标 | Origin / 旧 Repair | Repair v2 |
| --- | ---: | ---: |
| 全部 980 原始视频均分 | 0.28431 | 0.26173 |
| 原始正分帧 | 4,458 | 4,104（保留 92.06%） |
| 原始正分视频 | 442 | 362（保留 81.90%） |
| 原始模型复核双方可见帧检出 | 3,120/4,887（63.84%） | 3,023/4,887（61.86%） |
| 遮挡后模型复核双方仍可见帧检出 | 38/1,024（3.71%） | 19/1,024（1.86%） |

全量原始均分差为 **−0.02258**，家族 CI=[−0.02659, −0.01907]，在既有 0.03 均分容差内；这不代表每个视频无损，也不代表真实召回不变。原检测正分保留率是与旧检测器的一致性，不是人工召回率。可见性分组明确显示原始双方可见帧损失 97 帧，遮挡后仍可见帧损失 19 帧。

在相同 115 个“旧分数为正且模型复核严格隔离移除”的视频中，达到 0.03 降幅由 **114/115 降为 106/115**；若还要求两端没有任何弃权，新后端为 **87/115**。不能因新后端压低原始分数或更换正基例分母而掩盖此退化。具体见 [original-impact.md](original-impact.md)、[visibility.md](visibility.md) 和 [endpoints.md](endpoints.md)。

## 遮挡对照

Origin 全计划重度目标遮挡均分为 0.28431→0.01754；Repair 为 **0.26173→0.01040**。两者变换帧覆盖率均为 0.64872，几何失败保持原样，不能把缺失帧的零分当作有效视觉拒绝。

同一批两端完整的等面积目标/背景对照中，重度等级只有 **25/980 视频、16 家族**：Origin 目标/背景为 0.0775/0.4825，Repair 为 **0.0500/0.4725**，两者差分别为 −0.4050 与 −0.4225。不能外推为全部 980 个视频的因果结论。全部三级及 CI 见 [controls.md](controls.md)；全部 39 方案/条件汇总见 [scores.md](scores.md)。遮挡外像素、等面积背景及零遮挡哈希校验均通过。

## 验证与边界

- 完整四维 **88,980 条**复算；只有 Objects 两个 Repair 的 25,480 条进入新后端。其他 **63,500 条**与上一版逐字段相同，包括 Objects Origin、Spatial v2、Action v2 和 Scene。
- `legacy-official` 完整重现此前 **38,220 条 Objects 记录**；7,840 条 Repair 恒等/零遮挡记录的分数与确认状态精确不变。当前正分都是原始合取正分的子集。
- **219 项测试通过**，包括孤立/连续检测、框位置不符、精确阈值、多实例、多实体、未知实体、缺帧/坏框、单帧、时间反转、统一平移缩放、旧版回放，以及不能用变换元数据或可见性标签改变评分。已有 Trainer 文件 mtime 警告保留。
- 模型权重、原始 JSON、数据划分和官方协议未改；本轮仅 CPU 缓存重放，没有新训练、视觉推理、付费 API 或权重下载。H100 GPU 4 预检时已有高负载任务，未启动新 GPU 作业。
- 相邻帧并非独立检测器，持续错误可能连续得到确认。快速运动、短暂可见、小目标和形变可能无法满足 IoU；重复采样帧不能提供独立支持。剩余 15 个正分端点和所有弃权明细保留在忽略的 `output/deterministic/objects-repair-v2/residual-endpoints.jsonl`。

[report.json](report.json) 保存完整汇总、家族 CI 和输入/源码/产物 SHA；[acceptance.json](acceptance.json) 保存验收。原始预测、图片和逐帧数据不提交 Git。本轮工作区修改未提交或推送。

## 复现

```bash
uv run --no-sync python scripts/score_matrix.py \
  --matrix data/deterministic/matrix-v1/matrix.jsonl \
  --metadata output/upstream-parity/VBench_full_info.json \
  --cache spatial=data/backend-cache/matrix-v1/spatial-test.jsonl \
  --cache objects=data/backend-cache/matrix-v1/objects-test.jsonl \
  --cache scene=data/backend-cache/matrix-v1/scene-test.jsonl \
  --cache action=data/backend-cache/matrix-v1/action-test.jsonl \
  --transforms data/backend-cache/matrix-v1/spatial-transforms.jsonl \
  --transforms data/backend-cache/matrix-v1/objects-transforms.jsonl \
  --predictions data/repair/matrix-v1/spatial.jsonl \
  --predictions data/repair/matrix-v1/objects.jsonl \
  --predictions data/repair/matrix-v1/scene.jsonl \
  --predictions data/repair/matrix-v1/action-v9.jsonl \
  --visibility data/gold/objects-visibility-v1/merged.jsonl \
  --objects-backend repair-v2 \
  --out output/deterministic/matrix-objects-v2

uv run --no-sync python scripts/evaluate_objects_repair.py \
  --matrix data/deterministic/matrix-v1/matrix.jsonl \
  --cache data/backend-cache/matrix-v1/objects-test.jsonl \
  --transforms data/backend-cache/matrix-v1/objects-transforms.jsonl \
  --visibility data/gold/objects-visibility-v1/merged.jsonl \
  --predictions data/repair/matrix-v1/objects.jsonl \
  --metadata output/upstream-parity/VBench_full_info.json \
  --previous-paired output/deterministic/matrix-action-v2/paired-rows.jsonl \
  --paired output/deterministic/matrix-objects-v2/paired-rows.jsonl \
  --score-report output/deterministic/matrix-objects-v2/report.json \
  --out output/deterministic/objects-repair-v2

uv run --no-sync python scripts/analyze_object_controls.py \
  --paired output/deterministic/matrix-objects-v2/paired-rows.jsonl \
  --cache data/backend-cache/matrix-v1/objects-test.jsonl \
  --transforms data/backend-cache/matrix-v1/objects-transforms.jsonl \
  --out output/deterministic/matrix-objects-v2/object-controls.json
```

完整四维新版表格在 `output/deterministic/matrix-objects-v2/tables/`；Spatial/Action v2 及 matrix-v1 历史产物保留。复现首轮 matrix-v1 需同时指定 `--spatial-backend legacy-official --action-interface legacy-model --objects-backend legacy-official`；只复现上一版 Action v2 则仅切换 Objects 后端。
