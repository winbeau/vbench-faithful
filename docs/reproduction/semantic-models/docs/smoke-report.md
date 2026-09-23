# 冒烟报告：清洗 → 训练编码 → 最小闭环（T0–T3）

本轮按 [S3 分层冒烟子计划](plans/08-smoke-validation.md) 执行。**这是工程验收，不是效果或泛化结论**：数据主体是工程 fixture 与弱监督模板，独立金标准尚不存在。

## 1. 结论摘要

| 检查 | 结果 |
| --- | --- |
| T0 数据/CLI 纯 CPU | 通过：87 项测试、构建可复现、计数闭合、工程与候选数据物理隔离 |
| T1 随机微型模型 CPU 闭环 | 通过：四任务 forward/backward/保存/重载/路由均正确；**无任何语义能力** |
| T2 真实底座单卡 100 步 | 通过：Qwen3-0.6B × 4 任务，远端 4090 与本地 4060 各跑一遍；底座冻结、adapter 有更新、mask 正确 |
| T3 推理与初步验证 | 完成：解析任务 held-out 切分上 F1 0.91–1.00（对照底座 0.00–0.87）；Scene 严格格式失败率高，加"标签后即停"约束后 12/12 通过 |
| 泛化/收益 | **未验证**：无人工金标准、无独立自然测试、长度桶覆盖不全 |

## 2. 环境与代码状态

| 项 | 值 |
| --- | --- |
| 代码 | `21d0f0d5ee47092ce70089b60c1622f109cbf6f1` + 未提交改动（本轮未 commit/push） |
| 依赖锁 | `uv.lock` SHA256 `0ebe4906cb87ce69e4281ee1e54eb684f697e2de48c557bf4e3ab8894fde94ba`（未改） |
| 训练栈 | torch 2.7.1+cu126 / transformers 4.52.4 / trl 0.19.1 / peft 0.15.2 / datasets 3.6.0 |
| 底座 | `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`，`model.safetensors` SHA256 `f47f71177f32bcd1…2996874b`（本地与远端一致） |
| 远端 | `rtx4090`，物理卡 2（CUDA_VISIBLE_DEVICES=2，进程内 `cuda:0`），driver 580.105.08 |
| 本地 | RTX 4060 Laptop 8GB，CUDA_VISIBLE_DEVICES=0 |
| 数据 | `data/smoke/mix-0001/`：spatial 137（sha `ccebb1dd…`）、action 133（`9efc41b6…`）、objects 76（`86952f3d…`）、scene 95（`02d840f6…`）；held-out 诊断切片各 16 条 |

## 3. T0：数据与 CLI（纯 CPU）

- `uv lock --check`、`uv sync --locked` 通过；测试见 §7。
- `scripts/prepare_data.py --source all --build-id local-0001 --limit 400` 产出：
  - spatial 379 单关系 + 21 多关系（左右上下各 100 采样，21 条自关系进 quarantine，closure 成立）；
  - objects 400（Flickr，`train_ready=false`，许可未清）；action 400（K400 模板，70 条非动名词模板标 `grammar_review=required`）；
  - MovieGen 1,525 条无标签清单；SNLI 本地无 pyarrow → 明确 `unavailable_no_pyarrow`。
- 重复运行同参数得到相同内容哈希；`data/smoke`（engineering_only）与 `data/processed`（weak/candidate）分离。

## 4. T1：随机微型模型闭环（CPU，205,376 参数）

四任务各训练 6–8 步（`configs/smoke/*-cpu-tiny.json`）：

| 任务 | base_frozen | adapter_changed | prompt 全掩码 | 监督片段=目标 | 过预算样本 |
| --- | --- | --- | --- | --- | --- |
| spatial | ✅ | ✅ | ✅ | ✅ | 0 |
| action | ✅ | ✅ | ✅ | ✅ | 0 |
| objects | ✅ | ✅ | ✅ | ✅ | 0 |
| scene | ✅ | ✅ | ✅ | ✅ | 0 |

- adapter 保存/重载、三个命名 adapter 硬路由、`predict_base`（禁用 adapter）、Scene 单模型入口、严格输出解析与失败路径均有测试覆盖。
- 记忆诊断：spatial 8 条 fixture 训练 800 步后仍无法复现目标（loss ≈6.5，输出为 JSON 形状但内容错误）。**这与计划预期一致：微型模型只验证管线，不具备语义能力，不作为效果证据。**

## 5. T2：真实底座单卡 100 步

配置 `configs/smoke/*-qwen3-0.6b.json`：LoRA r16/α32 all-linear、BF16、梯度检查点、microbatch1×累积4、`max_length=2048`、`completion_only_loss=true`、无 packing。

| 任务 | 样本 | 远端 4090 墙钟 | 步/秒 | 本地 4060 墙钟 | 训练 loss（末） | 可训练参数 | 底座参数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial | 137 | 86.4 s | 1.16 | 107.6 s | 0.061 | 10,092,544 | 596,049,920 |
| action | 133 | 72.7 s | 1.38 | 102.0 s | 0.060 | 10,092,544 | 596,049,920 |
| objects | 76 | 70.5 s | 1.42 | 105.6 s | 0.183 | 10,092,544 | 596,049,920 |
| scene | 95 | 67.7 s | 1.48 | 100.7 s | 0.364 | 10,092,544 | 596,049,920 |

- 显存：远端物理卡 2 从 1,533 MiB 增至峰值 ≈5,455 MiB（进程约 3.9 GB）；本地 4060 峰值 ≈4.9 GB（含桌面占用）。
- 四任务均 `base_frozen=true`、`adapter_changed=true`；mask 报告示例（spatial）：123 tokens 中 16 个受监督，prompt 全 `-100`，监督片段解码恰为目标 JSON，prompt 探针未泄漏。
- **loss 0.06–0.36 是小样本记忆，不是泛化**：spatial 的训练集只有 137 条弱监督模板，100 步即接近记住。
- 本任务未使用 8B 底座：Qwen3-8B 约 17 GB，远端 home 仅 25 GB 空闲且到公网约 0.5 MB/s，正式底座选择见 §9。

## 6. T3：推理与初步验证

held-out 切片 = 同一来源但**未进入本次 mix** 的弱监督/候选记录（16 条/任务），仅作管线诊断：

| 任务 | adapter F1 | adapter 全对 | 底座 F1 | 底座全对 | 无法解析 |
| --- | ---: | ---: | ---: | ---: | ---: |
| spatial | 1.000 | 1.000 | 0.867 | 0.750 | 0/16 |
| action | 0.938 | 0.938 | 0.375 | 0.375 | 0/16 |
| objects | 0.909 | 0.688 | 0.000 | 0.000 | 0/16 |

（本地 4060 复跑得到 objects F1 0.895、exact 0.625，其余一致；同一 bf16 模型在不同设备上有细微数值差异，不能当作确定性复现证据。）

**Scene（12 条工程 fixture，属于训练集，只作调试）**

| 解码方式 | 可解析 | macro-F1 | 全对 |
| --- | ---: | ---: | ---: |
| 严格（默认，单标签契约） | 5/12 | 0.667 | 1.000（在可解析子集上） |
| `--scene-stop-at-newline` | 12/12 | 1.000 | 1.000 |

关键现象：Scene 模型**先输出正确标签**，随后继续生成解释（如 `contradicted\n\nThe caption describes…`），严格解析因此判失败。把训练步数提高到 300/600 步（loss 降到 0.135/0.069）仍不能消除该行为，说明这不是单纯训练不足，而是"单标签契约 + 生成式底座"的接口问题。三种后续选择：① 推理端加标签边界约束（本轮已验证可行）；② Scene 改用三分类头，从结构上消除续写；③ 目标里显式加入终止符并扩大真实场景数据。**未做结论性选择。**

## 7. 测试与验收

```bash
uv lock --check && uv run --no-sync pytest -q          # 本地：87 passed
# 远端同仓库：训练栈测试同样通过（含 100 步真实底座复用同一套 mask/冻结检查）
```

覆盖：JSONL/任务契约、跨度与词表门禁、teacher 客户端与预算账本（注入传输、无网络）、pilot 解析确定性、清洗脚本计数闭合与可复现、指标分母与失败样本、训练数据渲染、completion mask、底座冻结、adapter 保存重载、硬路由与未知任务拒绝、超预算样本拒绝、严格输出解析。

## 8. 未完成 / 未验证

- 未启动正式训练；未确定正式底座（8B 级）与权重来源；未下载任何 8B 权重。
- 长度泛化：smoke mix 的 action/scene 全在 1–14 词桶，spatial/objects 也以 1–14 为主；**201+ 词与 401+ 桶为空**，S4 的分桶训练/推理报告尚不能出具。
- 无人工金标准、无第二标注者、无真实 Tag2Text 场景数据；teacher 候选未审核；Flickr 许可未澄清；MovieGen 为 CC-BY-NC 4.0。
- 只做了单卡单任务串行；未测多卡、未测更长序列的显存与吞吐（本轮最长只到 260 tokens）。
- `evaluate_smoke.py` 的 Scene 严格指标会把"标签正确但续写"计为失败，这是契约要求；报告同时给出停止约束下的结果，不改契约。
- 未 commit/push；远端工作区为 rsync 后的未提交状态。

## 9. 对正式端到端训练的建议（待用户决策）

1. **正式底座**：优先在 Qwen3 家族内选 8B 级并固定 revision；注意锁定版 `transformers 4.52.4` **不支持 `qwen3_5`**，因此共享缓存中的 Qwen3.5 系列（0.8B/2B/9B）不能直接用，若要用需升级锁并重新验证。
2. **数据**：Action 需要"K400 约束 + 人工确认别名"，不能直接用 teacher 自然动作；Objects 需要摄影/后期词汇排除表；Scene 需要真实 Tag2Text 输出与人工标签，或用三分类头替代。
3. **长度**：先补 50–200 词与 201–400 词的自然 prompt 来源（含 MovieGen 长 prompt 的人工标注），再谈长度鲁棒性。
4. **评测**：固定独立测试与双人仲裁后，才把本轮 0.9+ 的弱监督数字升级为效果结论；正式门槛应预注册。

## 10. 产物

| 路径 | 内容 |
| --- | --- |
| `runs/smoke/{cpu-tiny,gpu-qwen3-0.6b}-<task>/` | adapter、`training_config.json`、`run_summary.json`、`run_manifest.json`（ignored） |
| `runs/smoke/eval-*/{predictions.jsonl,metrics.json}` | 逐条预测与完整分母指标（ignored） |
| `data/smoke/mix-0001/`、`data/processed/local-0001/` | 冒烟集与清洗构建（ignored） |
| [运行手册](runbook.md) | 全部命令、路径与 11 条踩坑记录 |
| [试标报告](teacher-pilot-report.md) | teacher 19 次请求与结论 |
