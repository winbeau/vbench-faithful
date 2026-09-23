# 运行手册：造数据与训练的可用命令

本文件汇总本轮实际跑通的命令、路径约定和踩坑记录。所有命令都在 `vbench-prompts-compile`
项目根目录执行；未跑通的步骤会明确标注"未验证"，不写成已知可用。

## 1. 环境与路径约定

| 项 | 本地开发机 | 远端 `rtx4090` |
| --- | --- | --- |
| 项目 | `~/Papers/ICASSP2027-VBench-Audit/vbench-prompts-compile` | `/data1/wenbiao_zhao/vbench-prompts-compile` |
| 解释器 | `.venv`（uv 0.9.17，Python 3.11.14） | 同左，uv 在 `/data1/wenbiao_zhao/tools/uv-0.9.17/uv` |
| 权重/缓存 | `~/models/…` | `/data1/wenbiao_zhao/models/…`，`HF_HOME=/data1/wenbiao_zhao/hf` |
| uv 缓存 | 默认 | `UV_CACHE_DIR=/data1/wenbiao_zhao/uv-cache`（home 只剩 ~25G，必须外置） |
| GPU | RTX 4060 Laptop 8GB | 8 × RTX 4090 49GB（本轮使用 1/2/3/4/6 中较空的一张） |

基础开发（无 torch）：

```bash
uv --version                 # 必须 0.9.17
uv lock --check
uv sync --locked             # 注意：会移除 train extra
uv run --no-sync pytest
```

训练环境（含 torch/TRL/PEFT）：

```bash
uv sync --locked --extra train
uv run --no-sync python -c 'import torch, transformers, trl, peft; print(torch.__version__)'
# 之后不要再跑不带 --extra train 的 sync，否则训练依赖会被移除
```

## 2. 造数据

```bash
# 2.1 五类原料清洗（小批量、可复现、计数闭合）
uv run --no-sync python scripts/prepare_data.py --source all --build-id local-0001 --limit 400
#   -> data/processed/local-0001/{candidates,quarantine,inventory,counts.json,manifest.json}
#   -> data/smoke/local-0001/{spatial,action,objects,scene}.jsonl（工程 fixture，90条scene/12条其余）

# 2.2 混合冒烟集（工程fixture + 弱监督 + teacher候选，只用于冒烟）
uv run --no-sync python scripts/build_smoke_mix.py --mix-id mix-0001
#   -> data/smoke/mix-0001/{task}.jsonl + manifest.json（含各来源占比与长度桶覆盖）

# 2.3 teacher 试标（每次 POST 先记账；预算用满即拒绝）
uv run --no-sync python scripts/run_teacher_pilot.py --config configs/teacher/pilot-01.json --dry-run
uv run --no-sync python scripts/run_teacher_pilot.py --config configs/teacher/pilot-01.json --confirm-budget 19
#   -> output/teacher/<run-id>/{plan.json,ledger.jsonl,candidates.jsonl,quarantine.jsonl,summary.json}
#   -> output/teacher/BUDGET.json（authorized_total=20，当前已用满）
```

## 3. 训练与评测

```bash
# 3.1 本地随机微型模型（无下载；仅验证管线）
uv run --no-sync python scripts/make_tiny_model.py --output models/tiny-qwen-smoke

# 3.2 CPU 微型模型闭环（T1）：四任务各 6–8 步
uv run --no-sync python scripts/train_adapter.py --config configs/smoke/spatial-cpu-tiny.json
uv run --no-sync python scripts/train_adapter.py --config configs/smoke/action-cpu-tiny.json
uv run --no-sync python scripts/train_adapter.py --config configs/smoke/objects-cpu-tiny.json
uv run --no-sync python scripts/train_scene.py   --config configs/smoke/scene-cpu-tiny.json

# 3.3 真实底座 100 步（T2）；先确认空闲卡，再单卡运行
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/data1/wenbiao_zhao/models/Qwen3-0.6B
REV=c1899de289a04d12100db370d81485cdf75e47ca
CUDA_VISIBLE_DEVICES=<空闲卡> .venv/bin/python scripts/train_adapter.py \
    --config configs/smoke/spatial-qwen3-0.6b.json --model "$BASE" --model-revision "$REV"
# scene 用 scripts/train_scene.py，其余两个 parse 任务同理

# 3.3b 正式底座 Qwen3-8B（revision b968826d9c46dd6066d109eabc6255188de91218）
#   远端预下载：按 models/Qwen3-8B.manifest.json 逐分片 curl -C - 并行拉取，
#   完成后 /data1/wenbiao_zhao/verify_qwen3_8b.sh 校验大小与 SHA256（日志 qwen3_8b_verify.log）。
#   实测：单连接 ~0.4MB/s，5 分片并行 ~2.2MB/s，16.4GB 约 2 小时。
BASE=/data1/wenbiao_zhao/models/Qwen3-8B
CUDA_VISIBLE_DEVICES=<空闲卡> .venv/bin/python scripts/train_adapter.py \
    --config configs/smoke/spatial-qwen3-8b.json --model "$BASE" --model-revision b968826d9c46dd6066d109eabc6255188de91218

# 3.4 评测（adapter vs 未微调底座，完整分母）
uv run --no-sync python scripts/evaluate_smoke.py --task spatial \
    --data data/smoke/mix-0001/spatial.jsonl --base-model models/tiny-qwen-smoke \
    --adapter spatial=runs/smoke/cpu-tiny-spatial \
    --output-dir runs/smoke/eval-cpu-tiny-spatial --limit 12 --compare-base
```

## 4. 踩坑与经验（本轮实测）

1. **磁盘**：远端 home 只剩 15G，装不下 8B 权重。项目、uv 缓存、模型、runs 全部迁到 `/data1/wenbiao_zhao/`，home 释放约 11G。
2. **HF 镜像不可用**：`HF_ENDPOINT=https://hf-mirror.com` 在本网络会把 `resolve` 请求 308 重定向到 `huggingface.co`，`huggingface_hub` 因此报 `FileMetadataError / LocalEntryNotFoundError`。可用路径：直连 `huggingface.co`（本地约 1.1MB/s）或 ModelScope `resolve` URL + `curl`（远端约 0.4–0.5MB/s）。
3. **跨机带宽极低**：本地→远端 `scp` 20MB 用了 6m47s（约 51KB/s），`rsync` 大批量还会 `exit 255`。结论：大文件不要跨机搬，在哪台机器用就在哪台下；小文本（代码、JSONL、manifest）走 `rsync` 正常。
4. **单连接限速、并行有效**：同一镜像单连接约 0.13–0.5 MB/s，多个分片并行可把总带宽提到约 2.2 MB/s；大权重务必并行分片下载并逐文件校验哈希。
5. **transformers 版本决定可选底座**：锁定版 `transformers 4.52.4` 支持 `qwen2/qwen3/qwen3_moe`，**不支持 `qwen3_5`**。共享缓存里的 Qwen3.5 系列（0.8B/2B/9B 等）因此不能直接用；要用需升级锁并重新验证，或改选 Qwen3 家族权重。
6. **completion mask 要用真实数据集验证**：自己构造 `input_ids/labels` 再喂 `data_collator` 会得到"未掩码"的假象。正确做法是取 `trainer.train_dataset[0]`，再检查监督片段解码后是否等于 target、且 prompt 探针没有泄漏。
7. **LoRA 冻结检查要同对象比较**：`PeftModel` 会给参数名加前缀，训练前后必须都通过 `trainer.model.get_base_model()` 取指纹，否则会误报"底座被改动"。
8. **Qwen 不接受 `token_type_ids`**：自定义 tokenizer 会带上它，生成前需 `pop`，否则 `generate` 直接抛 `ValueError`。
9. **检查 adapter 真的更新了**：除底座冻结外还对比 LoRA 参数指纹，若未变化说明梯度没到 adapter（例如梯度检查点配置问题）。
10. **微型模型只证明管线**：随机微型模型训练后能产出 JSON 形状的文本，但语义不正确，不能作为效果证据。
11. **teacher 不能直接造 Action 标签**：19 次试标中 4/5 个 action 请求因不是 K400 类别名被拒；Objects 会把 `Camera tracking shot` 里的 `Camera` 当实体；Spatial 的 `floats above` 这类不及物用法没有第二个具名实体。详见[试标报告](teacher-pilot-report.md)。
12. **工程/弱监督/teacher/金标准四级分离**：`data/smoke`（engineering_only）与 `data/processed`（synthetic_weak / teacher_candidate_unreviewed）物理分开，训练脚本会重新校验每条记录，非法记录直接报错而不是静默跳过。

## 5. 相关文档

- [主计划索引](plans/README.md) ｜ [冒烟主计划](plans/05-smoke-master.md) ｜ [teacher 子计划](plans/10-teacher-data.md)
- [试标报告](teacher-pilot-report.md) ｜ [数据盘点](data/README.md) ｜ [验证记录](verification.md) ｜ [冒烟报告](smoke-report.md) ｜ [决策记录](decisions.md)
