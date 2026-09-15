# Supplementary experiment utilities

The supplementary run is resume-safe and does not modify the frozen
`data/`, `results/`, `splits/`, or historical `runs/` contents.  New planned
records live under `output/supplementary_20260914/`.

Generate the fixed 16-row CSV and 8-row paper table from the currently
available E0 cache:

```bash
uv run python scripts/generate_supplementary_tables.py --no-compile
uv run python scripts/validate_supplementary_outputs.py \
  --main-csv figures/supplementary_main_table.csv \
  --tex figures/supplementary_main_table.tex
```

When repair outputs are available, add `--repair-metrics` pointing to the
output of `scripts/evaluate_pairwise_statistics.py` and
`--repair-predictions-root` pointing to the directory containing
`<dimension>/predictions.csv`.  Missing dimensions remain `--`; coverage and
source counts are recorded in `figures/supplementary_main_table.metadata.json`.

The TeX uses XeLaTeX, `fontspec`, Times New Roman, `booktabs`, and a 7 inch
resize box.  The script emits a blocked compile status when `xelatex` is not
installed.  `pdftoppm` is used for the PNG export when available.

Create a deterministic eight-dimension intervention manifest (including
positive, reverse, and stability controls):

```bash
uv run python scripts/generate_supplementary_manifest.py
uv run python scripts/validate_supplementary_outputs.py \
  --manifest output/supplementary_20260914/supplementary_manifest.jsonl
```

Attach a base JSONL with `dimension`, `base_id`, `split`, `prompt`, and `path`
fields using `--bases`.  The generator records hashes and video metadata slots
for later materialization, but never invents a score or video.

After workers append `score`/status fields to a copy of the manifest, summarize
with base-level macro statistics:

```bash
uv run python scripts/summarize_supplementary_statistics.py \
  --input output/supplementary_20260914/scored_manifest.jsonl \
  --output output/supplementary_20260914/statistics.json \
  --csv output/supplementary_20260914/statistics.csv
```

The independent unit is `base_id`; stable controls expose absolute change and
coefficient of variation, while ordered interventions expose Spearman and
strict-order rates.  Blocked or failed rows stay missing.

For remote execution on H200, use the cloned workspace under
`/data/chenjiayu/wenbiao_zhao/vbench-audit`, run `uv sync --locked`, and keep
the four workers serialized by dimension. The H200 allocation is physical
GPUs 4--7; each worker receives one physical GPU through its own
`CUDA_VISIBLE_DEVICES` mask and uses logical `cuda:0` inside that process.

Official sampled videos are fetched with
`scripts/download_vbench_official_by_dimension.py` from the remote
`<model-pack>/<dimension>` directory through `https://hf-mirror.com`. The
current VBench archive exposes `subject_consistency`, `human_action`,
`spatial_relationship`, `scene`, `multiple_objects`, and
`overall_consistency`. Dynamic Degree and Motion Smoothness officially reuse
the Subject Consistency prompt suite and videos; after that directory is
verified, run `scripts/link_shared_official_dimensions.sh DATA_ROOT` to create
`dynamics_degree`, `dynamic_degree`, and `motion_smoothness` links under each
generator. Human labels and scoring formulas remain dimension-specific.

GPU orchestration is deliberately dimension-serial.  The schedule starts one
dimension, launches four independent shards on logical GPUs 0--3, waits for
all four shards, and only then advances to the next dimension.  It never
terminates processes owned by another user:

```bash
uv run python scripts/run_supplementary_orchestrator.py
uv run python scripts/run_supplementary_orchestrator.py \
  --dimension dynamic_degree --worker-command \
  'uv run python worker.py --dimension {dimension} --shard {shard} --output {output}'
```

The second command still emits a schedule only.  Add `--execute` after the
worker command has been reviewed and the H100 resource check is clear.

## H200 四卡计时记录

在 H200-target-server 的物理 GPU 4--7 上完成了一组 `group_id=310` 的四生成器视频 smoke：CogVideo、LaVie、ModelScope、VideoCrafter 各 1 个，共 4 个视频，四个 worker 同时启动。LaVie、ModelScope、VideoCrafter 的媒体时长均为 2.0 s；CogVideo 的 GIF 没有写入 duration 元数据，记录为 33 帧，不能把未知时长冒充为秒数。端到端 worker 用时（含每个进程的模型加载和单视频推理）分别为 16.731 s、16.828 s、16.840 s、16.842 s，四卡墙钟总耗时为 **16.842 s**，4 个视频全部成功。计时原始记录保存在 `output/supplementary_20260914/timing/group_310/timing.json`；远端代码为 `vbench-audit` commit `1878d39`，锁定上游 VBench 为 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。

Motion Smoothness 的官方 AMT 不接受 GIF 容器；对原始 CogVideo GIF 只在每个 shard 的新运行目录缓存 16 FPS 临时 MP4，原始目录和 manifest 不改写，日志同时保留原始相对路径和缓存路径。

## 当前可复现实验 checkpoint

截至 2026-09-15，H200 的 Motion Smoothness repair 已经用 4 卡完成首批
722 条、再用 8 卡完成剩余 718 条，合并后为 1440/1440 条，全部为
`succeeded_scalar`。8 卡运行与同机已有的 `sglang` 进程共存，没有发送
终止信号。当前汇总值为：Dynamic Degree official/repair
`0.624/0.255`、Subject Consistency `0.898/0.909`、Motion Smoothness
`0.952/0.753`；对应 test tie-aware human preference 为
`0.684/0.590`、`0.585/0.596`、`0.636/0.395`。自然集结果和逐视频记录
在 `output/supplementary_20260914/`，表格快照在 `figures/`。

### 直接启动一个维度

`launch_dimension_shards.sh` 只启动新 worker，不会 kill 既有进程。默认是
4 卡；在 H200 上可以显式给 0--7 卡，在 H100 上改成 0--3 卡：

```bash
# H200：一个维度用 8 卡并行
GPUS=0,1,2,3,4,5,6,7 \
DATA_ROOT=/data/chenjiayu/wenbiao_zhao/vbench-official-v1 \
UPSTREAM=/data/chenjiayu/wenbiao_zhao/VBench-fd18b3d-clean \
bash scripts/launch_dimension_shards.sh motion_smoothness \
  data/processed/e0_scoring_manifest.csv \
  output/supplementary_20260914/repair_shards/motion_next

# H100：后续维度按四卡运行
GPUS=0,1,2,3 \
DATA_ROOT=/data/chenjiayu/wenbiao_zhao/vbench-official-v1 \
UPSTREAM=/data/chenjiayu/wenbiao_zhao/VBench-fd18b3d-clean \
bash scripts/launch_dimension_shards.sh dynamic_degree
```

每个分片完成后先检查 `shard*/<dimension>/repair_results.jsonl` 的行数，再
用 `scripts/merge_repair_shards.py` 合并。中断后跨两组 shard 恢复时使用
`scripts/combine_partial_repair_shards.py`，它按冻结 `video_uid` 去重并强制
检查完整覆盖；统计和表格命令如下：

```bash
uv run --no-sync python scripts/merge_repair_shards.py \
  --dimension dynamic_degree \
  --shards-root output/supplementary_20260914/repair_shards/dynamic_degree \
  --output-root output/supplementary_20260914/repair
uv run --no-sync python scripts/evaluate_pairwise_statistics.py \
  --predictions-root output/supplementary_20260914/repair \
  --dimension dynamic_degree \
  --output output/supplementary_20260914/repair/dynamic_degree/metrics.csv \
  --bootstrap-iterations 2000 --seed 2026
uv run --no-sync python scripts/generate_supplementary_tables.py \
  --official-metrics output/supplementary_20260914/official_scores/official_metrics.csv \
  --official-scores-root output/supplementary_20260914/official_scores \
  --repair-metrics output/supplementary_20260914/repair/repair_metrics.csv \
  --repair-predictions-root output/supplementary_20260914/repair \
  --output-dir figures
```

上述表格命令使用 XeLaTeX 的 `standalone` 画布，生成的 PDF 页面和 PNG
画布均为 7 英寸宽；若远端没有 XeLaTeX，先在远端生成 CSV/TeX，再把这两
个文件同步到有 XeLaTeX 的工作区编译。
