# E0 S0–S2 本地审计记录

截至 2026-09-11，本地检查结果：

- 官方 VBench 代码副本：`/home/msy625/vbench1/VBench-2.0`。
- 官方 human-preference JSON：未在本地找到；官方 README 指向 Google Drive 文件夹。
- 官方视频包：未在本地找到。
- 逐视频 automatic-score cache：未在本地找到；仓库包含 evaluator 源码和权重路径说明，但没有本实验所需的四模型逐视频输出。
- 当前未下载权重、未配置 CUDA、未运行 evaluator、未生成视频、未执行 E0-B。

已加入：

- `scripts/01_audit_data.py`：schema/计数审计；
- `scripts/02_build_pairs.py`：将官方 nested `human_anno` 展平为 canonical `pairwise_master`；
- `scripts/03_smoke_stats.py`：人工 toy records 的 prompt-disjoint split、tie、去重 smoke test；
- `data/README.md`：正式输入约定。

正式数据数量目前均为 `NA（原始标注未落地）`，不是零，也不是预期值。
