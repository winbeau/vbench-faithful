# Background Consistency：真实后端与独立验证

后续构造语义审核发现旧干预的误选和漏分。新的 104 类构造已完成 680 条开发输入重跑，
十条已审核干预的 origin/现有 repair MAE 为 0.011578/0.003054，未达到 origin ≥0.10 的新目标。
新定位候选未晋升默认，当前按用户要求暂停；当前结论和紧凑证据以
[阶段总结](../../docs/counterfactual-reports/background_repair_checkpoint_20260920.md)为准。
以下原独立测试数字保留其历史协议范围，不作为新构造已正确修复的证明。

2026-09-20 已由 M1 接口占位接通模型、评分和 CUDA 分片。
独立结果与限制见[正式报告](../../docs/counterfactual-reports/background_holdout_20260920.md)，
负结果保留在[开发报告](../../docs/counterfactual-reports/background_development_20260920.md)。

- `--vbench` 使用本地 CLIP ViT-B/32 和锁定 VBench 的视频解码/预处理，
  保留 fp16 特征上的官方逐项余弦；全数据集按帧转移数加权，与 GPU 数无关。
- `--audit --audit-variant diagnostic` / `aggregation` 为全帧对对照。
  当前全帧对实现用 float32 归一化和点积、float64 求均值，具有明确的精度差异，
  不能把接近数值量化级别的变化单独归因于时间聚合。
- `--audit --audit-variant repair` 映射到 `patch_frame_calibrated`：独立逐帧自动前景定位，
  按背景补集池化同一 CLIP 的 patch token，同精度全帧对余弦聚合，固定增益 1.75。
  `frame_official`、`frame_all_pairs`、`union_official`、`union_all_pairs` 保留历史填灰消融；
  `patch_frame_balanced` 保留开发期时间聚合混合消融。
  背景不足帧按固定分母零贡献，没有前景检出时保留全幅 patch，不删除失败样本。
- 模型配置示例为 [`h100-models.toml`](../../configs/background-repair/h100-models.toml)。
  显式本地 checkpoint 与哈希检查阻止自动下载；缺少配置返回逐视频 `failed`、`score=null`。
  CLI 的 official 和 repair 均检查 CUDA；直接 Python evaluator 可显式传入测试设备。
- public batch 入口与实验脚本使用相同的确定性和 TF32 关闭设置。
  真实含前景片段曾暴露默认 cuDNN TF32 造成的 0.0003128 repair 差异，统一后精确对齐。
- MobileSAM 与 COCO box adapter 位于 `audit-models`。本 metric 不导入 subject metric，
  也不读取数据构造掩码。每个实际干预版本重新定位，允许按完整内容/代码/配置哈希复用缓存。

独立测试：1,040 条自然视频、1,560 对 background 偏好、188 条干预基底的 1,504 个版本。
完整主体糊化平均绝对变化从 origin 0.016240 降至 repair 0.006733，自然准确率从 52.12%
提高到 61.22%，三时窗平均背景响应保留 88.80%，全部 18 项冻结联合检查通过。
所有原始评分、失败例和负候选均保留。存在粗定位缺失与个别较大分差，不声称逐片段完美稳定。

真实验证：1,040 条自然测试视频的 origin 与上游最大误差 0；188 条 clean 的视频/PNG
独立重算在全部 19 个预先声明方法上最大误差 0。显式候选和提升后的默认 `--both` CLI
均与冻结实验分数精确一致。默认别名是在独立检验通过后提升，评分公式和冻结配置未改。
完整测试 **604 passed、3 skipped**，锁文件、CPU torch overlay、9 个 CLI help 和 diff 检查通过。
