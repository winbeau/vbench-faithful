# 迁移与发布验收

## 代码与服务器

- 新仓库：<https://github.com/winbeau/vbench-repair>（私有），默认分支 `winbeau`。
- 来源：`msy625/vbench-audit` 的 `winbeau@fdf4890`；初始完整快照为 `809a42f`。
- 保留历史以及原工作区中最初 231 个未忽略的新文件。迁移期间原工作区仍有实验在运行，
  补充快照冻结于 `2026-09-23T10:05:52Z`，共 1,037 个源文件；逐文件哈希见
  [`source-snapshot.json`](source-snapshot.json)。原工作区保持原状，快照之后的研究另行同步。
- 独立本地环境：Python 3.11.14、锁检查通过，原有测试 **1236 passed / 3 skipped**。
- 新增发布合约测试 **10 passed**，检查标签方向、平局、跨维标签独立、
  错误生成器引用、路径越界、变更批次后的精确续传以及提交失败后的归档恢复；补充 Dynamic anchored/450 源码后相关测试 **11 passed**。
  12 个 CLI 的 `--help` 在本地及 H100 均通过。
- H100 checkout：`/root/wenbiao_zhao/vbench-repair`，仓库专用只读 deploy key；
  已成功从私有 GitHub fetch。运行入口：`scripts/h100_python.sh`。
- GPU runtime 单独复用现有 Python 3.10.20、torch 2.5.1+cu121；
  显式加载新 checkout 的包。未替换既有模型环境。
- Dynamic 原版与默认 Repair 对一个真实官方视频均成功：Origin=1.0，
  Repair=0.04317164457276153；两者公式/量纲不同，不据此作效果比较。
- 输入、环境、GPU、完整逐后端结果保存在 [`h100-smoke/`](h100-smoke/)。
  这只验证一维的运行链路，不是所有模型后端的 GPU parity 认证。

## 数据与权重

- 数据：<https://huggingface.co/datasets/xju-arlab/vbench-repair>（公开）。
- 权重：<https://huggingface.co/xju-arlab/vbench-model>（公开）。
- 模型已发布四维，每维仅一个 adapter；选择规则、版本、开发指标与字节哈希见
  [`model-selections.json`](model-selections.json)。Dynamic 按用户要求只记录、不发权重。
  Object Class / Color 只有固定步数训练记录，保留的 step 200/300 均无 dev 评估，
  本次遵照 best-only 标准先记录，不把最终 step 300 宣称为 dev 最优；见
  [`additional-model-records.json`](additional-model-records.json)。模型远端验收见
  [`model-verification.json`](model-verification.json)。
- 原始数据完整规划 16 份标注、27,720 个维度内条目、19,400 个独立源路径。
  Background 的 1,710 个上游错误引用显式保留，不能自动当成可用评测配对。
- 完整上传及远端验收结果由 `dataset-verification.json` 记录；该文件尚未产生时，
  不将上传任务描述为完成。

## 复现发布

`scripts/publish_dimension_origins.py` 根据原始标注和固定 HF 来源清单生成目录与配对；
`publish` 子命令在 `output/` 保存批次状态，可以重试续传。默认尝试 hf-mirror，
只有显式 `--allow-official-fallback` 才启用官方回退。此次用户已授权此回退。

`scripts/archive_counterfactuals.py` 按 `configs/publication/` 的来源清单，从源服务器
生成约 512 MiB 分片并逐个上传。每个成员与分片均记录 SHA-256，上传后只移除
临时 tar；合并提交模式允许在确认二进制预上传成功后移除临时 tar，保留 pending 状态及源文件以供确定性重建。原视频、标注、数组、构造参数及研究记录不变。

`scripts/publish_best_models.py` 使用原训练 dev probe 在保留的 checkpoint 中选择；
不上传底座、优化器状态或已知较差的平行版本。Action step 150 已被原训练清理，
因此其发布记录明确标注“best retained”。

`scripts/verify_repair_publication.py` 对照远端快照检查全部原片、原标注、分片哈希、
成员覆盖、可用偏好引用及下载抽样。运行时需指定数据 staging 根目录、两台服务器、本地及 H100 补充归档的
spec，确保尚未提交 manifest 的实验也被判为缺失。
