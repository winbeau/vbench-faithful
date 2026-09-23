# 迁移与发布验收

2026-09-23 复现审计另见[复现入口与范围](../reproduction/README.md)：新增固定版本的
轻量复算包和六维训练输入，从 HF 恢复后的九维主表复算已通过。
本页的上传完整性验收不等于全部模型端到端推理/重训已通过。

## 代码与服务器

- 新仓库：<https://github.com/winbeau/vbench-repair>（私有），默认分支 `winbeau`。
- 来源：`msy625/vbench-audit` 的 `winbeau@fdf4890`；初始完整快照为 `809a42f`。
- 保留历史以及原工作区中最初 231 个未忽略的新文件。迁移期间原工作区仍有实验在运行，
  补充快照冻结于 `2026-09-23T10:05:52Z`，共 1,037 个源文件；逐文件哈希见
  [`source-snapshot.json`](source-snapshot.json)。原工作区保持原状。当前 Dynamic 的 18 个新增/更新文件另按
  [dynamic-source-snapshot.json](dynamic-source-snapshot.json) 同步；其来源是未提交工作树，
  不虚构另一条研究 commit。
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
  这是旧默认入口验收；最新 aligned-v1 另通过 H100 单片与 H200 的精确 parity，
  见 [Dynamic 实测](../reproduction/h100-dynamic-aligned/verification.json)。
  仍不是所有模型后端的全量 GPU parity 认证。

## 数据与权重

- 数据：<https://huggingface.co/datasets/xju-arlab/vbench-repair>（公开）。
- 权重：<https://huggingface.co/xju-arlab/vbench-model>（公开）。
- 六个语义维度已发布，每维仅一个 adapter。其中四个 dev 选优维度的选择规则、版本、开发指标与字节哈希见
  [`model-selections.json`](model-selections.json)。Dynamic 按用户最新授权另发布 aligned-v1 评分头及配套冻结 backbone，
  见 [Dynamic 发布说明](../reproduction/DYNAMIC_ALIGNED.md)。原只记录决定已被本次授权替代。
  Object Class / Color 只有固定步数训练记录，保留的 step 200/300 均无 dev 评估，
  用户进一步明确选择论文采用的 step 300 发布，两维均注明未经 dev 选优；见
  [`additional-model-records.json`](additional-model-records.json)。模型远端验收见
  [`model-verification.json`](model-verification.json)。
- 用户明确允许模型库公开 `code/vbench_prompts_compile/` 与 `code/object_color/`
  两份复现源码，共 180 个文件（含说明与来源清单）；完整 GitHub 仓库保持私有。
- 原始数据已发布并核验：16 份标注、6,930 行原标注、41,580 个规范化无序比较，
  27,720 个维度内视频条目、19,400 个独立源路径。
  Background 的 1,710 个上游错误引用显式保留，不能自动当成可用评测配对。
- 反事实共 27 个实验版本、118 个 tar 分片、90,471 个成员（约 30.56 GB 源文件）。
  成员包含原片、控制、派生视频、帧数组、预览与记录，不能当作独立反事实视频数。
- [`dataset-verification.json`](dataset-verification.json) 对全部原片、86 个原始元数据文件、
  16 份原标注、全部分片及成员索引完成远端核验，另下载两种源存储形式的原片和
  一个完整媒体分片核对内容；无完整性错误。被验收的数据 revision 为
  `77da9f257cd6fbb314980aafd37a8114730edc2a`，逻辑数据量约 54.87 GB。
  数据卡与 catalog 的最终提交另记在 [`publication-receipt.json`](publication-receipt.json)。

## 复现发布

`scripts/publish_dimension_origins.py` 根据原始标注和固定 HF 来源清单生成目录与配对；
`publish` 子命令在 `output/` 保存批次状态，可以重试续传。默认尝试 hf-mirror，
只有显式 `--allow-official-fallback` 才启用官方回退。此次用户已授权此回退。

`scripts/archive_counterfactuals.py` 按 `configs/publication/` 的来源清单，从源服务器
默认生成约 512 MiB 分片并逐个上传；本机两个 dev5 版本因带宽限制使用 `--shard-mib 8`，
单个大文件不截断，因此部分分片会超过该目标大小。续传必须保留原分片设置。
每个成员与分片均记录 SHA-256，上传后只移除
临时 tar；合并提交模式允许在确认二进制预上传成功后移除临时 tar，保留 pending 状态及源文件以供确定性重建。原视频、标注、数组、构造参数及研究记录不变。

`scripts/publish_best_models.py` 使用原训练 dev probe 在保留的 checkpoint 中选择；
不上传底座、优化器状态或已知较差的平行版本。Action step 150 已被原训练清理，
因此其发布记录明确标注“best retained”。

`scripts/publish_object_color_models.py` 单独处理用户指定的论文最终 step 300，
核对 checkpoint、最终 adapter 与训练记录三者哈希一致，并明确保存未经 dev 选优的事实。

`scripts/verify_repair_publication.py` 对照远端快照检查全部原片、原标注、分片哈希、
成员覆盖、可用偏好引用及下载抽样。运行时需指定数据 staging 根目录、两台服务器、本地及 H100 补充归档的
spec，确保尚未提交 manifest 的实验也被判为缺失。

最后使用 `scripts/finalize_publication.py` 更新公开数据卡与覆盖目录；它要求远端验收通过，
并保留被验收的数据 revision。服务器磁盘不足时临时分片与上传缓存放在本次专用
`/dev/shm/vbench-repair-*` 目录；完成后先保全续传状态与索引，再清理临时数据。
# Dynamic 外部结果增补（2026-09-23）

[LASIESTA／BMC 外部验证](../reproduction/DYNAMIC_GENERALIZATION.md)补充当前 aligned-v1
的 127 片段／16 源结果；发布协议、逐条分数和审计，不上传外部源媒体、私有工程源码或新权重。
固定发布 revision 与逐文件下载验收见 [外部发布记录](dynamic-generalization-release.json)。
原有数据、权重与运行环境的固定 revision 不改写；下文保留首次迁移／发布记录。
