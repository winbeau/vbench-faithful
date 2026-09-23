# vbench-repair 整理与迁移计划

日期：2026-09-23。代码仓库：`winbeau/vbench-repair`（私有）。
数据仓库：`xju-arlab/vbench-repair`（公开）。HF 请求默认使用
`HF_ENDPOINT=https://hf-mirror.com`。

用户后续确认：原始数据补齐 **全部 16 维**；模型另发到
`xju-arlab/vbench-model`，**每维一个目录，只保留 best available model**。
Dynamic 暂不发布权重，只记录候选及验证。镜像实测不可用时允许官方回退。

## 1. 代码完整保存

- 从 `vbench-audit` 的 `winbeau@fdf4890` 保留 Git 历史。
- 将原工作区中 Git 已跟踪和未忽略的新文件一并迁入独立 `vbench-repair/`，
  提交最新算法、测试、配置、图示与实验记录。原工作区保留原状。
- 原始视频、模型权重、虚拟环境和中间缓存继续留在代码仓库之外。
- 保持 Python 包名和 CLI，不为仓库改名制造接口变化。
- 运行锁检查、独立环境安装、全部算法/合约测试与所有 CLI help。

## 2. H100 连通与实际运行

- 在 `/root/wenbiao_zhao/vbench-repair` 建立独立 checkout。
- 配置专用只读 deploy key，验证能从新私有仓库 fetch。
- 复用 `/root/wenbiao_zhao/venvs/vbench` 中已安装的 CUDA 模型依赖；
  通过显式源码路径确保实际加载新 checkout，不修改旧环境的安装。
- 固定官方上游 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`，使用既有本地权重。
- 对真实官方视频执行 Origin/Repair GPU smoke，保存代码 SHA、环境、输入和结果。
  一条 smoke 不等同于所有维度的模型 parity 认证。

## 3. 数据按维度优先

```text
README.md
catalog.json
provenance/
dimensions/<canonical_dimension>/
  README.md
  origin/
    manifest.jsonl
    videos/<generator>/<original_dimension>/<original_filename>
  human_preference/
    <original_official_name>.json
    pairs.jsonl
  counterfactual/<version>/
    manifest.json
    files.jsonl
    data-00000.tar
    records.jsonl  # when source comparison records exist
```

- 维度规范名使用包名的下划线形式；显式映射 `dynamics_degree -> dynamic_degree`、
  `multiplt_object -> multiple_objects`，保留原始标注文件名和原维度值。
- 人类标注原文件逐字节保留；另生成相对路径比较清单，标明标签方向、平局及源行。
- Dynamic、Motion、Subject 共享视频内容，但不共享标注。
  本地以硬链接或来源引用避免产生分叉；上传内容以 SHA-256 核对。
- 不转码原 MP4/GIF，不根据残留问题文本重新判维，不合并大小写不同的原文件。
- 反事实按实验版本保留；早期七维、后续确认集、八维主表和最新 Dynamic
  局部纹理抖动实验分开，避免把历史失败/诊断当作最终成功结果。
- 语义/文件名干预可能没有新视频：保留源视频、条件和配对清单，不伪造媒体。
- 每个维度报告实际原片、标注、比较、反事实数量和缺失项。
  未有资料的维度不标记为“完整 VBench 1.0”。
- 服务器磁盘紧张，优先硬链接。大量逐帧资产可采用可索引分片归档；
  模型特征缓存与模型权重不属于本次数据发布内容。

## 4. 发布与验收

1. 盘点来源并冻结整理清单，检查所有标注引用和反事实媒体存在。
2. 写数据卡、维度映射、split 与来源说明；不改变原始数据的使用条款。
3. 通过 hf-mirror 实际测试小文件上传，再按维度分批上传，保留续传状态。
4. 远端核对文件列表、字节数、哈希/提交版本，并下载抽样检查。
5. 将代码 SHA、H100 smoke、数据总数、远端版本与未完成范围写入验收记录。

## 模型选择与发布

- Spatial v8、Scene v8、Action v9、Objects v6：在各自仍存在的 checkpoint 中，
  按原训练记录的 dev exact match 选择，并列时取较晚步数；不使用 test 选模型。
- 原 dev probe 仅 24 条，不夸大为完整独立验证。
- Action 的最高已记录分数来自已被原训练流程清理的 step 150；保留下来的
  200/250/300 并列，因此发布 300，并在 `selection.json` 明确记录此限制。
- 权重逐字节保留，提供 SHA-256、训练配置、tokenizer、来源代码 SHA。
  适配器配置的私有底座路径改为固定 revision 的 `Qwen/Qwen3-8B`，另保留原配置。
- Dynamic 的 joint/anchored 选择记录保留，遵照用户明确决定不上传其权重。

## 已发现的上游数据问题

官方 Drive 的 16 份 JSON 已取得。15 维路径可与固定 HF 视频版本精确匹配。
`Background_Consistency.json` 有 1,710 条路径缺失或生成器键冲突，例如
CogVideo 被写作 MP4、LaVie 键指向 CogVideo 中文 GIF；不得猜测修复标签。
保留原始文件与逐项问题清单，另提供完整 Scene 官方视频 suite 作为候选素材。
这些问题记录为源数据限制，不能将相关偏好配对标记为可直接评测。

按维度布局共 27,720 个原片条目，跨维度去重为 19,400 个实际原文件。
LFS 原片从固定 HF revision 服务端复制；Git 二进制原片校验源 Git blob 后
重新上传，并补 SHA-256。服务器反事实打为约 512 MiB 的 tar 分片，每个成员有哈希。
本机两个 dev5 版本使用 8 MiB 目标分片，避免低带宽下的大请求超时；大于目标值的
单个文件独占一个分片，续传不能改动分片大小。

## 初始事实

- 新独立工作区的全量 CPU 测试：1236 passed、3 skipped。
- H100 可连接，既有 Python 3.10.20 / torch 2.5.1+cu121 可使用 CUDA；
  本地开发锁定环境为 Python 3.11.14 / torch 2.14.0+cpu，两者分别记录。
- 初始 H100 可用磁盘约 13 GB，不能复制整份视频/特征缓存。
- HF 目标已存在且公开，初始仅 `.gitattributes`；不覆盖其他数据。
- 镜像 `whoami` 失败，但 `repos create --exist-ok` 成功；以真实上传结果验收。

## 限流后的提交策略

HF 实际返回每仓库每小时 128 次提交上限。原片续传按已提交的文件范围去重，
允许增大批次而不遗漏原先的小批次；剩余 Git/LFS 原片分别合并提交，16 维元数据
一起提交。反事实使用官方 `preupload_lfs_files` 先传二进制，再将同一来源主机的
全部剩余分片与索引合成一个 commit。`--not-before` 可在限流窗口内先传文件内容，
稍后再提交。未提交的分片在状态文件中明确记为 pending；重试会重建并校验相同哈希。

补充本地两个 dev5 纹理位移版本和已废弃的合成开发集；废弃协议单独标识，不进入当前主实验。

本地废弃合成开发集在 H100 已有逐字节一致的副本，6,922 个文件 SHA-256 全部匹配。
因此使用 `configs/publication/h100-supplementary.json` 从 H100 发布这一版本，
`local-counterfactuals.json` 仅负责两个 dev5 纹理版本。

Object Class / Color 保留 step 200/300，原训练没有 dev 评估和 best checkpoint 决策。
已保存训练记录、两步权重哈希和论文采用 step 300 的事实；用户后续明确要求发布论文采用的 step 300，并注明未经 dev 选优。
