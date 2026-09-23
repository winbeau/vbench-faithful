# 复现范围、入口与尚未验收的部分

2026-09-23 审计结论：**不能把“上传和哈希验收完成”表述为“全部实验从零复现完成”。**
本次已从 HF 固定 revision 下载到新目录，恢复全部轻量输入并独立重算九维主表；
六维训练输入、原配置、词表和来源也已补齐。全九维重新执行视觉/语言模型推理、
从头训练和所有附表/消融的独立重跑，仍没有完成同等级验收。

资产实际分布在三个地址：私有 [GitHub 代码库](https://github.com/winbeau/vbench-repair)、
公开 [HF 数据库](https://huggingface.co/datasets/xju-arlab/vbench-repair) 和
公开 [HF 模型库](https://huggingface.co/xju-arlab/vbench-model)。
两个 HF 库提供数据、权重和两份获准公开的模型源码；完整工程、本文与执行入口仍在 GitHub。

## 已实际验收的层次

| 层次 | 当前证据 | 能据此说明什么 |
| --- | --- | --- |
| 16 维原始资料、27 个 CF 版本、六个 adapter | 全部远端大小/哈希核对及下载抽样 | 发布内容完整；上游 Background 引用错误仍保留 |
| 九维主表复算 | 从 HF 恢复 222 个轻量文件、校验 79 个解包成员，24,408 个得分单元及 primary/status 与冻结值逐项一致 | 九维主表及其缺失值、共同有效分母可以重算 |
| 四个语义维度计分公式 | 重新运行 `score_matrix.score_one`，使用冻结视觉证据与语义预测 | 不是简单复制最终 CSV，也不是重新执行模型推理 |
| Object/Color/Subject/Background | 从逐样本 GPU 记录重新组对、过滤和汇总 | 证明汇总可重现，不证明 GPU 数值跨环境一致 |
| 六维训练数据 | 四维与原训练主机逐字节一致；两维 shared records 与训练 manifest 哈希一致；schema/来源组隔离通过 | 可以恢复原训练输入；本次未重训六个模型 |
| H100 工程运行 | 最新 aligned-v1 的 H100 单片评分/latent/tokens/像素与 H200 冻结记录完全一致；旧原版/默认 smoke 另保留 | 证明该链路可运行，不是论文九维全部 GPU 复现认证 |

机器证据：[HF 九维恢复收据](restore-nine-dimension-receipt.json)、[九维主表复算](verified-nine-dimension-replay/verification.json)、
[九维复算表](verified-nine-dimension-replay/main-table.csv)、[训练输入检查](training-input-verification.json)。
所有远端 revision、文件和 SHA-256 固定在
[`configs/reproduction/release.json`](../../configs/reproduction/release.json)。

## 一条可实测的复算链路

在本仓库根目录执行。下载工具单独建环境，不改已有 CUDA 环境：

```bash
uv venv .venv-reproduction --python 3.11.14
uv pip install --python .venv-reproduction/bin/python 'huggingface_hub==1.32.0'
.venv-reproduction/bin/python scripts/prepare_reproduction.py \
  --output output/reproduction --allow-official-fallback
.venv-reproduction/bin/python scripts/reproduce_main_table.py \
  --bundle output/reproduction/bundle \
  --model-code output/reproduction/model-code \
  --k400-labels output/reproduction/assets/dimensions/human_action/training/k400-labels.json \
  --output output/reproduction-result
```

默认先请求 `hf-mirror.com`；`--allow-official-fallback` 显式启用镜像失败后的官方回退。
恢复过程核对下载与解包哈希，拒绝路径越界、重复成员和覆盖内容不同的已有文件。
它不启动训练、不调用在线标注服务。可加 `--include-models` 下载六维 adapter 与 tokenizer，
以及 Dynamic aligned 头和配套 backbone，并核对全部权重。复算不需要模型权重或 GPU。

输出中的 9 行均值对应同一批四格有定义的样本，不能拿不同分母的单格均值替换：

| 维度 | 主分析计划 / 四格共同有效 | 当前主实验 |
| --- | ---: | --- |
| Scene | 200 / 200 | 同义场景条件，v8 caption verifier |
| Human Action | 1,200 / 1,200 | 声明域内同义动作接口，v9 + repair-v2.1 |
| Object Class | 14 / 14 | 大小写元数据，确定性编译器是这行的 Repair |
| Subject Consistency | 240 / 240 | 720 候选的预定接受集，start/background_corrupt，hybrid_exclude |
| Dynamic Degree | 450 / 450（900 CF） | aligned-v1；每源先平均两个 8px 种子，450 编码控制单独核验 |
| Background Consistency | 188 / 188 | full/subject_blur，patch_frame_all_pairs_calibrated，系数 1.75 |
| Spatial Relationship | 106 / 106 | 实际视频镜像及预定单向几何支持集；不是框镜像代理 |
| Multiple Objects | 980 / 505 | 完整 16 帧遮挡证据；475 个不完整项保持缺失 |
| Color | 5 / 3 | 原版 CF 仅返回 3 个有效分数；不补零 |

复算通过不改变语义掩码、同义词声明域、模型标注与泛化方面的限制。
详细原协议和统计差别见轻量包中的 `reports/summary.json` 与
[方法总览](../DIMENSION_METHODS_AND_EXPERIMENTS.md)。

Dynamic 另见 [选用版本、权重与复现命令](DYNAMIC_ALIGNED.md)。
新九维重放记录见 [verified-nine-dimension-replay/verification.json](verified-nine-dimension-replay/verification.json)；
原 [verified-replay/](verified-replay/) 保留八维验收历史。

## 六维语义训练与推理材料

| 维度 | 训练输入 | 发布 checkpoint | 限制 |
| --- | --- | --- | --- |
| Spatial | v8，2,679 train / 312 dev | step 600 | 仅冻结的四方向任务空间 |
| Scene | v8，455 / 108 | step 300 | prompt + caption，独立 verifier |
| Action | v9，605 / 67 | step 300 | 历史 step 150 已被清理；当前发布 best retained |
| Objects | v6，7,688 / 811 | step 900 | 弱监督、覆盖与时序过滤限制继续适用 |
| Object Class | 424 条训练输入 | step 300 | 用户指定论文 final，未经 dev 选优 |
| Color | 465 条训练输入 | step 300 | 用户指定论文 final，未经 dev 选优 |

`--include-models` 同时下载六个 LoRA 及 Dynamic aligned 头/冻结 backbone（合计约 2.8 GB）。
Dynamic 使用独立的 `models/dynamic_degree/`，不是 PEFT adapter。

`prepare_reproduction.py` 恢复原始 train/dev、配置、K400 词表以及 Object/Color
的 byte-exact shared records。后者保留冻结 teacher 输出，复训不依赖重新调用付费 API。
逐维 training JSONL 是浏览视图；Object/Color 原训练使用的完整 `records.json` 不做筛选改写。

四维训练环境由下载源码的 `pyproject.toml` / `uv.lock` 固定：Python 3.11.14、
uv 0.9.17、torch 2.7.1+cu126、Transformers 4.52.4、TRL 0.19.1、PEFT 0.15.2。
底座需要另行预置 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`。
这套训练环境与下文旧视觉后端环境分开。

已有底座后重新运行准备命令时加 `--base-model /absolute/path/to/Qwen3-8B`。
生成的 `training-configs/*.json` 只改输入、底座、输出路径，不改实验超参。
在下载的 `model-code` 中安装锁定训练环境，然后调用其脚本，例如：

```bash
repro_work="$PWD/output/reproduction"
uv sync --project "$repro_work/model-code" --locked --extra train
"$repro_work/model-code/.venv/bin/python" "$repro_work/model-code/scripts/train_scene.py" \
  --config "$repro_work/training-configs/scene.json"
"$repro_work/model-code/.venv/bin/python" "$repro_work/model-code/scripts/train_adapter.py" \
  --config "$repro_work/training-configs/spatial_relationship.json"
```

Action / Objects 分别替换为 `human_action.json` / `multiple_objects.json`。
Object/Color 使用本仓库 `scripts.object_color_semantics train`，将 `--output` 指向
恢复出的 `output/reproduction/object-color`，并按[原运行手册](../object-color-repair.md)
设置 GitHub checkout 的 PYTHONPATH、固定词表和底座。不要重新跑 `silver` 标注子命令。
新训练输出应放新目录；本次没有重训验证，也不承诺跨 CUDA/硬件逐 bit 还原权重。

单条模型推理可调用下载的 `model-code/scripts/predict.py`；Spatial/Action/Objects 用
`--base-model` 和重复的 `--adapter task=path`，Scene 用 `--scene-model` 和 `--caption`。
**论文四维语义 Repair 的计分入口是这份模型源码中的 `score_matrix.py`；
不能把 GitHub 旧 metric CLI 的默认变体自动当成论文选定方法。**

## 从视频重新推理的外部依赖

需要官方 VBench `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`、GRiT/Detectron2、
Tag2Text、UMT、DINO、CLIP，以及构造/定位用的 MobileSAM、Mask R-CNN、SegFormer。
这些视觉后端预训练模型仍是外部依赖。Dynamic 的 V-JEPA 2.1 backbone 按用户最新要求
额外归档在模型库；这不表示其他视觉依赖也都已包含。

已有 H100 的实际包版本见 [h100-runtime.json](h100-runtime.json)，现有模型字节身份见
[external-assets.json](external-assets.json)。这是现场记录，不是经过空白机器重建验证的容器。
官方模型获取路径在固定 VBench 的 `vbench/utils.py`；其中部分硬编码 HF 官方域名，
因此镜像下载应预先完成、校验权重后再以 offline 方式评分。
SegFormer/MobileSAM 的 revision 与哈希见
[`assets.lock.json`](../../configs/subject-repair/assets.lock.json)。

完整媒体来自每维 `counterfactual/main-table-8d-20260922/` 的全部 tar 分片；
上述路径保留原八维快照名称；Dynamic 媒体使用 `vjepa-expansion450-v1/`。
需要保持 `scoring_base`、采样帧序和对应 CF，不用原始视频替换重新编码的评分基准。
九维的来源、冻结版本和构造入口都已保存，但从空环境恢复完整视觉依赖、重做语义预测、
重新 GPU 打分并逐项对齐全部结果，**本次尚未验收**。

## 其余实验与负结果

[失败、废弃与暂停实验索引](REJECTED_EXPERIMENTS.md) 明确哪些不进入当前复现目标，
并链接原 Markdown 证据；失败不能从历史分母中删除。
四维原项目的报告、曲线和附表另完整保存在
[semantic-models/](semantic-models/README.md)，包括
[完整矩阵](semantic-models/docs/deterministic/matrix-v1/README.md) 与
[消融/泛化](semantic-models/docs/deterministic/ablation-v1/README.md)。
这些历史文档中的 `data/`、`output/` 相对路径是原项目来源路径。
**完整四维矩阵、全部附表及 0.6B 对照尚未像九维主表一样从发布资产独立重放验收。**
其报告存在不等于全套评测输入和所有对照权重都已发布。
此前“每维仅发布一个选定权重”的决定继续适用，不额外发布 0.6B 对照；Dynamic 按用户最新授权发布当前 aligned-v1 与配套冻结 backbone。
