# Object Class / Color 修复与复现

2026-09-20 用户目标的两维实现、独立 LoRA 训练及实测已完成。工程验收通过；科学结论有明确边界：
Color test 只有 5 个合格基底，Official 的同义控制没有有效配对，人工标签审核为 0。
主表与引用限制见[实验报告](counterfactual-reports/object_color_repair_20260920.md)和
[方法审查](counterfactual-reports/object_color_repair_20260920.review.md)，不能统称“Repair 更好”。

## 包、接口与职责

- `metrics/object-class/`、`metrics/color/` 各自有 `pyproject.toml`、`src/`、`tests/`，纳入同一个 uv workspace/lock，互不 import。
- 两者均复用 audit-core 的 CLI、输入、元数据、GPU 调度、输出与 provenance，统一支持 `--vbench/--audit/--both`、`--video/--video-dir`。默认输出仍为 `output/<metric>/<backend>/<run-id>/`。
- `--vbench` 直接调用干净的 `Vchitect/VBench@fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490` 的 `compute_object_class` / `compute_color`；原始 prompt/aux 不经修复。上游漏返的视频保留 null 行；上游整批异常保留失败行，不伪造分数。
- `--audit` 默认确定性 repair。共享 `audit-models/{grit,labels,prompt_compiler,qwen}.py` 管模型与标签接口；命中规则、颜色谓词、分母和聚合在各 metric 的 `algorithms.py`。
- GRiT 两路原始实例、boxes、confidence、顺序和关联均保留。Color 先核验同序 box，再按唯一 IoU ≥ 0.9 绑定；歧义 abstain。`[2][0]` 是返回序列首实例，未经排序证据不称“最高置信度实例”。
- Object 全采样帧分母，分别记录 `exact_hit`、`exact_miss_alias_hit`、`no_detection`、`other_class_detected`、`runtime_failure`。Color 整词颜色、显式别名及全帧分母；缺颜色证据为支持值 0，不能当视觉真值。crimson/navy/maroon 不折成 red/blue。
- 推理错误与合法零证据分开：错误的视频 score=null；条件分母为空、文本 query 不支持也保留状态与行。部分覆盖时主 aggregate=null，已评分子集均值仅作诊断。语义编译合法的 null 不冒充“检测器不支持”。

模型 B 使用同一 Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`，Object/Color 各自独立初始化 LoRA，按外部维度硬路由，不叠加。只输入原始 prompt，输出分别为 `{"object":...}` 与 `{"object":...,"color":...}`，字段可 null；状态、置信度、解释不属于生成 schema。路由器可接受既有加载模型，保留五维 route 名；本次真实共享加载验证只覆盖这两个新 adapter，未重测另外三维 head。

语义编译与视觉评分分进程：Qwen 一次加载主干及两个 adapter，产生按原始 prompt 精确索引的冻结 JSON；视觉 CLI 读取该 JSON，不加载第二份 Qwen，也不向 Qwen 传检测结果。未缓存 prompt 返回明确失败。默认 `deterministic`，选择 `base/lora` 时须配置对应 compilation 路径；metadata 模式仅用于隔离词法反事实。

## 实际验收

| 项目 | 实际证据 |
| --- | --- |
| 全量纯算法/合约测试 | `uv run --no-sync --group test pytest tests metrics -q -rs`：**676 passed, 3 skipped**；3 个 skip 是其他维度的 CUDA parity 条件 |
| 工作区 | `uv lock --check`、`uv sync --locked --group test`、README CPU torch overlay 通过；12 个入口 help 通过 |
| 独立构建/安装 | core、models、两 metric 的 wheel/sdist 通过；两个仓库外临时 venv 各仅安装一个 metric，`python -I` import/help、显式输出、缺配置两行 null 通过 |
| Official 与真实 trace 对照 | Object 63/63 最大误差 0；Color 42/42 空值一致，33 个有效值最大误差 0；重放不冒充第二次 Official |
| 真实四卡 CLI | 两维各 4 视频 × 2 后端，8/8 结果与公共 batch 一致；四个 worker 各只见一张物理卡，逻辑 `cuda:0`，UID 无重复 |
| 数据与施工 | 50 候选全部入 ledger；Object 接收 16、Color 接收 7，无补选；105 请求/51 实际视频的字节 hash 全部核对，媒体均 2 秒 |
| LoRA 与四级消融 | 两个 adapter 各 300 steps；同主干 base/LoRA 编译完成；原始 query 的四级真实 GRiT 视频评分已完成 |

验证原始日志在 `output/object_color_20260920/verification/`，其中 `isolated-wheels.json` 记录单包安装位置和返回码。
独立安装另修正了 core provenance：找不到 workspace 时仍可用显式 `--output` 写结果，`code_sha=null` 并标明只记录已安装 core 源码；不会虚构 checkout SHA。

## 资产、代码与数据位置

本机 Python 3.11.14 负责准备、测试、silver 请求和报告；本机没有执行真实视觉模型评分。
H100 的 `/root/wenbiao_zhao/venvs/vbench/bin/python` 为 Python 3.10.20、torch 2.5.1+cu121、transformers 4.33.2、Detectron2 0.6。评分用物理卡 5，四卡验收用 1/3/4/5；各维串行。
RTX4090 使用既有 `/data1/wenbiao_zhao/vbench-prompts-compile/.venv/bin/python`，torch 2.7.1+cu126、transformers 4.52.4、PEFT 0.15.2、TRL 0.19.1；物理卡 5 训练两个 adapter，未替换 H100 视觉环境。

- H100 代码快照：`/root/wenbiao_zhao/vbench-audit/output/object_color_20260920/code`；同级任务 output 保留视频、施工 masks、原始实例和 scores。其外层旧 checkout SHA 不能冒充实验运行时的代码版本；run 记录源文件哈希，CLI 另记录 core/models/metric 源码哈希。
- H100 输入只读提取自现有 `datasets/vbench-1.0-human-preference/archives/{lavie,modelscope,videocrafter-09}.zip`，新文件仅进入本任务 output。按 source prompt hash 固定生成器，选最低可用 seed。
- GRiT：`/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth`，SHA-256 `53b6e9b3fd948eac55b574c9b6f94ad0743dff46ba449df7ac2d33009ee92ef1`；BERT tokenizer 仅使用本地 cache。
- 施工资产：`/root/wenbiao_zhao/models/subject-repair/` 下既有 Mask R-CNN、SegFormer-B0、MobileSAM。Mask R-CNN 独立于 GRiT；施工 masks 不进入评分接口。所有模型文件 hash 在交付的 `execution_manifest.json`。
- Qwen：RTX4090 的 `/data1/wenbiao_zhao/models/Qwen3-8B`；五个权重 shard 与 H100 已有指定 revision 相同。独立 adapters 保留在 `/data1/wenbiao_zhao/vbench-audit-object-color-20260920/output/adapters/{object_class,color}`，权重未下载到仓库。
- 本地结果镜像：`output/object_color_20260920/`，不含模型权重或完整施工媒体；可审阅主表、逐基底表、计时和环境身份已放入 `docs/counterfactual-reports/object_color_repair_20260920/`。

两维视觉评分仅需 GRiT，不需要 CLIP、ViCLIP、MUSIQ。资产缺失不下载，模型后端输出失败；纯算法、mock/合约、help 与构建仍能独立验收。真实 CUDA、权重加载和本任务 parity 已验；独立冻结 E0、人工偏好、人工 mask 正确率均未验。

## 复现命令

下列研究构造/训练命令拒绝覆盖已有冻结输出。复算报告可直接运行；要重新测量须使用新的 output 根并保持冻结配置，不能删旧结果重新挑基底。

本机，仓库根目录：

```bash
uv run --no-sync python scripts/object_color.py prepare
uv run --no-sync python scripts/object_color.py freeze-vocabulary
# 已有 silver 时会复用记录；不重复请求或伪造人工审核。
uv run --no-sync python -m scripts.object_color_semantics silver
uv run --no-sync python -m scripts.object_color_report --dimension object_class --output output/object_color_20260920
uv run --no-sync python -m scripts.object_color_report --dimension color --output output/object_color_20260920
uv run --no-sync python -m scripts.object_color_delivery
uv run --no-sync --group test pytest tests metrics
uv lock --check
```

H100，进入本任务 code 快照，以下变量全部是本任务路径：

```bash
export OC_ROOT=/root/wenbiao_zhao/vbench-audit/output/object_color_20260920
export OC_PY=/root/wenbiao_zhao/venvs/vbench/bin/python
export OC_UPSTREAM=/root/wenbiao_zhao/VBench
export OC_GRIT=/root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth
export OC_MODELS=/root/wenbiao_zhao/models/subject-repair
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.:packages/audit-core/src:packages/audit-models/src:metrics/object-class/src:metrics/color/src
# 从现有归档提取到新 OC_ROOT，再做固定的训练组观测；不修改原归档。
"$OC_PY" -B -m scripts.object_color extract --output "$OC_ROOT" --upstream "$OC_UPSTREAM" \
  --archives /root/wenbiao_zhao/datasets/vbench-1.0-human-preference/archives
CUDA_VISIBLE_DEVICES=5 "$OC_PY" -B -m scripts.object_color observe --output "$OC_ROOT" \
  --upstream "$OC_UPSTREAM" --checkpoint "$OC_GRIT"
"$OC_PY" -B -m scripts.object_color freeze-vocabulary --output "$OC_ROOT"
oc() {
  CUDA_VISIBLE_DEVICES=5 "$OC_PY" -B -m scripts.object_color_experiments "$@" \
    --output "$OC_ROOT" --upstream "$OC_UPSTREAM" --checkpoint "$OC_GRIT"
}
# 已完成的构造/评分命令；重测需新 OC_ROOT 并先 extract/复制冻结输入声明。
oc locate --dimension object_class --locator "$OC_MODELS/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth" \
  --locator-sha256 bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0
oc prepare-object --dimension object_class
oc score --dimension object_class --variant official
oc score --dimension object_class --variant repair
# 先完成并校验 Object，再启动 Color。
oc locate --dimension color --locator "$OC_MODELS/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth" \
  --locator-sha256 bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0
oc prepare-color --dimension color --locator "$OC_MODELS/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth" \
  --locator-sha256 bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0 \
  --segformer "$OC_MODELS/segformer-b0-ade20k" --sam-source "$OC_MODELS/MobileSAM-f706ad9" \
  --sam-checkpoint "$OC_MODELS/mobile_sam.pt"
oc score --dimension color --variant official
oc score --dimension color --variant repair
"$OC_PY" -B -m scripts.object_color_report --dimension color --output "$OC_ROOT" --replay-ablations
# 每维 original-query 视频消融，deterministic/base/lora 三者相同视觉模型、相同视频。
for dimension in object_class color; do
  for compiler in deterministic base lora; do
    oc score --dimension "$dimension" --variant repair --compiler "$compiler" --canonical-only
  done
done
# 已测四卡公共 CLI（每次一个维度），parity 子命令自己调用统一入口。
CUDA_VISIBLE_DEVICES=1,3,4,5 "$OC_PY" -B -m scripts.object_color_experiments parity \
  --dimension object_class --output "$OC_ROOT" --upstream "$OC_UPSTREAM" --checkpoint "$OC_GRIT"
# color 同一命令替换 --dimension。只读补录全部媒体时长与文件 hash：
"$OC_PY" -B -m scripts.object_color_inventory --root "$OC_ROOT" --upstream "$OC_UPSTREAM" \
  --output "$OC_ROOT/inventory/execution.json"
```

`prepare-color` 使用三个既有施工资产路径；冻结协议与具体 hash 见 `configs/four_dimension/object_color_construction.json`、交付 `execution_manifest.json`。`observe` 只选固定 24 个训练源，之后才 `freeze-vocabulary`，不能用 test 观测扩词表。复算使用已存在的冻结记录；H100 上新测量完成后，复制其 metadata、scores、evidence、inventory 到本机同任务 output，保留原路径与 hash，再执行报告命令。

RTX4090，进入其任务 code 根，在已复制且 hash 一致的 frozen config、semantics 下执行：

```bash
export OC_QPY=/data1/wenbiao_zhao/vbench-prompts-compile/.venv/bin/python
export OC_QROOT=/data1/wenbiao_zhao/vbench-audit-object-color-20260920/output
export OC_QBASE=/data1/wenbiao_zhao/models/Qwen3-8B
export PYTHONPATH=.:packages/audit-core/src:packages/audit-models/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
for dimension in object_class color; do
  CUDA_VISIBLE_DEVICES=5 "$OC_QPY" -B -m scripts.object_color_semantics train \
    --dimension "$dimension" --base "$OC_QBASE" --output "$OC_QROOT"
done
CUDA_VISIBLE_DEVICES=5 "$OC_QPY" -B -m scripts.object_color_semantics compile \
  --base "$OC_QBASE" --output "$OC_QROOT" --extra-prompts "$OC_QROOT/extra_prompts.json"
```

`extra_prompts.json` 是冻结两维 `families/*/metadata.json` 的原始 prompt 去重列表，结构为维度名 → 字符串列表，不含视觉或标签证据。将编译的四份 JSON 原样复制到 H100 `OC_ROOT/compiled/`，记录 hash 后再进行 `base/lora` 视频评分。公共 CLI 的配置示例为 `configs/four_dimension/object-color.h100.example.toml`；LoRA 使用 `[prompt_compiler] kind="lora"` 及 `path=".../compiled/<dimension>-lora.json"`。

## 交付与范围核对

别名、80 类词表、79+85 种子、协议、观测身份、50 候选族清单在 `configs/four_dimension/object_color_*.json`；主表、105 条逐 query/base 记录、语义参考一致性、四级视频分数与逐 base 表、覆盖率、逐视频时长、真实计时、模型身份均由 `scripts.object_color_delivery` 生成。

原四维计划的 Background/Temporal 和 Color 双实例自然互换扩展不在这个用户目标的实测范围；未把它们写成已完成。旧七维 `CONSOLIDATED.md` 未重跑，保持历史唯一主表；此报告是新增两维独立命题的结果，尚不并入旧 pooled CPA。未下载权重、未改上游、未修改冻结目录。实现验收时未 commit/push；后续按用户明确授权单独提交 Object/Color 及所需共享设施，保留其他任务未提交改动。
