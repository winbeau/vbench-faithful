# Object Class / Color：选定 LoRA 的反事实缺项补测

日期：2026-09-23（America/Los_Angeles；远端日志为 2026-09-24 UTC）。
本页补全选定 step-300 LoRA 的 CF 评分，不以旧 metadata Repair 代替 LoRA。
[旧九维表](verified-nine-dimension-replay/main-table.csv)保持原快照与方法身份；
本次两行的数值恰好与旧 metadata Repair 一致，但现在有实际 LoRA 链路的 CF 证据。

## 当前结果

分数为视频均值；每行四格使用同一批完整配对。

| 维度 | 计划 / 四格有效 | Origin（base → CF） | LoRA Repair（base → CF） | 反事实与预期 |
| --- | ---: | --- | --- | --- |
| Object Class | 14 / 14 | 0.9911 → 0.0000 | **0.9911 → 0.9911** | 辅助对象标签大写；对象、原 prompt 与视频不变，应保持分数 |
| Color Consistency | 5 / 3 | 1.0000 → 0.7167 | **0.9583 → 0.1875** | 遮挡目标颜色的视觉证据；应降低支持分数 |

Color 的 Repair 实际完成全部 **5 对 / 10 条**，全量均值为 **0.9750 → 0.1125**。
Origin 的两条 CF 仍为 `dropped_by_official`，保留 null，不填零；故与 Origin
对比使用 3 对，不能将全 5 对 Repair CF 均值接在 3 对的 0.9583 后面。
五个基底不能支撑广泛泛化或总体优越性；自动遮挡掩码并非人工语义金标。

Object Class 的 14 对 Repair 逐对完全不变。干预仅改变 metadata，而该 LoRA
只读取原始 prompt；这是对辅助字段词面依赖的接口鲁棒性验证，**不是 LoRA
在 14 个未见同义 prompt 上的泛化测量**。Color 沿用冻结的目标可见性构造，
不是把物体重新着色；结果不能直接称为“识别颜色互换”的准确率。

## 实际执行与验证

- H200 物理 GPU 4，进程内 `cuda:0`；Object 完成后再运行 Color。代码以
  `61e936d27347457c9c6156d6b6729c4968afdc44` 为基底，新增本页所列补测脚本。
- 两维均重新运行 Origin 和 Repair 的 GRiT 视觉推理：Object 各 28 条全部成功；
  Color Origin 8 条成功、2 条无定义，Repair 10 条全部成功。没有新增训练。
- 文本侧复用选定 LoRA 已冻结的 prompt-only 编译输出，**没有重跑 Qwen 文本推理**。
  配对两端 prompt 完全相同；输入选取只用冻结 test 分组与原 primary 端点，不看分数。
- 全部 38 条 Repair 的原始 GRiT 实例证据经 SHA 核对后重新计算公式，误差为 0。
  新测 19 条 base 与历史 LoRA canonical 分数逐项一致，最大误差为 0。
  新测 38 条 Origin 的分数和状态也与冻结历史记录逐项一致。
- 官方上游为 clean `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。
  Python 3.10.19，torch 2.4.1+cu121，Detectron2 0.6；不是本机 CPU 开发环境。
- Object Origin / Repair 墙钟分别 45.184 / 46.607 秒；Color 为 27.251 / 26.740 秒。
  均为单卡计时，不称为四卡加速结果。逐输入媒体时长、分辨率、帧数在 `families/*/media.json`。
- 首次准备因 `ffprobe` 不在 PATH 中停止，尚未评分；原日志及部分准备文件保留。
  将已有 ffprobe 加入 PATH 后使用新目录 `run-v2`，未覆盖旧结果。
- 59 项相关算法、模型接口、端点选择与共同分母测试通过；两个新增入口 help
  及文档链接检查通过，`git diff --check` 通过。未改默认模型、公式或冻结目录。

| 固定资产 | SHA-256 |
| --- | --- |
| GRiT | `53b6e9b3fd948eac55b574c9b6f94ad0743dff46ba449df7ac2d33009ee92ef1` |
| Object 编译输出 | `1476b75427e0ba80f8e2430514599a4d20194ac99af86b8e18b488d152609646` |
| Color 编译输出 | `adc01664745bc1279f23a326f9fce545d3d921aae3c030f4e9b530b4e308255e` |

## 证据与复现

[逐对分数与独立复算](object-color-lora-cf-20260923/verification.json)记录 19 对、缺失原因、
原始结果文件哈希及 canonical parity。远端完整产物：
`H200-target-server:/data/chenjiayu/wenbiao_zhao/object-color-lora-cf-20260923/run-v2/`。
本机镜像：`output/object-color-lora-cf-20260923/`，包含 scores、evidence、compiled、
families 元数据和 execution；未将视频、模型或大体积输出纳入 Git。

评分使用[冻结端点补测脚本](../../scripts/complete_object_color_lora_cf.py)：
传入 `--source`（原 object/color 实验根）、新 `--output`、`--upstream`、
`--checkpoint` 和 `--dimension object_class` 或 `color`。沿用
[原视觉环境的 PYTHONPATH](../object-color-repair.md#复现命令)，隔离一张 CUDA 卡；
脚本核对元数据、编译输出及每条媒体的哈希，不重建构造或重新选择样本。

```bash
uv run --no-sync python -m scripts.analyze_object_color_lora_cf \
  --run output/object-color-lora-cf-20260923 \
  --historical-dimensions output/reproduction-nine-from-hf/bundle/dimensions \
  --output output/object-color-lora-cf-20260923/verification-new.json
```

Multiple Objects 是另一项实验，其旧遮挡结果的问题与待确认修正方案见
[反事实审计](MULTIPLE_OBJECTS_CF_AUDIT.md)；本次补测没有替它制造新结果。
