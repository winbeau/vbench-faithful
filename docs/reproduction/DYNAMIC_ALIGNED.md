# Dynamic Degree：当前九维论文版本

当前选用 **aligned-v1**，不是早期 joint 或 anchored。用户接受 0.6 档尺度后冻结
模型完成 450 组评测，并明确授权上传当前模型和配套 backbone。
首次迁入时原始研究 HEAD 为 `fdf4890`；新增研究当时未提交，按逐文件哈希迁入私有库，
见 [源码快照](../publication/dynamic-source-snapshot.json)。论文修复范围为九维，
研发审计的十一维范围不因这次发布而改变。
后续 `0cd494d` 新增 [Dynamic 方法归档](../paper/dynamic-degree-current.md) 已同步，
四份 Dynamic 核心源码与研究工作树逐字节相同，评分实现未发生变化。

## 模型与数据

公开模型库 [Dynamic 目录](https://huggingface.co/xju-arlab/vbench-model/tree/main/dynamic_degree)：

| 文件 | 身份与选用依据 |
| --- | --- |
| `aligned.pt` | 51,393 参数、7 张量、207,940 字节；固定第 300 步，用户接受的论文版本，未经 dev 选优 |
| `backbone/vjepa2_1_vitb_dist_vitG_384.pt` | 原始 V-JEPA 2.1 ViT-B，1,664,223,428 字节；使用 `ema_encoder`，不训练 backbone |
| `selection.json`、`configs/`、`training/`、`evaluation/` | 两份权重 SHA-256、初始化链、训练曲线与 450 组冻结结果 |

头 SHA-256：`6dcfbdce2e604cdd669e54015fee34e034c8004726d06c46611532f451ca8a53`。
Backbone SHA-256：`848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`。
模型固定 revision：`5fe53c4c7eb1a8a4fcd5b6e22f748da1d073a78e`。
Backbone 是配套预训练编码器，不把它称为第二个训练好的 Dynamic Repair 评分头。
第三方来源及版权声明随权重保留；Dynamic 工程源码已集成本仓库，选定训练与构造入口见 [训练指南](../training.md)。

数据固定 revision：`8f0ae5b29480695dcf07a0474f2dce21da1ec339`。
新增 26 个文件（约 2.05 MB），包括 `dimensions/dynamic_degree/training/aligned-v1/`
及 `reproduction/aligned450-v1/evidence.tar.gz`。后者有 34 个成员，含 1800 条新头评分、
1800 条锁定 Origin 评分、输入清单、逐源配对、汇总和核验凭据。
450 原片、900 CF 与 450 编码控制继续复用已发布的 `counterfactual/vjepa-expansion450-v1/`，
没有重编码媒体或用新的变换覆盖已有字节。

## 复算与推理

按 [总入口](README.md) 运行 `prepare_reproduction.py` 和 `reproduce_main_table.py`，
即可从 HF 轻量包复算九行。新增 Dynamic 会检查输入身份和完整组合、重新计算 Origin
光流判据与新头 sigmoid，逐条对齐原配对，再每源平均两个种子，最后等权汇总 450 源。
450 编码控制单独核对，不混入反事实均分；历史构造警示不按分数剔除。

下载推理权重时，在准备命令中加 `--include-models`。Dynamic 资产将位于
`output/reproduction/models/dynamic_degree/`；其余六个 adapter 保持原目录。
另准备独立、干净的上游 checkout：

```bash
git clone https://github.com/facebookresearch/vjepa2.git output/vjepa2
git -C output/vjepa2 checkout --detach 204698b45b3712590f06245fbfba32d3be539812
CUDA_VISIBLE_DEVICES=1 scripts/h100_python.sh scripts/score_dynamic_aligned.py \
  --video /absolute/path/to/native-video.mp4 \
  --model-dir output/reproduction/models/dynamic_degree \
  --upstream output/vjepa2 \
  --output output/dynamic-aligned-score.json
```

要求原生 16 帧、8 FPS、正方形 RGB、2 秒；没有自动裁切、补帧或改变映射。
已有输出会被拒绝覆盖。加载权重必须严格匹配；脚本记录媒体、解码像素、tokens、
编码器与评分头的哈希。`dynamic_degree.cli` 的历史默认 Repair 不自动等于这个论文方法。

H100 已从模型库下载并校验两份完整权重，在原生官方 ModelScope 单片上实际推理，
得分 **0.6755738467711915**，latent、tokens、像素和媒体哈希与 H200 冻结记录完全一致。
[机器验收](h100-dynamic-aligned/verification.json) 明确这是单片 parity，并非 450 组全量 GPU 重跑。
H100 已持久保存模型在 `/root/wenbiao_zhao/models/dynamic-degree-aligned-v1/dynamic_degree`，
干净上游在同级 `upstream-vjepa2/`；使用这两个绝对路径即可运行上面的推理命令。

## 结果与限制

同一冻结权重另完成 [LASIESTA／BMC 外部泛化验证](DYNAMIC_GENERALIZATION.md)：
127 片段来自 16 条源录像，逐条记录和协议独立发布。不是重新训练或替换下列主结果；
外部标签／时间局限及个别大幅变分继续保留。

450 原片 / 900 CF：Origin **0.680000 → 0.857778**；当前 Repair
**0.629251 → 0.623035**，变化 −0.006216，MAE 0.021731；自然有序偏好 144/171。
原片与两个 CF 各自正确总数相同，不表示每对排序都不变。
原 DEV17/23 和严格尺度门槛失败保留；最大单条降分 0.377932，不能说逐片不变。
测试 prompt 曾用于此前评估，不标成全新独立留出；没有物理运动强度标定。
详细证据见 [研究报告](../counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-aligned450)。

从头训练沿 **probe → anchored → aligned** 初始化链进行。初始两个阶段是生成
当前权重的必要训练依赖，保留配置、训练入口和数据清单，不以旧版本失败为由删除。
按 `configs/dynamic-static-jitter/README.md` 的阶段命令恢复父阶段输出；缓存可重新编码。
当前发布头足够推理，但不是带优化器状态的断点续训包。本次未重做该训练链，
也未从空环境完成所有九维 GPU 评分、六个 LoRA 重训和全部附表/消融验收。
