# Dynamic Degree：外部泛化验证与发布

冻结当前论文 `aligned-v1`，新增 LASIESTA 43 片段及 BMC 84 片段，共 **127 片段／16 条源录像**。
它们不是 127 个独立场景；两套标签、时间协议不同，分开报告，不合并准确率。
本轮仅迁移、发布和复算已有结果，不训练、不改权重或统一评分入口，不替换 VBench 450 组主实验。

## 主要结果

反事实为原视频内部纹理的 **8px 局部往返位移**，不是 RGB 噪声或亮度频闪。
每片先平均两个固定种子；编码控制不混入反事实均值。

| 数据／分组 | 片段数 | Origin 原片→反事实 | Repair 原片→反事实 |
| --- | ---: | ---: | ---: |
| LASIESTA 静态 | 16 | 0.000000→0.875000 | 0.125877→0.163079 |
| LASIESTA 运动 | 27 | 0.592593→0.981481 | 0.589715→0.610019 |
| BMC 全部（主分析） | 84 | 0.261905→1.000000 | 0.282374→0.292002 |
| BMC 视觉静止（探索性） | 5 | 0.000000→1.000000 | 0.041793→0.014800 |
| BMC 明显运动（探索性） | 59 | 0.372881→1.000000 | 0.362253→0.369636 |
| BMC 不确定（仍纳入主分析） | 20 | 0.000000→1.000000 | 0.106876→0.132283 |

完整区间、逐片失败、版本、设备和原始审计分别见
[LASIESTA 报告](../counterfactual-reports/dynamic_generalization_lasiesta_20260923.md)、
[BMC 报告](../counterfactual-reports/dynamic_generalization_bmc_20260923.md)。

## 证据边界

- LASIESTA 使用官方对象动静标注，源 BMP 序列的原生时间尚未核实；只证明标准化时间协议下的表现。
- BMC 9 个候选源中 7 个合格；7fps 与非连续时间戳各排除一源。标签是评分前 agent 视觉复核，非官方真值或独立真人盲审；静止仅 5 片段／2 源。
- LASIESTA 静态／运动组平均涨幅分别为 Origin 的 4.25%／5.22%，但 10% 条件的区间均跨零。BMC 总体为 1.30%，总体配对区间在零以下，运动子组仍跨零。
- BMC 37/168 条反事实绝对变分超过 0.1；LASIESTA 静态组 5/32 条增分超过 0.1。不能称为逐片不变。
- 更大规模独立静止集、真实相机运动专门验证、独立真人标签和物理运动强度标定未完成；编码器预训练重叠未知。此前 DEV 失败继续保留。
- [Deep/Causal 长视频试验](../counterfactual-reports/dynamic_generalization_forcing128_20260923.md)保持暂停，不计入完成量。

## 发布内容与复现

HF 数据库新增目录：
[`dimensions/dynamic_degree/generalization/external-v1-20260923/`](https://huggingface.co/datasets/xju-arlab/vbench-repair/tree/e64af55bd59324fb3f5ce934776e091fd70843eb/dimensions/dynamic_degree/generalization/external-v1-20260923)。
其中保存两轮原始逐条分数、配对、汇总、构造／选择清单、配置、复核记录、原审计和下载来源。
原始数字与 JSON/JSONL 字节保持不变，`manifest.json` 提供全部发布成员的 SHA-256。
代码仍在私有 GitHub；**不重新分发两套原始视频、截图、模型权重或训练缓存**。
BMC 数据许可尚未核实，源媒体只从作者渠道获取；LASIESTA 仍遵守原 CC BY-SA 条款，不重新赋予许可。

发布脚本：`scripts/publish_dynamic_generalization.py`；只从明确白名单取轻量证据，
以远端父 revision 保护提交，并逐文件重新下载核验，不用“上传成功”冒充内容验收。
固定 revision 与上传验收记录见[发布记录](../publication/dynamic-generalization-release.json)。
数据 revision：`e64af55bd59324fb3f5ce934776e091fd70843eb`；
模型卡 revision：`368d8342efe1395cbd1e65c24501ed156392fdf5`（只更新 README）。
两份权重与原固定 revision 的远端 SHA-256 完全相同。
实际上传 65 个新成员并更新 2 份数据卡，**67/67 逐文件新下载验收通过**；
新下载副本再次复算通过，最大算术误差 `1.1102230246251565e-16`。
相关 **62 项测试通过**，17 份迁入评估脚本／配置／测试与研究源码逐字节一致；
锁文件、冻结目录与模型实现未变。首次上传后的导入错误已修复并补 2 项测试；
续验同一数据 revision，没有覆盖证据或重新评分。

```bash
hf download xju-arlab/vbench-repair --repo-type dataset \
  --revision e64af55bd59324fb3f5ce934776e091fd70843eb \
  --include 'dimensions/dynamic_degree/generalization/external-v1-20260923/**' \
  --local-dir output/hf-external
.venv/bin/python scripts/verify_dynamic_generalization.py \
  --bundle output/hf-external/dimensions/dynamic_degree/generalization/external-v1-20260923 \
  --output output/external-replay.json
```

复算重新检查输入／分数身份、508 个 Origin 判据及 508 个 Repair sigmoid、127 组编码控制、
配对、均值、MAE、源录像聚类区间及 AUROC。它**不是重新解码媒体或 GPU 推理**；原全媒体审计另保留。
完整 GPU 复跑用两份报告的控制器／下载入口，需显式适配其 H200 资产路径、匹配哈希并使用新输出目录。
这些独立外部评估脚本允许全帧矩形输入适配，不改变统一论文 CLI 已验收的正方形输入合同。
