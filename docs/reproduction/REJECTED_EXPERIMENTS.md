# 失败、废弃、替代与暂停实验索引

用户要求：已否掉、效果不佳的实验不要求作为当前成功方案重新跑，但必须有 Markdown 记录。
本页只作导航和状态说明；原始结果、失败分母和审查意见仍在链接报告中，不改写历史数字。
“暂停/部分完成/未做”与“实测失败”分开记录，也不把保留方法的局限说成整个方法已废弃。

| 范围 / 版本 | 状态与理由 | 当前处理及证据 |
| --- | --- | --- |
| 旧七维 pooled CPA | 不作为总体修复成功结论；平局 margin 饱和及实验族有效性有问题 | 保留逐族/拆半结论；[总报告](../counterfactual-reports/CONSOLIDATED.md)、[P1](../counterfactual-reports/P1_NATURAL_AND_CONTROL_RUNS.md) |
| 旧 Scene、Action、Spatial 干预族 | 审查否定其部分语义前提；高 CPA 不构成成功证据 | 使用后续固定元数据/实际镜像协议；[实验总览](../EXPERIMENT_INDEX.md) |
| Dynamic FPS v1 / v2 | v1 镜像了采样违约；v2 的聚合校正不能证明自然运动更好，P1 自然偏好退化 | 作为历史反例；[Dynamic 审计](../paper/dynamics_degree.md)、[P1](../counterfactual-reports/P1_NATURAL_AND_CONTROL_RUNS.md) |
| Dynamic 早期合成噪声/闪烁 | 用户明确替换为官方原片上的局部空间纹理位移 | 标记 superseded，已归档但不混入当前实验；[替代决定](../plans/2026-09-22-dynamic-static-jitter-protocol-superseded.md) |
| Dynamic CoTracker / NCC / SIFT / 支持区域候选 | 缺位移、身份错配、覆盖不足或未达到涨幅 ≤10% 的要求 | 不作为已完成修复；[完整开发记录](../counterfactual-reports/dynamic_static_jitter.md) |
| Dynamic V-JEPA joint | 自然偏好 20/23 及不变性指标不能消除静止/运动排序反例 | 候选说明保留；按用户决定不上传权重；[记录](../counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-expansion450) |
| Dynamic anchored | 静止约 0.008，但自然偏好 18/23，未达预定 ≥19/23 | 整体验收失败；不上传权重、不晋升默认；[验收](../counterfactual-reports/dynamic_static_jitter.md#dynamic-vjepa-anchored) |
| Motion 早期幅值聚合 / 方向修复 | 自然偏好不佳；旧反事实重评分只到 parity，新默认与旧报告版本不同 | 不算当前八维成功主表；[审计](../paper/motion_smoothness.md)、[P1](../counterfactual-reports/P1_NATURAL_AND_CONTROL_RUNS.md) |
| Subject 隔离裁剪与初期 CLS 候选 | 自然集退化或均值/最坏情况未过门槛；两条好例子不足以证明总体成功 | 保留诊断；[官方扩展](../counterfactual-reports/subject_official_extension_20260920.md)、[隔离开发](../counterfactual-reports/subject_isolation_development.md) |
| Subject 旧 60 条单首帧方案 | 已被 720 候选的 v9 主实验替代，旧主体响应偏弱 | 只作历史；[主实验](../counterfactual-reports/subject_stability_20260920.md) |
| Subject v9 单帧补充 | 用户要求收尾时停在 204/241；部分完成，不是模型失败 | 不当作全量验收；[停止记录](../counterfactual-reports/subject_stability_20260920.md) |
| Background 早期候选 | 开发阶段多种表示/聚合未过冻结门槛 | 保留失败路径；[开发报告](../counterfactual-reports/background_development_20260920.md) |
| Background Caption v2 / union 扩展 | Caption 未证实对现有 repair 非劣，union 仅输入探测；新联合目标暂停 | 不晋升默认；[定位报告](../counterfactual-reports/background_caption_localizer_20260920.md)、[暂停交接](../counterfactual-reports/background_repair_checkpoint_20260920.md) |
| Background 旧 holdout 的自动掩码 | 数值门槛通过，但后续发现语义误选/漏分 | 保留八维冻结数值和局限，不宣称语义构造已全量人工验真；[空掩码审计](../counterfactual-reports/background_empty_masks_20260920.md) |
| Scene 早期 0.319 vs 0.799 等比较 | 标签/口径错误；构造意图不能直接当可见证据真值 | 旧数值作废；使用后续 v8；[阶段审计](semantic-models/docs/STATUS-2026-09-20.md)、[执行审计](semantic-models/docs/evaluation-audit.md) |
| Spatial 早期弱关系和无符号几何 | 文本 LoRA 不会自行修复几何公式；框镜像不是视频镜像 | 保留诊断，当前主表用四方向模型与实际镜像；[方向修复](semantic-models/docs/deterministic/spatial-repair-v2/README.md) |
| Action 旧泄漏训练版本 | 来源审计后移除 113 个相关来源组并重训 v9 | 旧结果不作为当前主模型；[执行审计](semantic-models/docs/evaluation-audit.md) |
| Action 接口 v2 | 长文本/未执行动作的越界改写覆盖正确模型输出 | 默认 v2.1，旧 v2 只保留精确回放；[后验回归](semantic-models/docs/deterministic/ablation-v1/README.md) |
| 0.6B、长文本与未见同义表达 | 多组未过非劣门槛，所有原冻结系统全维 WG-CC=0 | 保留对照，不声称普遍泛化；[消融与泛化](semantic-models/docs/deterministic/ablation-v1/README.md) |
| Objects 相邻帧过滤 | 减少孤立误报同时损失正例，持续幻觉未解决 | 保留工程取舍，不能写成完全修复；[时序修复](semantic-models/docs/deterministic/objects-repair-v2/README.md) |
| Object Class / Color 的 LoRA 视频增益 | deterministic/base/LoRA 在已测原始 query 视频分数相同，无 LoRA 增益证据 | 论文 step 300 按用户要求发布，明确未经 dev 选优；[两维报告](../counterfactual-reports/object_color_repair_20260920.md) |
| Color 自然偏好、完整语义人工审核 | 未完成，不是已有正结果 | 保留 NOT RUN；[审查](../counterfactual-reports/object_color_repair_20260920.review.md) |
| Temporal Style / Overall Consistency | 前者只有 token 截断审计，后者退出本轮范围 | 不算已完成 Repair；[范围说明](../paper/unaudited-dimensions-triage.md) |

更细的训练尝试、教师拒收、版本替代与曲线见
[四维原项目文档快照](semantic-models/README.md)。本次只增加文档归档和可重放输入，
没有为补齐表格重跑已否定方案，也没有删除这些方案的原始证据。
