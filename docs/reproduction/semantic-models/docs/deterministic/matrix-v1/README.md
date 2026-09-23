# 四维矩阵表格索引

此目录为首轮共同官方评分公式的冻结结果。当前新结果见 [Spatial repair-v2](../spatial-repair-v2/README.md)、[Action repair-v2](../action-repair-v2/README.md) 和 [Objects repair-v2](../objects-repair-v2/README.md)；本目录旧表保留用于复现和消融。

[研究报告](../../deterministic-experiments-report.md)解释分母、来源、缺失和结果边界。此目录只提交汇总表与身份元数据；视频、图片、预测、缓存和模型保存在忽略的产物目录。

每张表都有同名 `.md`、`.csv`、`.tex` 文件。CSV 保留数值精度，展示列中的区间按四位小数排版。主表 [paper-main.md](paper-main.md)给出全部非恒等变换的三方案原始/变换分数，配合覆盖率阅读。

| 表 | 内容 | 必须注意 |
| --- | --- | --- |
| [A](table-a-parsing.md) | 原始文本的结构解析/F1/有效率 | 元数据参考，不是人工金标；实际接口与额外语义归一化分开 |
| [B](table-b-originals.md) | 原始视频、生成器分表、相对 Origin 的配对差 | 高分不等于更正确 |
| [C](table-c-transforms.md) | 全变换均分、配对差/CI、覆盖/弃权 | 缺失计零；适用/计划数并列 |
| [D](table-d-visibility.md) | 不可见端点、严格隔离移除及全计划下界 | 同一视觉模型多遍共识，human_reviewed=false |
| [E](table-e-scene-classification.md) | Scene 共同二分类与模型三分类 macro-F1 | 两种任务分母不同；test2 只有 5 个来源块 |
| [F](table-f-complete-pairs.md) | 两端完整的配对子集 | 不替代全计划统计 |
| [G](table-g-action-protocol.md) | Action 目标集合、other、不变性 | OOV 只有 1 个唯一输入，无 CI |
| [H](table-h-scene-synonyms.md) | Scene 正确且一致、原生标签一致、复制/常数对照 | 正确且一致采用 supported/not-supported 二分类；test2 同义仅 1 块 |
| [I](table-i-spatial-axes.md) | 水平、垂直镜像分开 | 官方无符号几何不等于方向正确率 |
| [J](table-j-matched-controls.md) | 相同完整视频上的目标/等面积背景对照 | matched/planned 显示选择后的覆盖；不可跨子集比较均值 |
| [K](table-k-metamorphic-checks.md) | 不变性与正基例掉分通过率 | observed drop 不是语义核验；verified 列仅已复核条件 |

表 K 的不变性通过要求两端完整、有非弃权输出且 |Δ|≤0.03。observed complete-pair drop 要求两端完整且 Δ≤−0.03，分母包含全部原始正分项；缺失不算通过。verified drop 进一步要求可见性复核等语义条件，不能把 N/A 当作 0 分。所有来源块 CI 都使用 2,000 次固定 seed bootstrap，少于两块不报。

原始结果目录：`output/deterministic/matrix-v1/`。其中 `tables/tables.pdf` 为完整附录，`tables/paper.pdf` 为紧凑主表，`gallery-final/index.html` 为真实帧案例。可从报告中的本地链接打开；这些大文件和图像不提交 Git。

`manifest.json` 绑定排版所用输入与渲染器 SHA；`provenance.json` 汇总固定 final 模型、权重 SHA、代码/数据身份、完整性、API token/request 账本与未知资源项。执行完整性指所有冻结项均已尝试和保留，不代表所有遮挡都有效或都有人工真值。
