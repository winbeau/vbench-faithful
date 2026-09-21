# Object/Color 方法审查与引用边界

2026-09-20，Codex 根据源码、冻结输入与实测文件作自动核对；**不是人工标签或视频真值审核**。
本文件与[结果报告](object_color_repair_20260920.md)一起阅读。交付完成表示实现和预注册测量已执行、结果如实报告，不表示所有预期科学关系都成立。

| 命题 | 可引用范围 | 不能由此推出 |
| --- | --- | --- |
| Object 精确成员规则对等义字符串脆弱 | 14 个 test 基底大写对照、13 个有别名基底；同一视频/原 prompt，仅 metadata 变化 | 自然 GRiT 经常产生同义词；语义 LoRA 修复了此 metadata-only 试验 |
| Object Repair 对这些字符串保持不变 | 配对全帧分数差为 0；全部有效请求保留；不在场检测代理上双方中位分 0 | 独立检测器漏检就是物体不在场；所有 COCO 类的视觉检测都可靠 |
| Color Repair 对可见帧比例响应 | 5 个自动合格 test 基底端点降分及五档严格单调达标 | 25 个候选都合格；自然颜色真值/人类偏好更好；两个 dev 基底也达标 |
| Official Color 条件率可能吸收变化 | 有定义档位的中位最大偏离 0，宽 CI [0,0.6]；缺失档位明确保留 | 所有档位完全恒定；CI 含 0 证明严格等效；用缺失视频剔除后抬分 |
| Color 同义控制 | Repair 与 binding+lexical 有定义且为零差 | Official 和 binding-only 同样通过：它们有定义配对 n=0，控制未获验证 |
| Color 绑定/词法/分母 | 各变体分别重放真实 trace；old-rule 与 true Official 数值/空值一致 | trace 重放是独立 Official 测量；单独改 `[2][i]` 就证明最终获益；自然双实例互换已完成 |
| 两独立 LoRA | 同一 Qwen 主干、硬路由、不叠加、仅原 prompt；独立 300 steps；silver 参考一致性分别报告 | 已经证明语义标注精度；已验证其他三维旧 adapter 的效果；在当前原始视频上胜过确定性规则 |
| 自动施工 | 独立定位、向内羽化、区域外像素零改变、无损/replay hash、施工 mask 与评分隔离 | 掩码语义或目标颜色经过人工验证；残余检测为 0 就证明目标完全不可见 |

两个可见性 dev 基底保留，其中一个零效应，没有以 dev 不达标为由改阈值或重新挑 test。
大写/复数是受控类别字符串干预，Color 控制可能出现 `a white cars` 这样的非自然语法；它验证标签形式，不冒充流畅自然 prompt 的语义等价测量。
颜色 caption 无证据为支持值 0，不等于视觉对象颜色错误。任何 runtime failure 都保留为失败/null，不降格成“没检测到”，也不与合法 unsupported/null 混为一谈。

原四维计划中需人工确认的完整自然研究阶段、Color 25 个合格主基底与双实例互换扩展、自然偏好与独立 E0 复现均未交付为已验证结论；本任务以最新用户的 Object/Color 两主族目标和独立自动施工为范围。人工审核只是预留 ≤200 位，当前完成 0，因此报告只使用参考标签一致性，保留 `a basin` / `a sandwich` 两处参考冲突。

## 目标逐项验收证据

| 用户要求 | 当前可检查证据 | 核对结论 |
| --- | --- | --- |
| 两独立包 + 共用 core + 同一 uv lock | 两 `pyproject.toml`/src/tests；四维 CLI/runner/workspace tests；仓库外分别只装一个 metric 的 wheel | 通过；不需要导入其他 metric |
| 官方直调且不改上游 | 两 `official.py` 的锁定 `compute_*`；实际 official-return/input artifacts；执行 manifest 的 fd18b3d clean 和源码 hash | 本次实测调用通过；不等于冻结 E0 |
| Object 全实例/置信度、别名、全帧分母 | `object_class/algorithms.py` 与 audit-models GRiT；1008 真实帧、2842 实例、纯算法/失败测试 | 通过；492 alias-hit 是受控 query 计数 |
| Color 正确关联/整词/全帧/不丢视频 | `color/algorithms.py`、`grit.bind_heads`、置换/歧义/red-substring 测试；42 条保留结果 | 通过；异常与无颜色证据分开 |
| Qwen 同主干独立硬路由/无视觉输入 | `qwen.py`、`prompt_compiler.py`、模型路由合约；同一次真实 compile 的两个 adapter 身份 | 两新 head 已验；其他三维不作效果声明 |
| 最小 nullable JSON，无额外解释字段 | `labels.validate_compilation`、teacher/训练/compile raw；无效输出计数 | 通过；模型状态在 JSON 外 |
| 冻结词表与确定性别名 | 官方 79 ∪ 训练源真实观测 56 = 80；aliases/protocol freeze hash；扩展颜色词表 | hash 校验通过；没有最近类映射 |
| seed→先分组→silver→训练 | 79+85 seeds、source split、API 身份/日期、164+1280 records、424/465 train manifests | 分组/规范化 prompt 泄漏为 0；4 网络失败隔离 |
| ≤200 人工审核位与不报标注精度 | 200 pending rows，完成 0；报告 explicit reference agreement | 通过预留要求，人工精度未声明 |
| 旧规则→确定性→未微调→LoRA | 真 Official + 三次原始 query GRiT 分数；prompt 与视频表分开 | 四级已测；metadata 主族不归功于 LoRA |
| 两主族及反例控制，先冻门槛 | 50 候选全 ledger；Object 16/Color 7 接收，拒收无回填；冻结文件与 run hash 对齐 | 主族实测完成；Color 双边控制缺失，dev 未达标，原样报告 |
| 独立施工/像素与 replay/不复用 mask | `construction.py`、构造声明、35 版本媒体/replay hash、7 份 mask hash、输入 schema | 自动工程检查通过；没有人工真值背书 |
| base 统计、零效应、失败与覆盖 | 10,000 base bootstrap；main/per_base/coverage；每 query null/status 留存 | 已分别报告，不引用 pooled/tie-margin CPA |
| 主表 + 逐基底 + 配置族清单 | 同名报告目录的 CSV/JSON；`configs/four_dimension/object_color_family_manifest.json` | 可从冻结原始结果复算 |
| 完整纯测试 + 真实验证边界 | 676 passed、3 个其他维度模型 skip；uv/build/help/单包安装；两维各 8 CLI parity 行误差 0 | 工程通过；没有本机真实 GPU 或 E0 复现声明 |
| 权重/目录/上游/提交边界 | 权重仍在服务器；冻结目录 git diff 空；本机/远端上游 clean；实现验收时未 commit/push | 遵守；后续提交依用户明确授权，其他任务未提交改动保留 |

原始运行日志和数据在 `output/object_color_20260920/`（被 Git 忽略）；可审阅的模型身份、文件 hash、候选拒收、主表、逐条状态和完整性核对进入 docs/configs。不能只引用任一平均分或 gate=true 而省略样本数、定义域、控制缺失和本审查。
