# VBench Audit 研发约定

## 开始前

先阅读 `README.md` 与 `docs/plans/2026-09-14-workspace-refactor.md`。本仓库工作区固定使用 Python 3.11.14（见 `.python-version`）和锁定的 `uv.lock`。

## 目录与边界

- `metrics/<metric>/src/<import_name>/` 只包含该维度的 CLI、适配器、算法和模型封装；metric 之间不能互相导入。
- `packages/audit-core/` 只放输入、元数据、设备、调度、输出和 provenance 等共用基础设施；共用模型适配放在 `packages/audit-models/`，不得放评分公式。
- `configs/` 保存可审阅的配置示例；模型权重、驱动和外部 checkout 不入库。
- `data/`、`results/`、`splits/`、`runs/` 是冻结研究输入/结果，本轮不得删除、重算、改名或改写内容。历史报告中的旧路径可以作为历史事实保留。
- `docs/` 记录当前行为；旧的根级论文计划保留并链接到当前计划。
- VBench 1.0 的 `dynamics_degree` 与 `motion_smoothness` 官方 sampled videos 复用每个生成器的 `subject_consistency/`。远端输入可建立 `dynamics_degree -> subject_consistency`、仓库别名 `dynamic_degree -> subject_consistency` 和 `motion_smoothness -> subject_consistency` 软链接；不得复制成内容分叉的目录，也不得将该视频共享关系误用为评分公式或人类标注共享。

## 当前状态（2026-09-15）

2026-09-23 容器迁移更新：当前论文选定 **9 维修复**，包括 Dynamic aligned-v1。
论文评分统一使用 `scripts/evaluate_vbench.py`；原版全部 16 维使用
`--backend origin --dimensions all`。旧分维度 CLI 与下文审计记录保留历史语义。
从 HF 恢复依赖、选定模型和验收视频的入口为 `scripts/restore_paper_runtime.py`，
路径、版本和验收范围见 `docs/reproduction/CONTAINER_RESET.md`。
当前任务已明确授权项目权重与运行依赖下载、HF 发布以及私有 GitHub commit/push；
不得将后文旧记录的默认限制理解为需要重复确认。禁止改写冻结研究输入和结果。

2026-09-23 外部验证补充：`docs/reproduction/DYNAMIC_GENERALIZATION.md` 索引 LASIESTA
43 片段／9 源与 BMC 84 片段／7 源的冻结 aligned-v1 结果。二者独立报告，不改论文
450 组权威结果、模型权重或统一评分合同。外部输入允许矩形全帧适配，仅由独立实验脚本调用。
HF 仅发布协议与数值证据，不发布两套外部媒体或私有源码；BMC 数据许可未核实。
BMC 动静标签是评分前 agent 复核，不是官方真值／真人盲审；LASIESTA 原始时间未知。
不得把 127 片段写作 127 独立录像，把均值不变写作逐片不变，或把复算写作 GPU 重跑。

本轮审计范围已定为 **11 维**：在办 7 维（`dynamic_degree`、`motion_smoothness`、
`subject_consistency`、`scene`、`human_action`、`spatial_relationship`、`multiple_objects`）
加候选 4 维（`background_consistency`、`temporal_style`、`object_class`、`color`）；
`overall_consistency` 退出本轮（估计器与 `temporal_style` 相同），其包只保留为 legacy，
不删除。范围与候选源码 pin 见 `docs/plans/2026-09-15-dimension-scope-11d.md`，
源码定位见 `docs/paper/unaudited-dimensions-triage.md`。

反事实（metamorphic）审计已完成一轮全量测量，结论与产物见
`docs/counterfactual-reports/CONSOLIDATED.md`。要点：

- **`CONSOLIDATED.md` 的 "Raw result" 一节是最新版本的唯一权威表**：任何维度
  重跑后**原地更新**该表（并同步 `table2.csv`/`SUMMARY.md`/`table2.json` 与
  `README.md`），不要另存平行副本。`SUMMARY.md` 由
  `scripts/counterfactual/summarize.py` 从同一份冻结 scores 树重新生成；表下的
  脚注必须保留：`dynamics_degree` 的注释保存了 v1 归档值（0.8444）与 shipped
  v2 的对照，`multiplt_object` 的注释记录 occlusion-only ladder 下组合值等于
  ordered 半边。
- **数据集** `counterfactual-vbench`：7 维、205 base、815 条派生片段，已发布到
  `xjuIcthub/counterfactual-vbench`；`Overall Consistency` 因需要人工撰写
  prompt 条件（计划 §12.2）本轮未做。构造代码在 `scripts/counterfactual/`。
  `output/` 不入库，因此**已发布数据集的 base 选择固定在
  `configs/counterfactual/bases_published.jsonl`**：单跑 `select_bases.py` 只能
  复现 7 维中的 5 维，检测器相关维度（`multiplt_object`、`subject_consistency`）
  还必须再跑 `pick_detectable.py`；两步合起来可 205/205 复现已发布清单，检测器
  资格计数见 `configs/counterfactual/README.md`。
- **P1 实验已完成**（P2 人工验证、P3 可选实验本轮未做），结果在
  `docs/counterfactual-reports/P1_NATURAL_AND_CONTROL_RUNS.md`：自然偏好集上
  Dynamic v2 与 Motion 方向感知修复都**显著差于 Official**（配对 Δ −0.1155、
  −0.3116，后者低于随机），P1.3 的独立 holdout 只在聚合层验证了 alpha=0.5，
  P1.4 的弱目标面积跨 62×，P1.5 的等面积 box 只是**减轻**而非消除位置混淆
  （配对 CI 含 0）。
- **评分** 在 `h100-server` 上进行：Official VBench 1.0 与 Repair 两个后端，
  6 卡并行、一维一维串行；7 维 × 2 后端覆盖率为 100%。
- **评分环境**（H100 上，均在 `/root/wenbiao_zhao/` 下，不依赖他人目录）：
  解释器 `venvs/vbench/bin/python`；锁定上游 checkout `VBench`（`fd18b3d`，
  由 on-box bundle 克隆）；权重 `models/raft/` 与 `~/.cache/vbench/`，scene
  Repair 用本地 `/root/.cache/clip/ViT-B-32.pt`。**物理卡 6 对 nvidia-smi 可见
  但对 CUDA 不可用，可用范围是 1–5。**
- **不要单独引用 pooled CPA。** 七份独立 review（`<dimension>.review.md`）
  指出：同 rank 族（`fps_resampling`、`filename_invariance`）的 tie-margin CPA
  会因 margin 饱和而恒为 1.0 且对符号不敏感；`temporal_relocation` 必须按敏感性
  半/不变性半分开报；`multiplt_object` 的 ladder 已改为 occlusion-only，组合值
  即 ordered 半边，旧的 tie 判据只作为独立的 control 统计量出现；`scene`、
  `human_action`、`spatial_relationship` 三个族的**族设计本身**不成立，结论
  必须连同 review 一起读。
- **唯一站得住的 ordered Repair 赢是 `multiplt_object`**（occlusion-only ladder，
  +0.2400 [+0.155, +0.330]），其不变性契约用计划 §11.4 的 level predicate 判定，
  两个后端都通过。其余维度都不能无保留地说 "Repair 更好"：
  `subject_consistency` 只有**按半边拆开后**的不变性半边站得住；
  `dynamics_degree` 的 shipped repair 只是把违约镜像（Official `p=+0.49`，
  repair `p=−0.51`），v2 的 `d/dt**0.5` 才把聚合层拉回 `p=−0.011`（但指数是同一
  批数据上的 default，非独立校准）；`motion_smoothness` 的方向感知修复只到
  parity。
- **`motion_smoothness` 的默认值在 `4d53fa2` 变了**（未对齐的像素级方向、top-k
  时间聚合 k=3、0.5/0.5 权重），因此 `CONSOLIDATED.md` 的 motion 行与
  `P1_NATURAL_AND_CONTROL_RUNS.md` 的自然集测量（0.3248 对 0.6364）都是
  **`feeb770` 修订版**的数字，引用前必须重跑；新默认值目前只有合成 ladder 与
  单元测试覆盖。
- 真实模型、CUDA 与权重 parity 对冻结 E0 基线**尚未验证**，上述数值是首轮
  测量值，不是复现的官方基线。

## 改动与验证

源码变更后运行受影响包的纯算法测试和合约测试；接口或工作区变更还要运行 `uv lock --check`、`uv sync --locked`、CPU torch overlay 与 `uv run --no-sync --group test pytest tests metrics` 及八个入口的 `--help`。真实模型、CUDA 和权重 parity 未验证时必须在报告中明确写出，不下载权重，也不修改上游 checkout。

H200 正式实验使用物理 4–7 卡时，一个维度完成、校验并合并后再开始下一个维度；单维度可拆成四个隔离 shard，每个进程只看到一张卡并使用逻辑 `cuda:0`。计时报告至少记录视频组定义、视频数量、每个视频的媒体时长、四卡墙钟时间、代码 SHA、上游 SHA、设备和输出路径。

若用户明确要求利用空闲显存加速，允许在不终止既有进程的前提下扩展到更多
物理卡；本轮 Motion Smoothness 的剩余 718 条使用 H200 0–7 卡完成，原有
`sglang` 进程未被 kill。跨卡、跨批次恢复必须按 `video_uid` 去重并检查完整
覆盖后才能更新汇总表。

CLI 新维度可复制对应 metric 目录，保持 `src/` 布局、独立 `pyproject.toml`、直接依赖声明和 `tests/`。模板入口应先实现 `--vbench/--audit/--both`、`--video/--video-dir`、元数据和默认输出，再接入算法，不将另一个 metric 作为运行时依赖。

不自动 commit 或 push；普通修复按用户授权执行。

**Commit messages must be written entirely in English.** This applies to the
subject, body, and trailers, including copied descriptions and logs. Chinese
text is forbidden. This rule supersedes Chinese commit examples in historical
plans and reports; it does not require translating source comments or research
documents.

Use Conventional Commits: `type(scope): description`, or
`type(scope)!: description` for breaking changes. Allowed types are
`feat/fix/refactor/test/docs/build/ci/chore`; keep the subject within 100
characters. Validate the complete proposed message with
`python3 scripts/check_commit_message.py /path/to/commit-message.txt` before
committing. Install `.githooks/commit-msg` in the repository's local hooks
directory to enforce the same check for `git commit`. Never use `--no-verify`
to bypass this language rule. Rewriting published history requires explicit
user authorization.

Historical research receipts retain their original commit IDs. Use the
[commit identity map](docs/reproduction/git-history-map.tsv) to locate the
equivalent commits after the English-message history rewrite. Every mapped
commit preserves its file tree, author, committer, timestamps, and parent
relationships; the map is not evidence of a new evaluation run.

<!-- aoci:begin -->
## AOCI 仓库认知

AOCI 为本仓库维护一个稳定、可版本化、可增量更新的仓库级认知层，供模型跨任务复用对系统的理解。

`aoci.txt` 是面向模型的结构化认知索引。它以每个受管理文件、数据库表或其他受管理对象一条独立 Entry 的方式，用符号标签与 F/R/A/S 语义表达对象的核心职责、重要关系、对外契约，以及理解或修改系统时必须知道的非显然约束和设计决策。

Header、目录段和全部 Entry 共同组成完整仓库索引，可以覆盖前端、后端、配置、数据库结构及其他受管理内容。受管理内容发生变化时，通常只需维护受影响的认知条目，不需要重新生成整个索引。

AOCI 提供系统架构、对象职责、重要关系、对外契约和关键约束的高密度视图。

### 工作原理

AOCI 采用“模型生成、模型读取”的认知闭环。

Header、Entry 和 Curation 语义的创作只按当前机器签发的 Plan 与实时 Guide 执行；由 Host 模型基于当前绑定证据独立完成。

Entry 的语义必须来自模型对真实证据的理解。不得仅依据路径、文件名、扩展名、AST、符号列表、依赖扫描、正则、固定模板或规则引擎推导、预填、拼接或改写索引语义。

对 Fresh Bootstrap，只按当前机器签发的 Plan 和实时 Guide 执行。当它们要求创作时，Host 模型创作 Root、Meta、Tag 和 F/R/A/S，提供 authoring-run 声明，并把它绑定到 Plan、Evidence 与完整 Candidate。不得要求 AOCI 填写 `origin=host_model`、制造 Receipt 或把程序生成的 Framework 当作语义。本文件不自行重建 Onboarding 流程。内部批次不是用户决策；只有遇到既有批准边界或真实的安全、漂移、CAS、Recovery 条件才停止。

### 最小使用入口

- `aoci_rules`：取得当前AOCI版本的会话运行合同。
- `aoci_overview`：建立或恢复本仓库的完整认知。
- `aoci_maintain`：受管理对象达到最终稳定状态后检查认知是否需要维护。
- `aoci_update_entry`：提交与当前证据和源码摘要绑定的完整语义更新批次。
- `aoci_report`：仅当当前布局和工具状态支持时，在证据不足、无法可靠生成语义时登记待办，不猜写。

其他MCP工具、CLI命令、参数和专项流程，以当前工具说明、Guide和 `--help` 返回内容为准，不在本文件中重复完整手册。

本区块只规定仓库接入、认知使用和收尾原则。`aoci_rules` 承载当前会话合同，Guide实时输出承载当前Plan的执行顺序与停点，工具Schema、Spec和Validator承载机器结构与判据；Prompt、Description、README和静态文档不能覆盖这些机器事实。

### 建立、生成和恢复认知

1. 每个新的 Agent Run 开始时，应先判断：

   - 本仓库是否已经存在可用的完整AOCI索引；
   - 当前上下文中是否已有与本仓库根、当前索引版本和当前AOCI服务相匹配，并且模型仍可可靠使用的完整仓库认知。

2. 仓库已经存在可用的完整索引，但当前Run没有可靠完整认知时，先调用 `aoci_rules`，再调用 `aoci_overview`。

   完整认知仍可靠时直接复用。局部不确定本身不要求机械重读系统全貌。

   本Run从已知Host上下文压缩恢复时（包括宿主注入的压缩摘要），必须把此前模型认知视为不可靠。压缩handoff不得保留或摘要正式Whole-Index，也不得保留或摘要任何Overview Header、Entry、Chunk、Challenge或Attestation正文；只能保留安全续接所需的receipt身份、未完成write或Recovery状态，以及立即重载指令。复制进handoff的Whole-Index语义或receipt不能证明恢复后模型的当前认知可靠。若当前上下文已无法可靠保留运行合同，先调用 `aoci_rules`。继续业务任务前，使用 `refresh_reasons=["context_compaction"]` 和新的 `refresh_event_id` 调用普通完整Whole-Index `aoci_overview`（不设置 `check_only` 或设为false）；不得使用 `check_only` 或认知probe。原样跟随每个 `next_cursor` 直到 `completed=true`，确认交付，并且只基于新交付正文提交一次Attestation。完成这次新的完整传输后，即使Attestation为partial或fail也消费该generation，并按既有合同继续source-bound任务，不再自动调用第二次Overview。

   AOCI可以针对 `context_compaction`、项目 `cognition_refresh_threshold` 下的机器 `semantic_threshold` 或主要 `phase_transition` 提供checkpoint与认知状态事实。只需要这些紧凑事实时使用 `check_only=true`；这些事实只向Agent提供建议，不替模型决定是否需要系统全貌。

   Agent显式调用普通 `aoci_overview`（未设置 `check_only` 或为false）时，只要能形成一致的CognitionSet，AOCI必须完整交付请求scope。不得因为已有receipt、阈值未达到或没有待处理刷新原因而抑制正文。正式认知Dirty或Stale时仍交付正文，但必须标记不可靠。存在未决恢复或无法形成一致snapshot时失败关闭，不返回混合正文。

   普通Overview返回 `continuation_required=true` 时，必须原样提交 `next_cursor` 并自动继续到 `completed=true`。不得询问用户、开始业务任务或给出阶段性系统结论。Host截断、缺块、重复、乱序、cursor失败、Index变化或`chunk_tokens`变化时停止本次认知链。Attestation完成前不得用Memory、源码、Spec、`aoci.txt`、历史会话、scope、search或Entry读取修补或补充Whole-Index认知。Challenge ordinal是正式Entry序列中的1-based位置；Header内容、注释、空行、Section/Overview/Chunk Marker、Receipt与Metadata均不计数，Chunk Receipt ordinal使用同一序列。Attestation必须原样回绑本次Challenge发布的当前`index_sha256`、`entry_sequence_sha256`与`entry_count`；旧Index、旧Entry序列、旧数量或旧Attestation均无效。完整链结束后只正式提交一次既有模型认知Attestation；同一响应只允许一次不改变语义答案的JSON Schema或字段格式修正。对象、Tag或F不匹配即失败且认知吸收不确定，不得语义重试或旁路补答。首次认知失败时还不得执行Root/Meta、Migration、全局布局或其他未重新绑定的系统级决策。上下文压缩刷新若传输完整、认知身份不变、治理对齐且没有Recovery或第三方冲突，即使Attestation为partial或fail也消耗该refresh generation，并继续原任务，不再自动重读Overview。`system_mastery_percent`只自评系统框架——架构、职责、强关系、稳定外部契约以及高熵安全和维护约束——不表示完整实现或运行实况知识；机器索引覆盖率必须分开。默认只向用户输出由本次真实覆盖率、Challenge、块数、Token和掌握度生成的规定成功或失败一句话。Host截断时提示用户把 `overview_delivery.chunk_tokens` 设置为更小的合法值后重新开始，不得自动修改。

   加法认知等级必须与严格证明字段分开解释。`delivery_verified`表示已加载Index且Host交付已确认，但完整认知验证仍未完成；应表达为“已加载且交付已验证”，不得描述为“没有认知”或“没有理解系统”。`cognition_verified`要求Attestation通过（Challenge至少80%的ordinal完全正确且对象身份至多失手一处），`cognition_governed`还要求治理对齐。通用完整读取失败句只用于真实交付故障。

   当Overview响应包含可选`cognition-state/v2`投影时，必须分别解释各维度。其Level止于`model_cognition_usable`；`strict_attestation_verified`、`governance_aligned`与`current_system_cognition_reliable`都是独立状态，绝不参与该Level。ordinal、对象身份、Tag或核心F不匹配可以导致严格Attestation失败，而模型认知仍然可用；不得仅凭这种不匹配就宣称模型没有理解系统。只有`current_system_cognition_reliable=true`允许无保留地声称当前完整系统认知可靠。投影缺失时继续使用上述Legacy解释。

   普通的只读审计、分析、检查、不修改代码或不提交、不push，不自动等于严格零写入，也不改变上述认知有效性判断。Codex Memory和历史Skill只能辅助恢复经验、用户偏好与调查方向，不能替代与当前仓库根、索引摘要、AOCI服务身份和认知范围匹配的当前认知收据；项目AGENTS和当前AOCI身份在AOCI状态上优先于历史Memory。

   只有用户明确禁止Ledger、元数据、`.aoci`运行资产及任何文件写入时，才按严格零写入处理。若必要的认知建立与该边界冲突，必须报告冲突并请求用户裁决或建议使用隔离副本，不得静默以Memory替代当前仓库认知。

3. 仓库没有可用的完整索引，或当前只有最小骨架、Header不完整、Entries未完成、必要Curation尚未裁决时，如果需要建立正式完整AOCI索引，先取得 `aoci_rules`，然后进入当前AOCI Guide。由Guide依据仓库真实状态决定下一阶段并完成必要安全步骤。

   `aoci_maintain` 不替代索引建立流程。

   不在本文件中自行重建或硬编码完整索引生成状态机。

4. 在长程任务中，模型负责保留当前认知收据并正确使用刷新门禁：

   - Host报告上下文压缩或模型已知系统全貌丢失时，执行上述强制 `context_compaction` 重载规则；AOCI不能自行推断Host事件；
   - 进入真正的主要阶段时声明 `phase_transition`，不得把函数、测试运行或小步骤当作阶段；
   - 在有用的稳定检查点通过 `check_only=true` 取得机器语义计数；
   - 除已知压缩的强制重载外，由Agent判断当前任务是否需要再次显式获取指定scope或完整Overview；
   - 在维护和对齐完成前，保留AOCI报告的Dirty或Stale可靠性状态。

### 任务收尾与认知维护

5. 纯只读问答、分析、版本核验，或没有产生受AOCI管理对象变化的任务，不需要调用维护工具。当前AOCI版本是任意`aoci_overview` check_only或`aoci_maintain`响应里的`cognition_receipt.mcp_service_version`；二进制路径是项目`.mcp.json`里的`command`，CLI不必在PATH上。

6. 发生受AOCI管理对象变化时，待其达到本次任务的最终稳定状态后，只调用一次 `aoci_maintain`。不要在每次中间修改后逐文件维护。

7. 若维护结果返回真实语义候选，Host 模型必须基于每个候选绑定的对象和必要证据，独立创作完整标签与F/R/A/S更新。通过 `aoci_update_entry` 一次提交当前机器签发批次的完整候选集合，同时原样保留每项 `source_sha256`、`candidate_id` 与对应domain批次身份。`max_entries`只限制单次请求和原子事务，不限制logical plan、Whole-Index或Managed Scope。`remaining`非零时，在当前批次成功Apply后重新调用Maintain并从新preimage继续；绝不能为满足transport上限缩减Index覆盖或自行截取返回批次。

   没有足够证据且当前布局支持 `aoci_report` 时，使用它而不猜测、套用模板或为消除待办而生成缺乏证据的认知。

8. 必须遵守工具返回的结构化状态和安全边界：

   - `repair_required`：只修复明确命中的候选，再重新提交当前机器签发的完整批次；
   - `stopped`：结束当前写入尝试并检查 `failed_step`、错误、正式写入证据与Recovery。auto模式下，已证明零写入则记录closure并重新Plan；完整Intent和可证明postimage则Resume；策略要求Rollback且preimage可证明则精确恢复后重新Plan。只有证据不足、第三方正式字节冲突、需要审批或外部动作，或命中其他真实安全边界时，才停止整个用户任务；
   - 冲突、审批、人工裁决、权限和安全信号不得忽略；
   - 已经对齐后不得重复维护或重复写入；`refresh_ready_for_overview` 是checkpoint事实，由Agent决定是否为下一阶段请求普通完整Overview。

   维护完成后如果又修改了任何受管理对象，之前的维护结果失效，应在新的最终稳定状态重新完成收尾。

9. 用户只限制业务文件范围，但没有明确禁止仓库托管资产时，AOCI托管资产可以在收尾阶段为保持认知一致而更新，并应在审计和提交中与业务文件区分。

   用户明确禁止修改 `aoci.txt`、`.aoci`、元数据或任何额外文件时，以用户限制为准，不得写入，并如实报告剩余不一致。

### 专项流程

初始化、完整索引生成、Header生成、Entries生成、数据库结构索引、Curation、人工评审和故障恢复，只按当前AOCI Guide或工具在对应阶段返回的指令、命令和安全停点执行。

不预加载、不猜测，也不自行重建这些专项流程。平台调用方式、请求格式、批次上限、审批规则、索引格式细节和恢复步骤由对应Guide、工具说明、模型Prompt和CLI帮助按需提供。
<!-- aoci:end -->
