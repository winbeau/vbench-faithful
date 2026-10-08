# 初始化验证记录

## 本地

- Python 3.11.14；uv 0.9.17。
- uv lock：成功，解析67个包（包含训练extra的依赖闭包）。
- uv sync --locked：成功，基础开发环境未安装torch。
- uv run --no-sync pytest：5 passed。
- vbench-prompts --help：成功；CLI仅展示schema。
- git diff --check：通过。

## 远端预检查

- host：rtx4090；user：luxliang。
- nvidia-smi 报告8张 NVIDIA GeForce RTX 4090，每卡49140 MiB；driver 580.105.08。
- 预检查时各卡有既存显存占用；不终止进程，不承诺空闲卡长期可用。
- 原全局uv为0.11.14，已保留，项目专用0.9.17位于 ~/wenbiao_zhao/tools/uv-0.9.17/uv。
- 私有GitHub remote通过SSH，只读deploy key隔离在仓库之外。

## 远端同步

已通过SSH clone及两次 git pull --ff-only。功能验收代码SHA：d899f9efca843c3c10d6f6730d016aed5b24a213；后续文档验收提交不改代码/依赖。

- 项目 .venv：Python 3.11.14。
- 专用uv：0.9.17；uv lock --check通过。
- uv sync --locked --extra train：成功；首次准备65个安装包约13分钟（网络下载为主）。
- pytest：5 passed；CLI --help通过。
- TRL SFTTrainer/SFTConfig、PEFT LoraConfig、Transformers分类模型入口import成功，completion_only_loss字段存在。
- 精确版本：torch 2.7.1+cu126 / transformers 4.52.4 / trl 0.19.1 / peft 0.15.2 / accelerate 1.7.0 / datasets 3.6.0。
- HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1，CUDA_VISIBLE_DEVICES=5下执行2×2矩阵乘法及反传，成功；这是逻辑cuda:0的微型算子检查，不是模型训练，也没有测试其他7卡。
- 远端git工作区clean。
- uv.lock SHA256：0ebe4906cb87ce69e4281ee1e54eb684f697e2de48c557bf4e3ab8894fde94ba。

复验：

```bash
../tools/uv-0.9.17/uv lock --check
../tools/uv-0.9.17/uv sync --locked --extra train
.venv/bin/python -m pytest
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/check_training_stack.py
# 仅在已确认可用GPU时追加 CUDA_VISIBLE_DEVICES=<卡号> 和 --cuda
```

初始化期间已解决：旧gh不支持deploy-key子命令，改用GitHub keys API注册同一只读公钥；uv wheel可执行文件位于bin/uv，已纠正专用软链接。Python安装提示已有非uv管理的python3.11入口，未覆盖；项目正确使用托管3.11.14。

## 第二轮：清洗、teacher 试标与冒烟闭环（2026-09-19）

- 新增实现：`records.py`（契约/门禁）、`sources.py`（只读原料加载）、`teacher.py`（DeepSeek 客户端+预算账本）、`pilot.py`（试标解析）、`metrics.py`、`training.py`、`inference.py`、`provenance.py` 与 `scripts/{prepare_data,build_smoke_mix,run_teacher_pilot,make_tiny_model,train_adapter,train_scene,evaluate_smoke}.py`。
- 测试：本地 87 passed（含训练栈测试）；远端 75 passed / 12 skipped（跳过项为依赖本地 VG raw 快照的数据测试，远端未搬运该 75MB ZIP）。
- 清洗构建：`data/processed/local-0001`（spatial 379+21、objects 400、action 400、MovieGen 1,525 无标签、SNLI 本地缺 pyarrow 记为 unavailable）；工程 fixture `data/smoke/local-0001`（12/12/12/90）。
- teacher：19 次 `deepseek-flash` 请求用满授权（累计 20/20），15 候选 + 4 quarantine，详见[试标报告](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/teacher-pilot-report.md)。
- 训练冒烟：Qwen3-0.6B（revision `c1899de289a04d12100db370d81485cdf75e47ca`）四任务各 100 步，远端 4090 物理卡 2 与本地 4060 各跑一遍；底座冻结、adapter 更新、completion mask 正确，耗时 67–108 s/任务，峰值显存约 3.9–4.9 GB。
- 评测冒烟：解析任务 held-out 切片 F1 0.91–1.00（对照未微调底座 0.00–0.87）；Scene 严格单标签 5/12 可解析，加"标签后即停"约束 12/12。详见[冒烟报告](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/smoke-report.md)与[运行手册](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/runbook.md)。
- 未改依赖锁；未 commit/push；未改动相邻 audit 仓库冻结文件。

## 未验证/未执行

未启动正式训练；未下载或指定 8B 级正式底座（注意锁定版 transformers 4.52.4 不支持 `qwen3_5`）；无人工金标准、无真实 Tag2Text 场景数据；201+ 词长度桶为空；未做多卡与长序列显存/吞吐测量；未验证最终 VBench 收益。

## 2026-09-20：Scene 标注恢复门禁

- 新增逐请求响应持久化、按观测去重、视觉冲突保留、并发预算锁、401/403 停止批次。
- `uv lock --check`、完整 `uv run --no-sync pytest -q`（110 passed）、CLI 与标注入口 help、`git diff --check` 通过。
- 真实纯文本试跑：1/719 观测，两次 DeepSeek 请求成功；分段返回码 2 正确表示仍有 718 项未完成。
- 来源泄漏、历史重复与资源修复见 [评测审计](evaluation-audit.md)。GPU 重训、官方 parity 与最终实验矩阵尚未完成。

## 2026-09-20：四方向训练契约与来源划分

- 115 passed：新增四方向目标投影、保留 unsupported 空结果、adapter/checkpoint 契约读取、越界输出拒绝、Scene 双端来源隔离与 SSL 断连异常回归。
- Spatial v8：2,679 train / 312 dev；固定 600 步配置。Scene 300 步配置已冻结，数据待证据标注完成。
- 真实标注：chiyi 首条视觉任务两次请求成功，test2 全批已启动；DeepSeek 在 103 条处 SSL 中断，确认 exit 1 后已修复并从持久化响应恢复，未重跑前 103 条。
- H100 新 checkout `/root/wenbiao_zhao/vbench-prompts-compile-git` 使用 Git bundle + `pull --ff-only`；Python 3.11.14 / uv 0.9.17；旧副本与数据保留。rtx4090 的既有删除保存在 `output/deploy-backups/20260920/`，pull 后仍保留该删除。

## 2026-09-20：官方规则复算与 UMT 缓存入口

- 最新完整测试 121 passed；缓存/规则检查 CLI help 与 diff check 通过。
- 源码规则比较 2,000 组几何 / 6,000 帧及 Scene、Objects 边界用例通过；保留官方 Spatial 忽略方向符号及允许同名框配对的行为。
- Scene evidence 719/719 完成，1,225 次预记账（其中 1 次 SSL 失败），供应商返回 298,684 输入 / 7,584 输出 tokens；金额未提供。
- Spatial 训练进程已在用户指定 GPU 5 启动，任务 `vpc-spatial-v8`；Scene 同卡排队，不与其并行占卡。真实训练完成状态以 exit code / run_summary 为准。

## 最终表格与身份汇总入口（执行验收前）

- `render_matrix_tables.py` 增加分轴 Spatial、同批完整 Objects 背景对照、Scene 同义及条件通过率表，分类分母/来源块与单输入 OOV 均明确列出。
- `select_evidence_cases.py` 每个镜像轴各取字典序首个检测几何代理例；案例不替代主统计，不声称人工方向真值。
- 新 `summarize_matrix_provenance.py` 绑定训练、推理权重、固定最后 checkpoint、API 用途账本与评分/表格输入哈希；拒绝执行未完整或身份不一致的产物。
- 完整测试 150 passed；CLI help、lock check、diff check 通过。真实最终矩阵/PDF/资源清单将在全部 Objects 分片结束后另记验收，当前不以脚本存在冒充完成。

## Objects 最终合并失败请求回归

- 8 个复核分片全部完成；首轮合并日志保留。修复只涉及资源 journal 汇总，区分有 response 与 error 的条目，保留失败和重试计数，并绑定 journal SHA。
- 新回归包含同一请求先失败后成功及另一个失败，不因缺少 response 崩溃、不重复计成功 tokens。完整测试 151 passed，合并 CLI help 与 diff check 通过；不重跑视觉 API 或模型推理。

## 四维矩阵最终验收

- H100 `final-matrix-v1-retry2` 的 merge/score/controls/outer 全部 exit 0。29,660 清单项、88,980 配对记录、81 汇总；所有原缓存/变换/预测实际记录完整性检查通过，13 个输入 SHA 在本地逐项复验一致。
- Objects 980 个复核 ID 无重复；505 模型标注、475 几何不完整，275 不可见、142 严格隔离；全 12,740 个条件在三方案中的原/变换分数与覆盖逐项相同。背景匹配/遮挡外像素/level 0 检查通过。
- Scene/Action/Spatial 的负结果与不适用项均保留；Action OOV 协议统计实际 n=1、CI=null。没有以未执行或几何失败冒充正确拒绝。
- 151 passed；CLI help / lock / diff check 通过。最终 PDF 两遍编译：主表 2 页、完整附录 14 页，无溢出或警告；已查看主表与可见性表图像。11 个实际案例导出成功，已查看 Objects 遮挡残留、检测器误报和隔离控制帧。
- 四个 final 与最后 checkpoint 权重 SHA 相同；底座 5 个分片与上游固定 revision LFS SHA 相同。训练/推理产物身份绑定通过。API 本轮 5,829 请求 / 5,820 成功响应，8 次 Objects 失败和 1 次 dev SSL 失败保留；货币费用和训练峰值显存未知。
- E0 split SHA 未变，相邻 audit 冻结产物只读。最终报告/汇总表提交后，两服务器继续仅做 pull --ff-only；RTX 既有删除保留。

## Spatial 方向后端 v2 验收

- 用户明确要求完成方向修复后，新增 `spatial_repair.py` 并接入正式 `score_matrix.py`：主体/客体分别配对，按符号判四方向，保留主轴/IoU 权重；纯文本冠词清理后接公共实体词表。Origin 代码未改，`legacy-official` 可以重现旧接口和公式。原始模型 JSON、final checkpoint 与视觉缓存未改。
- 170 passed：新增四方向端到端四格、随机框反射/反关系/换主体客体、IoU 与轴并列、双主体缺客体、多实例、非法框、未知名称、多关系/缺帧、常数评分不能通过契约、原来反向不得硬设为下降、重复观测拒绝和新旧表格口径回归。只有已有 Trainer 文件 mtime 警告。
- 全 980 个四方向视频的证据镜像精确核对；新 Spatial 7,800 清单项 / 23,400 记录全部执行。106 个几何单向正确视频为 0.4442→0→0→0.4442；98 个单向反例为 0→0.3724→0.3724→0。各自四格全部通过；0.03 幅度门槛分别为 102/106 和 90/98，未混同这两个判据。
- 真实视频镜像复用既有重跑 GRiT 缓存：上述 106 视频为 0.4442→0.0094，同步方向词后 0.3919；101/106 降幅至少 0.03，只有 40/106 同步恢复到原分 ±0.03。报告明确检测器误差与数学等变性的差别，几何条件不是人工视觉金标。
- legacy 模式逐字段重现 23,400 条 Spatial 旧记录；v2 Origin 的 7,800 条不变。四维正式入口复算 88,980 条，Objects 38,220 / Scene 9,360 / Action 18,000 与旧记录逐字段相同。新渲染器按报告所声明的 Spatial 后端生成解析表，旧报告仍使用旧序列化。
- `uv lock --check`、主 CLI 和四个相关脚本 `--help`、`git diff --check` 通过。完整来源/输出 SHA、新表及复现命令见 [Spatial v2 报告](deterministic/spatial-repair-v2/README.md)。本轮纯 CPU 缓存复算，无新训练、视觉推理或付费 API；旧 matrix-v1 保留，未提交或推送本轮修改。

## Action 同义接口 v2 验收

- 35 个同义 other 中，31 个为非 other 原始短语被旧归一化器拒绝，4 个为原生 other；保留 raw、旧目标、新目标与接口覆盖 trace。既有 60 对词典移动至生产模块，与 ea9d500 的字典逐项相同，未按错例新增映射。
- 推理、批量缓存、评分和排表已接入统一接口；`legacy-model` 保留旧结果，原生模型冒烟评测默认仍用旧路径。新系统明确标注“v9＋确定性接口”，未宣称纯模型学习收益。
- 195 passed，覆盖所有 400 原名、同义/异类条件、未知与混合未知、否定/观看/意图、歧义、多动作、含 and 类别、缺失/非法输出、raw 重放、同一接口推理接线与表格版本；已有 Trainer mtime 警告保留。
- 60/60 同义类别正确且一致；1,200/1,200 视频评分精确一致，Repair 0.88917→0.88917，CI=[0,0]；Origin 0.88917→0，Δ CI=[−0.92833,−0.84333]。仅改输出归一化为 51/60、0.865→0.770，说明完整当前输入契约也有贡献。
- 类别替换 60/60 正确、均分 0.00167；单个 OOV 仍 other/0，混合已知＋未知 60/60 集合正确、均分 0.44458，已知类分量不丢失。未把一个 OOV 短语重复的视频当作独立拒识样本。
- 88,980 条全矩阵中，只有 Action Repair-model 的 6,000 条采用新接口，其余 82,980 条与 Spatial v2 版本逐字段相同。原始模型字符串全部不变，raw 后处理与正式评分 6,000 条一致。
- `uv lock --check`、主 CLI 与 7 个脚本 help、diff check 通过；CPU 复算，没有新增训练、GPU 推理或 API。新表、来源哈希和复现命令见 [Action v2](deterministic/action-repair-v2/README.md)。本轮修改保留在工作区，未 commit/push。

## Objects 相邻帧确认 v2 验收

- 新增纯 Python `objects_repair.video_scores`，Repair 当前实体须得到相邻同名框 IoU≥0.5 确认；固定阈值、16 帧分母、不跨缺帧/环绕/补检测。接口只接收实体与检测框，原始/目标/背景/非目标遮挡统一处理；原始模型、缓存、可见性标注和官方源代码未改。
- `score_matrix.py` 默认新后端，`--objects-backend legacy-official` 保留旧链路。确认不足明确记为 unconfirmed/弃权，不能作为已验证阴性；不可见端点“无正分”与“全部明确拒绝”分别统计。新增独立审计脚本及表格版本标识。
- 完整 **219 passed**；已有 Trainer mtime 警告。测试覆盖孤立/持续检测、定位不一致、IoU 边界、多实例/多实体/未知实体、缺帧/坏框/单帧、时间反转/平移/缩放、元数据不能控制评分、弃权不可冒充真阴性、重复记录与回归分母。主 CLI 与 4 个相关脚本 help、lock、diff/新增文件 whitespace 检查通过。
- 275 个模型复核不可见端点中，正分残留视频 29→15，配对正分率差 −0.05091，家族 CI=[−0.08392, −0.02244]；正分帧 158→118/4,400。零分端点 246→260，但全部明确阴性仍为 **246**。475 个几何不完整端点保持未知，全计划分母 980 不变。
- 原始均分 0.28431→0.26173，差 −0.02258，CI=[−0.02659, −0.01907]；原始旧正分帧保留 4,104/4,458，正分视频保留 362/442。模型复核双方可见帧中，原始检出 3,120→3,023/4,887；遮挡后仍可见检出 38→19/1,024。固定 115 个旧正基例隔离移除的降分成功 114→106，要求无弃权时 87/115。上述损失与持续残留均进入报告，未宣称全面修复或新模型收益。
- 全部 88,980 条正式复算；仅 Objects 两个 Repair 的 25,480 条进入新后端，其余 **63,500 条**逐字段不变。legacy 全量逐字段重现 **38,220 条 Objects** 旧记录；7,840 条 Repair 恒等/零遮挡的评分和状态精确不变。遮挡外像素/等面积背景/零遮挡检查通过，新版四维 81 条汇总表已生成。
- 评分耗时 30.159 s（脚本计时），纯 CPU；无新训练、GPU 推理、权重下载或 API。H100 GPU 4 只做可用性预检，当时有既有高负载作业。全部报告输入、代码、支持文件及输出 SHA 核对通过。报告和复现见 [Objects v2](deterministic/objects-repair-v2/README.md)；本轮改动未 commit/push。

## 当前四维结果统一整理

- 新增 [current-summary](deterministic/current-summary/README.md)：合并 Spatial/Action/Objects v2 与现有 Scene，列出 12 条原始维度×方案结果、81 条全条件 CSV、专项统计 JSON 和来源 manifest。原始均分、条件子集、正确性与可见目标代价分开报告。
- 三个 v2 专项报告与最新 matrix-objects-v2 的均分、差值、覆盖、弃权及 CI 逐项一致；汇总输入/输出 SHA 和文件链接核验通过。历史结果与代码保持，不重跑训练、推理或评分。
- 本次仅文档与汇总产物更新；差异格式检查通过。最近一次完整实现验收仍为 219 passed，不将其写作本次新运行。未 commit/push。

## 同数据规模消融、泛化挑战与 Action 范围保护

- 用户授权消融和小模型研究，沿用已确认的 RTX4090 GPU 5（UUID `GPU-70f0792b-3dcf-6b1b-8c58-bffaf406353a`）；预检空余 46,392 MiB。在独立目录重新训练四个 0.6B LoRA，复用当前 8B 数据 SHA 与配置，只改变模型/revision/输出路径。固定 600/300/300/900 步全部完成，Trainer 共 2,527.3913 秒；新旧八个模型底座冻结、adapter 更新、final=末次 checkpoint 全部通过。
- 推理前冻结 672 条工程契约、74 条原始人工 SNLI 标签迁移诊断、144 条既有 dev 样本，并补充 60 条边界诊断，共 950 条；最终数据 SHA `cb09758edc198850129ac8ed3b9a3cc245e949a592a0de9a907f529bd7ef5011`。既有 dev 不称盲测，SNLI 不称 Scene 专用人工真值，词表类别重叠与 singular 序列化边界单列。
- 每种规模完成 2,292/2,292 次 base/SFT/Scene 去证据或错配证据预测，共 4,584 次，无缺失/重复；最大输入 428 tokens，无输入溢出或输出预算耗尽。所有底座分片、adapter、tokenizer、配置、锁和预测输出都有 SHA。推理峰值 allocated 为 0.6B 1.761 GiB、8B 16.858 GiB；资源统计不冒充交互延迟或纯模型吞吐。
- 新指标 WG-CC 绑定双端正确目标，缺失失败保留全分母；24 个预声明组的最差率及名义同时 Wilson 下界、家族 bootstrap 长短差齐全。Oracle=1，恒定空与复制 Oracle 原始输出=0。原冻结八个系统 WG-CC 均为 0；Action 新类名映射、Scene 长文本失败保留。
- 消融发现 Action v2 越界解析整段 and/without，覆盖正确输出。v2.1 限制声明契约范围并保留模型正确空动作；默认推理/批量/评分已接入，旧 `repair-v2` 与 `legacy-model` 可选。新同义词与模型权重未改。后验回放中 8B 长 Action 72/72、未执行动作 24/24 恢复；新改写严格类别仍 0/24，不宣称该修补得到新盲测证明。
- 冻结旧解释器 1,238 个全结果一致；原消融 JSON/CSV 逐字重现。旧 234 唯一预测目标不变，三方案各 6,000 条 Action 分数、目标、覆盖、弃权、缺失和帧分数相同。未改 Origin 或视觉缓存。
- 完整 **226 passed**，仅已有 Trainer mtime 警告；新/改 CLI help、uv lock --check、git diff --check 通过。加入恒定/复制捷径、缺失分组/预测、空间逆关系、证据任务不读标签、400 类在非动作风格文本下保持模型目标、空结果保护、生产 v2/v2.1 路由回归。报告见 [ablation-v1](deterministic/ablation-v1/README.md)。未 commit/push，未调用新付费 API 或下载模型。
