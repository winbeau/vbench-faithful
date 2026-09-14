# VBench Audit workspace 重构实施计划

日期：2026-09-14。规划：用户指定的 6-astra；实施：后续由 luna 按本计划执行。工作分支为 `winbeau`，本轮不创建 commit、不 push。本计划基于现有源码与同级官方 VBench checkout 的只读检查，不是从空仓库重新搭建。

## 1. 已核实的起点与目标

根 `pyproject.toml` 已有 uv workspace，八个 metric 均已有包、CLI、算法和测试，`uv.lock` 已被 Git 跟踪，`.python-version` 当前为 `3.10`。旧 README、`IMPLEMENTATION_PLAN.md` 及部分 `文档/` 仍描述“只有 scaffold / 没有 lock / 大部分 backend 未接入”，不能作为当前事实或阻止实施的前置门禁。

本次完成英文目录迁移、可重现开发环境、八维一致的用户 CLI、实际并发的视频分片、来源可追溯及研发规范。全部既有官方适配、audit 算法、消融、统计、测试及研究成果保留。算法存在不等于其当前环境具备 CUDA、模型权重及真实 parity 验证条件，验收必须区分这两层。

源码检查发现的具体缺口：

| 位置 | 当前行为 | 本次处理 |
| --- | --- | --- |
| `公共/audit-core` | 已有参数、输入、GPU 解析、轮转分片、结果写入；通用 `execute()` 仍是 audit stub | 复用有效基础设施；实际 metric 不接回 stub，不重写已有算法 |
| `motion_smooth/.../cli.py` | 裸 `--gpu` 不可用，多卡列表被拼成 `cuda:0,2,4`；audit 可自动回退 CPU；只有传 output 才落盘；异常可能中断整批 | 改为统一 parser、CUDA 校验、并发 batch 和默认输出；逐视频失败隔离 |
| `scene/.../cli.py`、`metric.py` | 不选模式默认为 audit；目录为 official/environment_grounded/global；按 GPU 分片后串行循环；factory 为 lambda | 用户入口强制三选一、规范目录；保留研究消融入口；改用可序列化配置与进程 worker |
| `overall_consistency/.../cli.py` | 输出 official/repair；`evaluate_sharded()` 实际串行 | 统一外部 backend 名与真实多进程并发 |
| `subject_consistency/.../cli.py` | official 聚合随请求 GPU 数改变，单卡按 transition 加权，多卡视频均值 | 明确并固定聚合协议，见第 6 节；不将硬件数量作为科学公式选择器 |
| dynamic / human / spatial / subject / multiple 的 `metric.py` | 已有 `multiprocessing.get_context("spawn")`、临时 JSON、轮转 worker，但实现重复且部分缺重复校验/中断清理 | 保留 batch 计算，收敛公共调度生命周期与合并校验 |
| `motion_smooth/models.py` | 延迟导入 `dynamic_degree.models.RaftFlowModel`，包依赖 dynamic-degree | 抽出小型共享 RAFT 模型适配，不让 metric 互相依赖评分实现 |
| 八维上游定位 | `/root/vbench1`、`/home/msy625/vbench1`、两个环境变量和旧 fork SHA 混用 | 使用同一受控上游配置和校验；历史 provenance 保留 |
| `scripts/` | 多处硬编码中文源码路径、机器路径和 PYTHONPATH | 更新可执行路径，使用 workspace 安装后的 import；历史结果中的路径不改写 |

主代理已核实当前机器有一块 RTX 4060 8 GiB，尚无已安装的 torch/模型权重；不能描述成没有 GPU。主代理负责记录 Python、uv 与基线测试事实；规划阶段未运行真实模型评测。

## 2. 目录迁移与保留边界

目录名及 distribution / console script 用连字符，Python import 名维持下划线。对既有文件做保留内容的移动，随后修改确有必要的路径与实现。Git 应能识别主要内容为 rename，不批量格式化科研代码。

| 当前路径 | 目标路径 | Python 包 / 命令 |
| --- | --- | --- |
| `指标/dynamic-degree` | `metrics/dynamic-degree` | `dynamic_degree` / `dynamic-degree` |
| `指标/motion_smooth` | `metrics/motion-smoothness` | `motion_smoothness` / `motion-smoothness` |
| `指标/subject_consistency` | `metrics/subject-consistency` | `subject_consistency` / `subject-consistency` |
| `指标/scene` | `metrics/scene` | `scene` / `scene` |
| `指标/human_action` | `metrics/human-action` | `human_action` / `human-action` |
| `指标/spatial_relationship` | `metrics/spatial-relationship` | `spatial_relationship` / `spatial-relationship` |
| `指标/overall_consistency` | `metrics/overall-consistency` | `overall_consistency` / `overall-consistency` |
| `指标/multiple_objects` | `metrics/multiple-objects` | `multiple_objects` / `multiple-objects` |
| `公共/audit-core` | `packages/audit-core` | 保持 `vbench_audit_core` |
| `文档/` | `docs/` | 保留并更新架构、CLI、上游映射、依赖说明 |
| `配置/models.example.toml` | `configs/models.example.toml` | 与已有 `configs/spatial_relationship/` 合并目录，不覆盖同名文件 |
| 无 | `packages/audit-models/src/vbench_audit_models/` | 仅在提取现有共用 RAFT 适配时新增，禁止容纳 metric 评分公式 |

目标骨架：

```text
vbench-audit/
├── pyproject.toml / uv.lock / .python-version
├── AGENTS.md / CONTRIBUTING.md / README.md
├── metrics/<metric>/pyproject.toml
│   ├── src/<import_name>/       # cli、metric、backends、已有科研模块
│   ├── tests/                  # 保留原测试
│   └── IMPLEMENTATION_REPORT.md
├── packages/audit-core/        # 通用输入、调度、输出、路径、provenance
├── packages/audit-models/      # 共用 RAFT 模型适配，按需建立
├── configs/                   # 模型配置示例、upstream.toml、原实验配置
├── docs/plans/                # 本计划及后续执行记录
├── docs/                      # 架构、CLI、开发环境、上游映射
├── scripts/                   # 保留研究脚本；新增有限开发检查
├── tests/                     # 跨 metric CLI/调度/输出合约测试
├── .github/workflows/         # 轻量 CPU CI
├── data/ / results/ / runs/    # 保留全部现有数据与成果
└── output/                    # 新运行输出，继续忽略
```

`data/`、`results/`、`runs/` 以及原始论文/计划/报告不做重命名清理，不重算、不删除、不修改其中冻结数据。根历史 `IMPLEMENTATION_PLAN.md`、`SCENE_AUDIT_PLAN.md` 保留，可在开头增加历史说明及新计划链接。`scripts/` 内活跃代码、tests 和当前文档中的旧路径必须更新；历史报告中的绝对机器路径允许保留并标注历史。不要用中文目录兼容软链接掩盖未迁完的运行引用。

## 3. uv workspace、依赖及开发环境

1. 根项目继续 `package = false`，显式依赖全部八个 metric 与 audit-core；members 改为 `metrics/*`、`packages/*`；同步全部 workspace sources。每个 metric 保留自己的 pyproject、src 布局、console script、直接依赖与 tests。对子项目依赖公共 workspace 包的 source 映射进行检查，不能靠根目录 sys.path 恰巧可用。
2. 使用当前 `uv.lock` 作为可追溯起点，迁移完成后运行 uv 重新生成锁并验证；不可手写锁，也不可照旧文档删锁或留下旧中文 editable 路径。记录实际采用的 uv 版本并在 CI 固定，`.python-version` 固定经验证的 Python 3.10 补丁版；兼容范围与科学依赖共同核实。
3. 默认开发安装保持可运行八个 `--help`、纯算法及合约测试。允许将 torch / torchvision / decord / CLIP / timm / transformers 等模型运行闭包归入各 metric 的 `models` extra；NumPy、OpenCV、SciPy 等按实际顶层/纯算法 import 决定是否属基础依赖，不机械全部移入 extra。提供根 `models` 聚合 extra，确保 `uv sync --locked --extra models` 有明确含义。帮助与参数错误不应触发模型导入、CUDA 初始化或下载。
4. 模型依赖移动至 extra 时仍必须声明完整已知闭包并由同一 uv.lock 解析。uv 会同时解析 optional dependencies；extras 不能藏住版本冲突。优先使用原有锁中的相容版本，依据实际 resolver 错误调整，避免无关全量升级。
5. Detectron2/GRiT 等需本地编译的部分单独写出受控构建步骤、源码 revision、Python/torch/CUDA 组合与检查命令。暂未验证的构建不能伪装为已可重现。权重在外部缓存，路径可配置，run 记录 SHA-256。uv.lock 不管理权重与驱动，不自动拉取大型权重。
6. 将 motion 与 dynamic 共用的现有 RAFT 包装迁至小型 `audit-models` 包，原 `dynamic_degree.models.RaftFlowModel` 可保持兼容 re-export。共享包直接依赖 core 的上游装载辅助及必要模型 extra；其余时间采样、motion audit 公式仍属各 metric。删除 motion 对 dynamic-degree 的发行包依赖，并补共享包装等价性测试。
7. dev dependency group 放实际需要的 pytest / 检查工具；避免为规范引入庞大工具链。生成新锁后在干净 `.venv` 执行 `uv sync --locked` 与 `uv lock --check`。独立 metric 以 `uv run --package motion-smoothness motion-smoothness --help` 验证，安装说明不依赖私人 conda 路径。

交付环境文档必须分别描述“开发合约可运行”和“特定模型推理已验证”，提供缺依赖时的准确安装提示。若真实模型环境暂缺资源，完成可验证的重构与锁文件，明确剩余的外部运行条件。

## 4. 官方上游与来源校验

规划时只读检查到：

- 官方 remote：`https://github.com/Vchitect/VBench.git`。
- 参考目录：`/home/winbeau/Papers/ICASSP2027-VBench-Audit/VBench`，即本仓库 `../VBench`。
- 参考 HEAD：`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`；检查时工作区 clean。
- 旧实现绑定 `https://github.com/msy625/VBench.git` 的 `13dee903cc97e2633ed6e8f50dea61bc90717935`；当前官方 clone 没有该对象，不能宣称二者源码等同。
- 八维均在官方根 `vbench/`，函数为 `compute_dynamic_degree`、`compute_motion_smoothness`、`compute_subject_consistency`、`compute_scene`、`compute_human_action`、`compute_spatial_relationship`、`compute_overall_consistency`、`compute_multiple_objects`。不导入 VBench-2.0 或 beta/i2v/long 的同名指标。

新增 `configs/upstream.toml` 作为当前源码 identity 的单一来源，记录 URL、固定 SHA、八维模块入口及需要核对的源文件哈希。路径按显式配置/`VBENCH_AUDIT_UPSTREAM` 优先、同级 `../VBench` 默认；已有 `VBENCH1_ROOT` 可作为带提示的兼容回退，不能八维取不同来源。路径在运行时解析，禁止将 import 时环境变量固化进默认参数。

core 提供统一 Git identity、clean 状态和 import origin 检查；允许固定 SHA 的 detached HEAD，不把分支名字当作代码版本。import 后确认模块来自选定 checkout，拒绝已缓存的其他 `vbench` 来源。旧 bundle 的来源历史保留于文档，不能作为任意来源的白名单绕过。旧 fork 与官方 revision 若需双支持，必须是明确的两个受控 profile，不能放宽成“任意 SHA 都能跑”。默认以本次官方固定 revision 为准。

实施者逐维核对实际调用签名、返回结构、预处理和第三方闭包；尤其 dynamic RAFT / spatial 源文件哈希、AMT、UMT、DINO、Tag2Text、ViCLIP。如没有旧源码供比较，如实写“旧版本差异未验证”，仍可记录已核对的新接口；不能仅替换常量就声称 official parity。上游 checkout 只读，不 checkout、patch 或修改其权重路径。当前 run 必须记录所用固定 revision，不能把当前结果写成旧论文实验的来源。

## 5. 统一 CLI、输入与输出

所有八维 public entrypoint 使用相同基础 parser：

```bash
uv run dynamic-degree --vbench --video /data/video_000.mp4
uv run dynamic-degree --audit --video-dir /data/videos --gpu
uv run dynamic-degree --both --video-dir /data/videos --gpu 0,2,4
uv run motion-smoothness --both --video-dir /data/videos --output ./output-custom
```

- `--vbench / --audit / --both` 必选且互斥；不新增 `--all`。vbench 指官方算法，audit 指保留的现有 audit 或未来明示未实现状态；both 分别计算两套，不能拿一套结果复制冒充另一套。
- `--video / --video-dir` 必选且互斥。复用 `enumerate_videos()`：目录只枚举直接子文件，`video_` 后至少三位数字、`.mp4`，数值排序，重复数字索引/非法 mp4/空目录在加载模型前失败。单视频接受任意有效文件名，不给视频改名。
- 不传 `--gpu` 和裸 `--gpu` 都是可见 GPU 0；`--gpu 0,2,4` 是选定设备列表，拒绝空值、重复、负数、非法编号。无 CUDA 不自动改 CPU。
- 保留必要 metric 专用参数与 `--metadata`。scene 的 global/environment_grounded 和 overall 的 repair 是算法内部变体；用户三种模式始终是外层协议。保留研究消融能力可用 `--audit --audit-variant global` 或独立研究脚本，迁移原脚本调用；不要允许单独 `--mode` 绕开必选三种标志。内部评分函数兼容名称可保留，外部字段规范化。
- 默认输出为**本仓库根** `output/<metric>/<vbench|audit>/<run-id>/`，不能依据偶然 cwd 落在 metric 子目录。显式 `--output DIR` 替换输出基目录，结果仍为 `DIR/<metric>/<backend>/<run-id>/`；相对 DIR 按调用 cwd 解析并在 run 中保存绝对路径。
- 同一次 both 的两套结果共用 run-id，重复运行不覆盖；每套保留 `results.json`、`results.csv`、`summary.json`、`run.json`、`run.log`。JSON/CSV 中 `backend` 同样统一为 vbench/audit，`variant` 或 `algorithm_mode` 单独保留 official/repair/environment_grounded 等科研含义。
- 默认根路径通过 core 从包源码向上定位带 workspace 配置的根，所有入口共用；不能依靠各 CLI 不同 parents 下标。显式 output 应可在独立安装环境使用；不在本 workspace 时给清晰说明而非猜测另一个 Git 仓库。

语义前置约束按选择的 backend 验证，而非强迫所有模式满足 audit 专属字段：

| 维度 | 最低语义要求 |
| --- | --- |
| dynamic-degree | 官方不依赖 prompt；audit 沿用现有 prompt/motion_target 路由与缺省行为，非法类型提前报错 |
| motion-smoothness | 官方与当前 audit 不要求文本 |
| subject-consistency | 当前视觉相似度无需语义元数据 |
| scene | 所选视频具有有效 scene 标签；保留当前嵌套 dimension_metadata/auxiliary_info 适配 |
| human-action | audit 要显式可解析 K400 target_action 或既有合法 prompt；vbench 保留官方按文件名取标签的语义，不把 metadata 标签偷换为官方标签 |
| spatial-relationship | 显式有序 object_a / object_b / relationship，复用现有 query parser |
| overall-consistency | 非空 prompt，audit 条件及参数合法性保持现有规则 |
| multiple-objects | 复用 parse_target_objects 与官方元数据转换的完整目标集合约束 |

复用 `metadata.json` 的 `videos` 列表协议与显式 `--metadata` 优先规则，逐视频匹配，保留已有所有字段及研究脚本扩展。编号文件名不含语义；特别 human-action 官方文件名基线在 `video_000.mp4` 下可能给出低/零分，文档和 run 应说明该事实，不重命名输入制造官方分数。

## 6. GPU worker、并发、聚合与失败

采用现有 spawn + 每 GPU 一个 worker 的设计，不引入 DDP、Ray 或跨视频分帧。core 的共享 coordinator 接收可 import 的顶层 worker/batch 函数及可序列化配置；模型实例、lambda、打开文件对象不跨进程传递。metric 负责构造自身模型、逐视频计算及聚合。重构优先迁移现有五套 spawn 外壳，再接通 scene / overall / motion，避免八维各写一套新调度器。

确定规则：

1. 父进程完成输入/元数据/GPU/配置前置校验，使用输入数值排序的稳定列表；`shard[k] = videos[k::len(gpu_ids)]`。只启动非空分片；单视频只落到列表首张 GPU。
2. GPU 编号是当前 `CUDA_VISIBLE_DEVICES` 之下的**可见逻辑编号**。例如 mask=`2,4,6` 时 `--gpu 0,2` 选择 mask 中第一和第三张，即可映射到原编号 2 与 6；不能再次当物理编号进行索引。无 mask 时按 CUDA 枚举顺序，不能无证据声称和 nvidia-smi index 永远同义。
3. 最小可靠实现让 spawn worker 继承原 mask，在任何模型/CUDA allocation 前 `torch.cuda.set_device(gpu_id)`，显式传 `cuda:<gpu_id>`。不要在已 import/初始化 CUDA 后改 mask，也不要 fork 已初始化的父进程。为每张卡设置当前设备可兼容上游裸 `.cuda()` 调用。父进程仅校验设备和协调，不加载模型。
4. 所有 worker 先 start，再进入 join/结果收集，确保实际并发。每个 worker 每种模式只初始化一次模型；both 先完成一整套、join 并释放进程资源后再运行另一套。单卡也走一致的受控生命周期，或保留已验证的单卡路径并显式释放模型。
5. 防止继承外部 torchrun 的 WORLD_SIZE/RANK 等使上游再次分片：入口检测并给清晰错误，或在隔离 worker 环境明确采用 world_size=1；选一种并测试/文档化。本项目由自身协调全部视频，不允许二次分片导致漏算。
6. 写 worker 临时结果，父进程校验每个 `(backend, video)` 恰好一个记录，拒绝重复、额外和遗漏键，再恢复原输入顺序。不自动重试模型推理，以保证每个已分配视频每模式至多执行一次；异常退出导致未计算项有明确失败记录。
7. Seed 显式传入每个 worker，并在 worker 内初始化。需要逐视频随机性的实现以统一 seed 与稳定输入身份派生，使结果不因 worker 分配改变；不能用 Python 随机 hash 派生。GPU 非确定性仍如实记录，不承诺浮点逐位一致。
8. run 记录请求编号、原 mask、可用卡数、每个 shard 的视频、worker PID、起止时间、设备名及能取得的 UUID/PCI bus ID。物理标识拿不到则记录 null/原因，不编造。

聚合由 metric 处理完整逐视频充分统计量，禁止平均分片均值。scene / multiple 的官方按实际 frame totals 聚合；spatial 保留官方帧级汇总语义；其他维度沿用已核查公式。特别 `subject_consistency` 官方源码存在单进程 transition 加权、多 rank 视频均值的差异，本项目不运行官方分布式，而是独立视频 worker，统一采用官方单进程 transition 加权公式用于所有 GPU 数量，并更新 formula_version / 文档 / 回归测试。保留视频均值作为命名诊断可选项，不混充同一 aggregate。此项是为并行调度固定协议的显式变化，不重写逐视频官方公式。

退出及失败协议：

- 参数、输入、缺必需元数据、无可用设备等共享前置错误：stderr 清楚说明，退出 2，加载模型前结束，不创建伪成功记录。
- 单视频损坏：该项 failed、score=null，其余继续；父进程完整写结果，整体退出 1。
- backend 专属依赖/权重/上游不满足：为该 backend 保存 blocked/failed 与原因，both 仍尝试另一套满足条件的 backend；不能把一个 backend 专属问题变成无记录的全局提前返回。所选 backend 自己需要的参数/元数据先验证。
- worker 崩溃、结果缺失/重复、序列化失败：检测并保存可用结果，对不能信任的项明确 failed；不得静默丢失分片、覆盖已有视频结果或返回 0。
- KeyboardInterrupt：终止并 join 全部存活 worker，保留已收到结果和剩余 interrupted 状态，退出 130；不留下后台 GPU 子进程。
- 未来 audit 确无实现才用 not_implemented + score=null + 非零退出；不得将已有 audit 改成 stub 以方便通过接口测试。

输出 provenance 在现有 run 信息上统一添加 code SHA/dirty、uv.lock 哈希、输入/元数据哈希、上游 SHA/source hash、模型路径/哈希、Python/uv/torch/CUDA、参数、公式版本、时间和调度信息。已有逐视频 diagnostics、论文 scalarization 和自定义字段不丢失。

## 7. 研发规范与适量 CI

创建根 `AGENTS.md`：先读 README/本计划；遵守 metric/core/model 包边界；禁止改上游或冻结实验数据；不自动提交/推送；用户已授权时不重复确认普通修复；源码改动后运行对应算法及合约测试；官方公式变化需固定来源和 parity 证据；硬件验证缺失明确报告。明确本次用户指定先 astra 规划、luna 实施，不追加无关代理。

`CONTRIBUTING.md` 给出 uv sync / tests / CLI / extras / 路径规范，以及 commit 注释规则：`type(scope): 描述`；type 采用 feat/fix/refactor/test/docs/build/ci/chore，scope 如 core/workspace/dynamic-degree；正文说明动机、行为变化、验证与限制，重大协议变化用 `!` 或 BREAKING CHANGE。可接受中文描述，标题保持清晰短句。不要把论文数据改写混进代码 commit。当前任务仅建立规范，不真的创建 commit。

用一个轻量 `scripts/check_commit_message.py` 校验标题和允许的类型，提供正反例测试及可选的本地 hook 接入命令；不擅自修改全局 Git hooks 或配置。CI 检查 PR 标题或新 commit 范围，不能 retroactively 判整个历史为失败。

CPU CI 至少检查锁一致、workspace sync、八入口 help 与参数合约、核心/metric 纯测试和目录/边界检查。主代理发现各 metric 的空 `tests/__init__.py` 导致同名 `tests.test_cli_smoke` 错误复用：即使 `--import-mode=importlib` 也曾收集到其他维度运行 dynamic 测试。删除不需要的测试包标记或建立唯一模块命名，并检查每个真实测试函数的 collection provenance，不能仅凭测试数量宣称八维已覆盖。GPU parity 保持显式 opt-in，普通 CI 不下载模型、不依赖私有绝对路径。不要启用全仓强格式化来制造无关 diff。

根 `tests/test_pairwise_statistics.py` 与对应脚本含 `/root/vbench-audit` 硬编码，应改为本仓库相对定位；所需 E0 CSV 和 split fixture 已存在于 `results/e0/raw_official_scores/` 与 `data/processed/pairwise_master_split.csv`，保留全部统计断言并实际运行，不为路径问题跳过。动态 CLI smoke 的上游校验 mock 应补在测试边界，不能通过放宽生产 provenance 校验修测试。

## 8. 执行顺序与验收

### P0：记录基线

- 确认 `git branch --show-current` 为 winbeau，保存现有测试通过/失败/skip 清单、uv/Python/CUDA 状态；列出 tracked data/results/runs 的 blob identity 供迁后比对。
- 记录官方固定 revision、八维入口、旧 fork 限制。旧测试失败先分类为基线缺陷/环境缺失，后续不能删除或 xfail 掩盖回归。

### P1：迁移目录与文档入口

- 按第 2 节逐项移动；更新 workspace、脚本、测试、当前文档链接。
- 创建 AGENTS / CONTRIBUTING 与来源配置骨架；保持研究数据及算法文件内容。
- 检查旧目录运行引用，生成锁并先让八个 help 在 uv 环境中可用。

### P2：统一基础协议和来源

- core 增加路径解析/来源校验/输出 backend 规范化，更新各适配器调用；保留 metric 原算法和输出扩展字段。
- 统一 parser 和前置校验；motion 修复默认输出、裸 GPU、CPU 回退，scene/overall 外部模式映射。
- 迁移 RAFT 共用适配与直接依赖，完成 dev/models 环境说明。

### P3：调度收敛并接通剩余三维

- 在已有 spawn 实现基础上提取共享协调器，验证成功/故障/中断合并，再让八维全部使用可靠并发路径。
- 固定 subject aggregation 协议；保留 scene/overall 消融，通过科研脚本或 audit variant 显式选择。
- 确保 both 两套结果、资源释放与 backend 专属失败行为符合合同。

### P4：全量合约、回归、文档与交接

以下命令中的模型环境和数据路径须以机器实际情况替换，不把示例执行成功当作事实：

```bash
git branch --show-current
uv --version
uv lock --check
uv sync --locked
uv run python scripts/check_environment.py
uv run --package dynamic-degree dynamic-degree --help
uv run --package motion-smoothness motion-smoothness --help
uv run pytest --import-mode=importlib tests metrics
git diff --check
git status --short
```

新增有针对性的 tests，而不是为目录移动堆砌镜像测试：

- 八入口参数化检查：help、缺模式、缺输入、互斥冲突、三种 GPU 表达、拒绝 `--all`/无效 GPU；parse 测试不要求 CUDA。
- 从根与子目录调用同一入口，mock 模型并断言默认 output 根一致；both 共用 run-id、只有 vbench/audit 目录、保留旧 run。
- 8 个视频、3 张模拟设备分片为 0/3/6、1/4/7、2/5；非整除、单视频多卡、稀疏编号、输入顺序恢复及每模式全覆盖唯一性。
- 使用真实 spawn 的无模型顶层 worker（不依赖 lambda），记录 PID/起止与同步信号，证明多个 worker 在重叠时间内运行；检测生产入口的接线，不能只有 round_robin 单测却 CLI 仍串行。
- GPU 可见编号/mask 映射、设置当前设备先于模型构造；外部分布式环境不发生二次分片。
- 一个视频异常、worker 非零退出、缺/重复/额外结果、backend 不可用与中断清理；验证退出码、保留的成功项及另一 backend 行为。
- 用不等视频长度/分片大小测试 subject 的 GPU 数无关加权聚合，以及 scene/multiple frame totals，禁止平均 shard 均值。
- 保留既有八维纯算法、反事实、官方适配 mock/parity 测试；真实 parity 保留 opt-in，报告其未执行原因。
- 校验所有 distribution/console scripts 与迁移映射一致；metric 不导入另一个 metric；现有冻结研究文件 blob 未改变。

具备 CUDA、官方依赖与权重后，以一批真实有效的编号视频运行单卡和多卡对照：

```bash
uv sync --locked --extra models
uv run dynamic-degree --both --video-dir /data/videos --gpu 0 --output ./output/acceptance-single
uv run dynamic-degree --both --video-dir /data/videos --gpu 0,2,4 --output ./output/acceptance-multi
uv run motion-smoothness --both --video-dir /data/videos --gpu --output ./output/acceptance-motion
```

对另外六维提供相同形式的验收示例和所需 metadata；比较逐视频及聚合允许误差、覆盖数量、worker 设备与两模式目录。若本机没有三张可用 GPU 或无权重，真实三卡和模型 parity 标为未验证；仍须完成不需要这些资源的真实进程调度合约。所有通过/失败/skip 数字写在最终交接记录，不能仅写“测试通过”。

## 9. 风险与非目标

- 官方新 SHA 与旧 fork 不是已证实相同源码，固定来源和接口核对必须做；真实数值 parity 需权重和 CUDA。
- GPU spawn 会暴露先前 lambda、闭包、顶层模型 import 和异常序列化问题，优先解决可序列化边界，不退回串行伪多卡。
- 依赖 extras 不等于依赖冲突隔离；uv.lock 只覆盖可解析的 Python 环境，外部源码编译/权重/驱动需单独证明。
- 输出模式名、必选标志与目录迁移会影响旧研究脚本；更新活跃脚本但不改冻结报告中历史运行记录。
- 本轮不设计新 audit 评分、不重新标注数据、不扩大到 VBench 2.0、不发布 PyPI、不运行长耗时全量实验、不提交或 push。现有 algorithm-level baseline 问题仅在阻断请求的统一行为时做最小修复，并明确记录。

交接给 luna：按 P0→P4 执行，参考本计划列出的真实缺口；优先保留源码再统一基础设施。完成后报告迁移路径、实际依赖/上游设置、八维 CLI 行为、已执行验收及外部限制，所有修改留在 winbeau 工作树供用户审阅。

主代理如采用其拟定的三个 luna 分工，先由 layout 完成迁移并通知，再并行分配互不重叠的文件：layout 负责 pyproject/lock、目录、规范/当前文档及开发 CI；runtime 负责 core CLI/paths/devices/runner/outputs、metric `cli.py`/`metric.py` 与调度合约；upstream 负责 core `upstream.py`、各 `backends/vbench.py`、必要 `models.py`、共享 RAFT 包与来源测试。新增 package 依赖声明由 upstream 提交具体需求给 layout，避免同时改 pyproject；算法文件和相邻测试若需交叉修改，先向主代理协调所有权。

## 实施记录（2026-09-14）

实际验证采用 Python 3.11.14（系统可用且已由 uv 验证），相对计划中的 Python 3.10 是有记录的兼容性偏差；包仍保留 `>=3.10,<3.13`，并为 Python 3.10 声明 `tomli` 条件依赖。实际环境检查命令为 `uv run --no-sync python scripts/check_environment.py`（脚本位于仓库 `scripts/`）；不会引用私人 conda 路径。锁检查和同步分别使用 `uv lock --check` 与 `uv sync --locked`。
