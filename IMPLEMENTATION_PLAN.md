# vbench-audit 架构与实施计划

## 1. 目标与实施前提

在 vbench-audit 中参照已 clone 的 VBench1.0 fork，将 8 个维度抽离为可独立维护的 Python 子项目，以 uv workspace 统一依赖解析，并提供 `uv run dynamic-degree` 等一致的命令行入口。

- 目标仓库绝对路径：`/home/msy625/projects/vbench-audit`。
- 参考 fork 绝对路径：待核实；当前工作区未发现参考源码。
- 参考版本：实施前记录 remote URL、分支、commit SHA、工作区改动及子模块版本；不能把“VBench1.0”当作已验证的 Git tag。
- 当前目标目录是 Git 仓库，但尚无 commit；参考源码路径尚未确认。因此下述结构是设计方案，具体依赖、权重、源码入口和公式必须在实施第一阶段核实。
- 不覆盖已有改动，不自动提交、推送或修改参考 fork。

## 2. 完整目录结构

保留中文目录与已有命名；CLI 统一使用连字符，Python 导入名使用下划线。

```text
vbench-audit/
├── pyproject.toml                 # 根项目，依赖全部 metric 和公共包
├── uv.lock                        # uv 生成的唯一 workspace 锁文件
├── .python-version                # 核实兼容性后固定
├── .gitignore
├── README.md
├── 指标/
│   ├── dynamic-degree/            # src/dynamic_degree/
│   ├── motion_smooth/             # src/motion_smoothness/
│   ├── subject_consistency/       # src/subject_consistency/
│   ├── scene/                     # src/scene/
│   ├── human_action/              # src/human_action/
│   ├── spatial_relationship/       # src/spatial_relationship/
│   ├── overall_consistency/        # src/overall_consistency/
│   └── multiple_objects/           # src/multiple_objects/
├── 公共/audit-core/                # src/vbench_audit_core/
├── 脚本/
│   ├── check_environment.py
│   └── smoke_test.sh
├── 配置/models.example.toml
├── 文档/
│   ├── architecture.md
│   ├── cli.md
│   └── upstream-mapping.md
├── tests/
│   ├── test_cli.py
│   ├── test_dispatch.py
│   ├── test_output_contract.py
│   └── fixtures/
└── output/                         # 生成文件，不提交
```

每个 metric 使用相同结构：`__init__.py`、`cli.py`、`metric.py`、`backends/vbench.py`、`backends/audit.py`；必要时在 `vendor/` 保留带许可证的上游实现。`metric.py` 负责输入要求与聚合，模型推理和官方预处理归各自 backend。公共包只放通用基础设施，不集中存放各维度算法。

| 子目录 | 包名与 CLI | Python 模块 |
|---|---|---|
| dynamic-degree | `dynamic-degree` | `dynamic_degree` |
| motion_smooth | `motion-smoothness` | `motion_smoothness` |
| subject_consistency | `subject-consistency` | `subject_consistency` |
| scene | `scene` | `scene` |
| human_action | `human-action` | `human_action` |
| spatial_relationship | `spatial-relationship` | `spatial_relationship` |
| overall_consistency | `overall-consistency` | `overall_consistency` |
| multiple_objects | `multiple-objects` | `multiple_objects` |

## 3. uv workspace 与依赖边界

“独立”指各 metric 拥有自己的源码、依赖声明、构建入口和测试；workspace 共享一份锁文件和解析环境，不等同于八套可容纳冲突依赖的隔离环境。

根项目应显式依赖全部成员，使根目录可直接运行入口：

```toml
[project]
name = "vbench-audit"
version = "0.1.0"
dependencies = [
  "audit-core", "dynamic-degree", "motion-smoothness",
  "subject-consistency", "scene", "human-action",
  "spatial-relationship", "overall-consistency", "multiple-objects",
]

[tool.uv]
package = false

[tool.uv.workspace]
members = ["指标/*", "公共/audit-core"]

[tool.uv.sources]
audit-core = { workspace = true }
dynamic-degree = { workspace = true }
motion-smoothness = { workspace = true }
subject-consistency = { workspace = true }
scene = { workspace = true }
human-action = { workspace = true }
spatial-relationship = { workspace = true }
overall-consistency = { workspace = true }
multiple-objects = { workspace = true }
```

每个子项目声明构建后端、src 包发现配置、直接依赖，并注册 console script，例如：

```toml
[project.scripts]
dynamic-degree = "dynamic_degree.cli:main"
```

metric 只能依赖公共包及自身声明的第三方库，不允许相互导入。单包构建和依赖检查必须防止共享环境掩盖漏报依赖。

实施顺序：先检查 Python、uv、虚拟环境、CUDA、PyTorch 和模型库版本，再解决真实兼容性问题，最后由 `uv lock` 生成锁文件。不凭空指定模型依赖版本，不手写锁文件。若依赖无法共存，先记录冲突和替代方案。模型权重独立缓存并记录哈希，不进入 Git 或 `uv.lock`；大型下载前报告大小、耗时、设备和磁盘需求并等待确认。

## 4. 统一 CLI 契约

```text
uv run <metric> (--vbench | --audit | --both)
                (--video FILE | --video-dir DIR)
                [--output DIR] [--gpu [IDS]]
                [--metadata FILE] [--seed INT]
```

| 参数 | 行为 |
|---|---|
| `--vbench` / `--audit` / `--both` | 必须且只能指定一个；分别执行官方、audit、两种后端 |
| `--video` / `--video-dir` | 必须且只能指定一个 |
| `--output DIR` | 默认 `<仓库根>/output`；相对路径按调用目录解析 |
| 不写 `--gpu` 或只写 `--gpu` | 默认使用 0 号卡 |
| `--gpu 0,2,4` | 使用指定的 CUDA 可见设备编号 |
| `--metadata FILE` | 显式元数据入口；语义维度缺失时自动查找或报错 |
| `--seed INT` | 默认建议 42，并记录到运行清单 |

`--both` 是需求中的“两种后端”，不另设 `--all`。输出路径为 `<output>/<metric>/<backend>/<run-id>/`，两后端共享 run-id 但目录隔离，重复运行不覆盖旧结果。

无 CUDA 或 GPU 编号无效时提前报错，不自动切 CPU；检查空列表、重复编号、负数和非数字。编号基于当前 `CUDA_VISIBLE_DEVICES` 的可见顺序，日志同时记录可见编号和可取得的物理标识。

audit 尚未定义时返回 `not_implemented`，不得生成假分数或调用官方逻辑冒充 audit。`--both` 仍保存 vbench 结果和 audit 未实现状态，并返回非零码。

## 5. 批量输入与语义元数据

- 仅扫描指定目录的直接子文件，接受 `video_` 加至少三位数字再加 `.mp4`，按数字索引排序。
- 空目录、重复索引和不符合约定的 mp4 在加载模型前报错；忽略非视频文件。
- 单个损坏视频记录失败原因，其余视频继续；部分失败整体返回非零状态。
- 统一 `metadata.json` 包含 `videos` 列表，每项用相对文件名 `video` 对应 `prompt` 和 `dimension_metadata`。字段须根据锁定 fork 的输入协议确定并转换为官方结构。
- 自动查找批量目录或单视频所在目录的 `metadata.json`；显式 `--metadata` 优先。缺项、重复映射或标签不完整时推理前报错，不从编号猜测语义。
- 任意视频上的官方算法结果不等同于完整官方 benchmark 分数。

## 6. 多 GPU 调度与聚合

采用按视频的数据并行：一个 worker 对应一张 GPU，worker 完整处理视频，不跨 GPU 拆帧；使用 spawn，并在子进程 CUDA 初始化前设置设备。

对排序后的视频使用 `shard[k] = videos[k::GPU数量]` 轮转分配。只启动非空分片，单视频只使用列表中的第一张卡。worker 写临时结果，主进程校验无重复、无遗漏并按输入顺序合并。聚合必须使用维度官方公式，不能简单平均分片均值。

`--both` 先运行 vbench、释放模型资源后再运行 audit。两后端共享输入与元数据，但分别记录预处理协议。

## 7. 输出和可复现性

每个 `<metric>/<backend>/<run-id>/` 保存：

- `results.json`：逐视频状态、原始分数或指标结构、错误信息。
- `results.csv`：逐视频表格结果，失败分数为空。
- `summary.json`：聚合值、公式版本、有效样本数、失败数及完整性状态。
- `run.json`：命令、参数、种子、输入和元数据哈希、代码 SHA/dirty 状态、上游 SHA、模型版本和权重哈希、Python/uv/PyTorch/CUDA/驱动、设备、耗时、输出路径和锁文件哈希。
- `run.log`：主进程和 worker 日志；可附 `workers/` 中间结果。

官方 backend 保留官方采样、解码、归一化、阈值、推理和聚合方式；必要适配记录在 `upstream-mapping.md`。固定种子仍有非确定性时如实记录。

## 8. 五阶段实施与验收

1. **盘点源码与环境**：检查两仓库 Git 状态，固定参考 SHA，定位 8 个维度的函数、依赖、权重、元数据、采样与聚合；输出源码映射和依赖兼容性表，确认没有误用 VBench++/2.0。
2. **建立 workspace 与公共接口**：创建根配置、8 个成员和公共包；完成参数互斥、输入校验、GPU 解析、结果协议和入口注册。八个入口的 `--help` 应可用，缺失必选参数应返回非零。
3. **打通 dynamic-degree**：抽取最小依赖闭包，实现单视频、批量、官方 backend 和 audit 未实现状态；先做小样本 smoke test，再与锁定 fork 比较。
4. **迁移剩余七维度**：逐个接入和验证，为语义维度增加适配器；检查源码差异、单包依赖和官方等价性，不以仅能 import 作为通过标准。
5. **验证多卡并完成文档**：验证分片覆盖、非整除数量、少视频多卡、失败传播、目录隔离和聚合一致性；无多 GPU 时只验证调度单元测试并明确限制。

核心验收：同一输入在单卡和多卡下逐视频结果及官方聚合在算法确定的合理误差内一致；mock 测试只能证明调度和协议，不能证明模型评测正确。首次真实推理前检测 CUDA，并先使用小规模样本。

## 9. Ubuntu / WSL Bash 使用示例

目标接口当前尚未实现：

```bash
uv sync --locked
uv run dynamic-degree --vbench --video /data/videos/video_000.mp4
uv run dynamic-degree --vbench --video-dir /data/videos --gpu
uv run dynamic-degree --both --video-dir /data/videos --gpu 0,2,4
uv run dynamic-degree --audit --video-dir /data/videos --output /data/results
uv run scene --vbench --video-dir /data/videos --metadata /data/videos/metadata.json
uv run motion-smoothness --vbench --video /data/videos/video_000.mp4
```

开发阶段首次锁定由 `uv lock` 生成；复现实验使用审核后的锁文件和 `uv sync --locked`。Ctrl+C 后主进程应回收 worker 并保存 interrupted 状态，不能标记为成功。

## 10. 实施依据与待核实事项

- [uv workspace 官方文档](https://docs.astral.sh/uv/concepts/projects/workspaces/)：共享锁文件、根项目运行语义和 workspace 依赖来源。
- [VBench 官方仓库](https://github.com/Vchitect/VBench)：实施须对照指定 fork 的确切 SHA，不能直接照搬当前主分支全部组件。
- 待核实：参考 fork 绝对路径、8 个维度源码及许可证、模型权重和依赖版本、Python/CUDA 兼容组合、语义元数据字段、各维度公式与数值容差。

## 当前边界

本次仅新增此计划文档；未修改目标仓库源码，未安装依赖、下载模型、生成 `uv.lock` 或运行模型实验。
