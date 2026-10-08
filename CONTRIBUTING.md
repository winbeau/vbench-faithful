# 贡献指南

## 开发环境

```bash
uv sync --locked                 # 默认开发工具和工作区包
uv sync --locked --group test    # 科学/合约测试组
uv pip install --python .venv --index https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
uv lock --check
```

模型推理需要显式安装完整闭包：`uv sync --locked --extra models`。这只解析 Python 依赖，不提供项目权重；某些第三方模型构造器在运行时可能下载默认 pretrained 资源，因此正式运行应预置本地权重并按 metric 文档传入路径。权重路径、CUDA、Detectron2/GRiT 等外部条件须配置并记录哈希。没有这些资源时仍可运行纯算法、输入/输出合约和 CLI help。

八个入口分别是 `dynamic-degree`、`motion-smoothness`、`subject-consistency`、`scene`、`human-action`、`spatial-relationship`、`overall-consistency`、`multiple-objects`。入口要求显式选择 `--vbench`、`--audit` 或 `--both`，并提供互斥的 `--video` 或 `--video-dir`；帮助和参数校验不应加载模型或下载资源。

```bash
uv run --no-sync --group test pytest --import-mode=importlib tests metrics
uv run dynamic-degree --help
uv run motion-smoothness --help
```

## 目录与数据

新增 metric 使用 `metrics/<kebab-name>/pyproject.toml`、`src/<python_name>/` 和 `tests/`。直接 import 的科学库写入该包基础依赖，模型库写入 `models` extra；不得通过另一个 metric 间接提供依赖。论文选定的方法、协议和发布哈希在 `configs/reproduction/`，复现证据在 `docs/`；修改时保留版本依据。新结果统一写入 `output/`，不提交权重或私人绝对路径。早期 E0 数据已清理，不再以根目录 `data/`、`results/`、`splits/`、`runs/` 作为必需输入。

## 提交与检查

Commit messages must be entirely in English, including the subject, body,
trailers, and copied logs. Chinese text is not allowed. Use
`type(scope): description`, for example `fix(core): correct video ordering`
or `docs(workspace): document the locked environment`. Allowed types are
`feat`, `fix`, `refactor`, `test`, `docs`, `build`, `ci`, and `chore`; mark
breaking changes with `type(scope)!: description` or a `BREAKING CHANGE:`
trailer. Keep subjects within 100 characters. Describe the reason, behavior,
validation, and limitations in the body when needed; keep research data
changes separate from code changes. Historical Chinese commit examples do
not override this policy.

Validate the entire message before committing:

```bash
python3 scripts/check_commit_message.py /path/to/commit-message.txt
```

The checker rejects Chinese text anywhere in the message. Do not bypass the
check with `--no-verify`.

本地可选安装 commit-msg hook（不修改全局 Git 配置）：

```bash
mkdir -p .git/hooks
cp scripts/hooks/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
```

提交前运行 `git diff --check`、锁检查、测试和入口 help。CI 检查锁一致、CPU 测试、目录边界及八个入口；GPU parity 只在明确 opt-in 的环境中执行。
