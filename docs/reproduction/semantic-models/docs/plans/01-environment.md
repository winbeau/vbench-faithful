# P0：环境、仓库与部署

## 固定配置

- 本地：与 vbench-audit 同级的 vbench-prompts-compile。
- Remote：git@github.com:winbeau/vbench-prompts-compile.git，private。
- Remote host：rtx4090，登录用户 luxliang，项目位于 /home/luxliang/wenbiao_zhao/vbench-prompts-compile。
- Python 3.11.14；uv 0.9.17。远端已有 uv 0.11.14，保留不改，专用 uv 安装至 ~/wenbiao_zhao/tools/uv-0.9.17/。
- 基础 sync 只装开发依赖；--extra train 安装冻结训练栈。
- 全部同步须使用 --locked；传递依赖由 uv.lock 固定。

## 执行步骤

1. 检查本地工作区、gh登录、远端Python/uv/driver/GPU，不改动已有工作。
2. 初始化独立 Git 仓库及 private GitHub remote。
3. 写入协议、计划、最小CLI及纯CPU测试；uv lock/sync并测试。
4. Conventional Commit 后 push main。
5. 远端使用SSH clone；已存在时核对remote、clean状态后 pull --ff-only。
6. 安装项目专用uv，uv python install 3.11.14，sync --locked --extra train。
7. 跑测试、训练库import、CUDA小张量smoke；不加载模型，不启动训练。
8. 回填版本、命令、结果与未验证范围，再提交并让远端pull一致SHA。

## 访问与安全

- 初始化使用仓库外 ~/wenbiao_zhao/vbench-prompts-compile-access/id_ed25519，只读 deploy key。
- 通过项目本地 core.sshCommand 选择密钥，不改全局SSH配置。
- GitHub始终使用SSH Git remote，不保存个人token，不提交密钥。
- 不改驱动、不kill GPU进程、不启动常驻服务。

## 专用 uv 安装（仅首次）

已有全局uv可用于获取指定版本，但不替换它：

```bash
mkdir -p ~/wenbiao_zhao/tools/uv-0.9.17
uv pip install --python /usr/bin/python3 --target ~/wenbiao_zhao/tools/uv-0.9.17-package 'uv==0.9.17'
ln -s ~/wenbiao_zhao/tools/uv-0.9.17-package/bin/uv ~/wenbiao_zhao/tools/uv-0.9.17/uv
~/wenbiao_zhao/tools/uv-0.9.17/uv --version
```

Python安装如提示 ~/.local/bin/python3.11 已存在，不用 --force 替换；uv仍可为项目选择已下载的托管3.11.14。以项目 .venv/bin/python --version 验收。

## 回滚

安装和环境仅在新目录及uv缓存内，不替换远端全局uv/Python。同步失败先诊断，不通过解除版本锁、静默降级或换模型绕过。
