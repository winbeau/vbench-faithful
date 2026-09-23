# 容器清空后的恢复与评分

本入口固定论文当前选定的九维修复，并提供原版 VBench 1.0 的全部十六维。
配置见 [`paper-methods.json`](../../configs/reproduction/paper-methods.json)。九维以外不输出虚构的 Repair 分数。
旧的分维度研发 CLI 保留历史默认值；论文评测使用下列统一入口。
实际完成的检查与按用户要求停止的部分见 [容器重置交接记录](container-reset/README.md)。

## 从新环境恢复

需要私有 GitHub 仓库权限、Linux amd64、Ubuntu 24.04 兼容系统库和可用 NVIDIA 驱动。
CUDA 运行库、Detectron2 编译产物、Python 3.10.20 视觉环境、Python 3.11.14 语义环境均已归档。
恢复目录必须在允许执行程序的文件系统上，不能放在挂载为 `noexec` 的 `/dev/shm`。
建议为下载缓存、解包目录、Qwen 基座和测试产物预留 100 GB。

```bash
git clone -b winbeau git@github.com:winbeau/vbench-repair.git
cd vbench-repair
# Ubuntu 24.04；已有这些系统依赖时可跳过。
sudo apt-get update
sudo apt-get install -y python3-venv git ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libgomp1
python3 -m venv output/bootstrap
output/bootstrap/bin/pip install 'huggingface_hub==1.32.0' requests
output/bootstrap/bin/python scripts/restore_paper_runtime.py \
  --output /absolute/path/vbench-runtime \
  --downloads /absolute/path/vbench-downloads \
  --allow-official-fallback
```

所有 HF 读取先尝试 `hf-mirror.com`；只有失败后才使用允许的官方回退。
大归档按 128 MiB 分片下载，逐片及重组后均核对 SHA256；已下载且哈希正确的文件可复用。
若解包后下载模型中断，在同一命令后加 `--resume` 续跑；仅接受本脚本创建且 manifest/模式相同的目录。
[`runtime-release.json`](../../configs/reproduction/runtime-release.json) 固定每份归档及分片的 HF revision。
恢复不会读取旧容器的 `/root/wenbiao_zhao`，也不需要旧的 Python 环境。

脚本恢复所有评分依赖、六个选定 LoRA、Dynamic aligned-v1 评分头与 V-JEPA backbone，
并从固定的 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` 下载冻结基座。
Qwen 的十二个必需文件另有完整哈希清单。数据和模型的原发布 revision 保持固定，
后续增加运行环境归档不改变原论文数据。

公开运行环境只包含第三方依赖、其必要源码和模型资产；私有项目源码仍在 GitHub。
归档已去掉旧项目的 editable import hook、本地安装来源 URL、Git remote 配置和 Python 字节码。
VBench 1.0 源码采用 235 个原始文件的逐文件固定清单，不依赖 Git 历史；
额外文件、缺失文件或源码修改都会拒绝加载。账户令牌、SSH 密钥及其他项目均不在恢复包中。

## 安装验收

选择一张有足够空闲显存的 GPU；Qwen3-8B 推理约需 20–24 GB 空闲显存，视觉与语义进程依次运行。

```bash
CUDA_VISIBLE_DEVICES=1 output/bootstrap/bin/python scripts/verify_paper_install.py \
  --assets /absolute/path/vbench-runtime/assets.json \
  --output output/install-check
```

该命令重新解码视频、运行模型并核对冻结分数：九维的 19 个原片／反事实输入，以及十六维各一条官方输入。
其中 Dynamic 包含原片和两个局部纹理抖动反事实。`verification.json` 记录每条误差、每维覆盖率和是否通过。
Repair 允许误差为 `1e-6`。这是代表性端到端验收，不能称为全论文样本 GPU 重跑或从头训练验收。

完整九维主表仍可独立复算：

```bash
/absolute/path/vbench-runtime/semantic-env/bin/python scripts/reproduce_main_table.py \
  --bundle /absolute/path/vbench-runtime/selected/bundle \
  --model-code vendor/vbench_prompts_compile \
  --k400-labels configs/reproduction/k400-labels.json \
  --output output/paper-table
```

该复算使用全部冻结视觉证据与语义预测，核对 24,408 个分数单元；不把缓存复算冒充视觉模型重推理。
逐项相等验收使用项目锁定的 Python 3.11.14；不要使用 Ubuntu 24.04 默认的 Python 3.12 bootstrap 解释器复算。
一次额外的 H200/Python 3.12 重放触发了严格相等校验；未放宽校验或改写冻结结果。

## 评测自己的 VBench 视频

标准 VBench 命名为 `prompt-0.mp4` 到 `prompt-4.mp4`。先生成带维度和官方辅助标注的输入：

```bash
python3 scripts/prepare_vbench_inputs.py --video-dir /absolute/path/generated-videos \
  --output output/my-inputs.json
# 原版全部 16 维
CUDA_VISIBLE_DEVICES=1 python3 scripts/evaluate_vbench.py --input output/my-inputs.json \
  --assets /absolute/path/vbench-runtime/assets.json --output output/my-origin \
  --backend origin --dimensions all
# 论文当前采用的 9 维修复
CUDA_VISIBLE_DEVICES=1 python3 scripts/evaluate_vbench.py --input output/my-inputs.json \
  --assets /absolute/path/vbench-runtime/assets.json --output output/my-repair \
  --backend repair --dimensions paper
```

不完整的标准套件默认拒绝；有意做局部评测时显式用 `--allow-missing`，并用 `--dimensions` 指定待测维度。
自定义输入可直接使用 JSON/JSONL，逐条提供 `id`、`video`、`prompt`、`dimensions`，
以及原版所需的 `auxiliary_info`。Subject Repair 还需明确的 `subject_en`；标准套件使用冻结主体标注，
自定义输入缺失时会报错，不从整段 prompt 猜测主体。

```json
{"id":"sample-1","video":"clip.mp4","prompt":"a white car","dimensions":["color"],"auxiliary_info":{"color":{"color":"white","object":"car"}}}
```

`--backend both` 仅适用于论文九维，可在同一输入上同时报告两种后端。
Origin 使用上游各维度的原生聚合；例如 MUSIQ 的逐视频原始值为 0–100，而官方汇总为 0–1。
Human Action 的原版按文件名提取动作，标准视频命名必须保留；Repair 使用显式 prompt。
失败、遗漏或不支持的输入保留为 `null`，覆盖率不足时不输出伪完整汇总。
输出必须为新目录，冻结的 `data/`、`results/`、`splits/`、`runs/` 不可覆盖。

## 方法边界与训练

Dynamic aligned-v1 保留论文的原生 16 帧、8 FPS、正方形 RGB 协议；不自动裁切、补帧或重新标定分数。
其他长度、帧率或缺少时长信息的 GIF 不能据本验收宣称具有同等 Repair 效果，原版 Dynamic 仍可按原协议运行。
Spatial 使用有符号几何，Action 使用 v2.1 语义接口，Objects 使用相邻帧确认；
Subject 固定 hybrid/exclude/preencode_crop，Background 固定 patch_frame_calibrated。

“最优”指本论文已选择并获用户确认的版本。Action 为保留下来的最佳 checkpoint；
Object Class、Color 使用论文 step 300，未经 dev 选优；Dynamic 为用户接受的 aligned-v1 固定末步 300，亦未经 dev 选优。
早先的失败结果和小样本局限继续保留，见 [废弃实验索引](REJECTED_EXPERIMENTS.md)与
[最新 Dynamic 方法归档](../paper/dynamic-degree-current.md)。
从头训练入口、数据、初始化链和配置见 [复现总说明](README.md)及 [Dynamic 训练链](DYNAMIC_ALIGNED.md)。
恢复既有论文方法不需要重新调用教师模型；本次容器恢复验收不等于已重训六个 LoRA 和 Dynamic 的全部阶段。
