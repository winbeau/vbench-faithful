# 四维三方案矩阵：已完成

本轮用户授权的 VBench 1.0 原始/变换实验已跑完并验收。最终口径见 [研究报告](deterministic-experiments-report.md)、[全部表格](deterministic/matrix-v1/README.md)、[资源与身份](deterministic/matrix-v1/provenance.json)。本文件覆盖早期状态快照的活任务/待授权描述；历史过程与失败日志保留。

## 已完成的实测产物

- 冻结矩阵 29,660 项 → Origin / Repair-rule / Repair-model 共 88,980 条配对评分、81 个汇总条件；scorer `execution_complete=true`，全部 13 个评分输入哈希在本地副本核对一致。
- 原始视觉缓存 4,520 视频；Spatial 变换 2,600 项全部尝试、1,960 适用项均完整。Objects 11,760 项全部尝试，4,716 完整、7,044 部分有效，缺失始终保留。
- Spatial v8 固定 600 步、Scene v8 300 步、Action v9 来源隔离后 300 步；Objects 复用 v6 900 步。全部 final 权重等于最后 checkpoint，底座冻结、adapter 改变，无 dev/test 选模型。
- Scene test2 492 观测、5 连通来源块；同义补充普查 3,200 caption 对、10 来源家族；ocean→sea dev 20 视频/1 家族均完成。API 标签不称人工金标。
- Objects 980 端点保留：505 项模型复核、475 几何不完整；275 项全帧目标不可见、142 严格隔离移除，均 human_reviewed=false。8 个分片 exit 0；1,230 请求包含 8 次失败，不清零账本。
- 11 张附表 A–K 与紧凑主表提供 CSV/Markdown/LaTeX。实际 PDF 为 2 页主表、14 页完整附录，无 LaTeX 溢出/警告。11 个真实帧案例已核验来源/变换/UMT 输入哈希。
- 本地完整测试 **151 passed**；CLI help、`uv lock --check`、`git diff --check` 通过。本轮不存在训练、缓存或标注待跑项。

## 结论界限

Scene 对模型标注 caption 的判别改善；Spatial 官方几何忽略方向符号，文本修复不能解决；Action 模型同义仅 25/60 正确，35/60 输出 other；Objects 三方案逐项相同，275 个不可见端点中 246 个全帧拒绝、29 个仍有正帧。严格隔离且正基例 114/115 下降至少 0.03。任何均分升高都不自动表示视频质量提升。

ocean→sea 不移入 test；OOV 仅 1 个唯一短语，无 CI；test2 的 60 条无官方 scene key 候选不是全部已确认的无场景金标。遮挡缺失、弱覆盖和模型复核局限详见最终报告。

## 运行与恢复身份

训练授权：rtx4090 GPU 5（UUID `GPU-70f0792b-3dcf-6b1b-8c58-bffaf406353a`）。视觉缓存授权：H100 GPU 4（UUID `GPU-3af22086-0ac4-4277-1486-96909b440d1c`）。本轮 GPU 工作已结束，不占用其他卡、不终止他人任务。

最终评分成功作业：H100 `output/jobs/final-matrix-v1-retry2/`，merge / score / controls / outer 均 exit 0，评分代码 commit `4d2a78d`，代码哈希校验通过。

首轮 `final-matrix-v1` 因失败请求 journal 没有 response 字段中止；新增失败/重试资源汇总回归修复。`retry1` 因新 checkout 缺 K400 词表中止；补齐本地冻结文件并核对原清单 SHA 后，`retry2` 成功。两次恢复均未重复 API 或模型推理，原日志保留。

rtx4090 项目 `/data1/wenbiao_zhao/vbench-prompts-compile`；H100 `/root/wenbiao_zhao/vbench-prompts-compile-git`，data/output 仍链接旧产物目录。项目 Python 3.11.14 / uv 0.9.17；上游视觉环境继续用 Python 3.10.20 / torch 2.5.1+cu121。H100 通过 Git bundle 执行 `pull --ff-only`；RTX 原有 `D scripts/score_deterministic.py` 保留，不 reset。

相邻 vbench-audit 冻结 data/results/splits/runs 只读，E0 split SHA 仍为 `831bed0aee880cb0c501ac89fe4cbe4e1e97422427619e24983a9e86a5d3c328`。模型/数据/缓存/预测/图片不进 Git，只提交代码、汇总表和报告。
