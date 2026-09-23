# Dynamic Degree：外部生成器128-prompt冻结泛化评估

日期：2026-09-23。状态：**PAUSED_USER_REDIRECT，未完成评分，不报告总体结果**。用户改为优先测试权威数据集中的真实运动/真实静态片段；长生成视频不再继续扩展。本报告独立于既有450组VBench结果，不覆盖原始报告或主表。

## 目的与冻结边界

用户要求在Deep Forcing、Causal Forcing的128-prompt视频或其他数据上检验泛化。只评价当前`aligned-v1`，不训练、不选检查点、不调分、不修改公开CLI默认。头SHA-256为`6dcfbdce2e604cdd669e54015fee34e034c8004726d06c46611532f451ca8a53`。

本次可以检验“新生成器/新内容上对同类8px局部纹理抖动的稳健性”，不能仅凭不变性实验宣称通用运动理解、未知扰动泛化或真实运动强度准确。没有这些视频的人类运动偏好标签，此项为**NOT RUN**。

## 数据盘点

- Deep Forcing：H200已有完整30秒标签的128条MP4及`prompts.csv`，编号000–127。来源目录`/data/chenjiayu/wenbiao_zhao/Pyramid-Forcing-Experiments/deep-forcing-30s/`。视频实测16fps、832×480，首例489帧、30.5625秒；不把目录名视为精确片长。
- Causal Forcing：已从用户提供的`kv-compression/Pyramid-Forcing-Experiments`下载128条MP4及提示词到H100独立目录`/root/wenbiao_zhao/tmp/dynamic-generalization-forcing128-20260923-download/`。固定revision为`81162a0668747e07701bc08bbfb8e4aeb0e34a9c`，129文件共4,877,529,365字节，逐文件SHA记录于`completion.json`。转移到H200尚未完成，按用户转向停止；部分输出保留。**Causal构造/评分NOT RUN**，未新生成视频。
- 模型名称来自既有用户实验目录；没有从目录名推断生成checkpoint/源码SHA。缺失生成身份如实记录为NOT VERIFIED。
- 已核对Deep的128条完整来源SHA、prompt规范化文本；与训练/开发/校准预留prompt的字面交集为0，与270个已用TRAIN/DEV原片的字节交集为0。未打开45校准预留视频；文本去重不等于人工语义去重。

参考方法来源：[Deep Forcing官方仓库](https://github.com/cvlab-kaist/DeepForcing)、[Causal Forcing官方仓库](https://github.com/thu-ml/Causal-Forcing)。这些链接说明方法身份，不证明本地视频必然由某个官方checkpoint生成。

## 评分前锁定协议

配置：[forcing128-v1.json](../../configs/dynamic-generalization/forcing128-v1.json)，SHA-256 `66db3814e39a55bf036ce9b20b4bf3d44864747d48880fcf1ea3b5d2730e679f`。

1. 每条源视频固定取开头、中间、末尾三个窗口；在VBench原采样网格上各取16帧、8fps。保留源帧索引与时间，不插帧，不把整段30秒压进2秒。
2. 原始空间分辨率不变；在各窗口原生坐标构造同一8px干预、两个固定种子1701/2904。非RGB加噪、非亮度频闪；几何警示全部保留，不据评分筛选。
3. 每窗口原片、无改动编码控制、两个反事实，共4输入；每个128源群体384窗口、768CF、384编码控制，**1536输入/后端**。
4. 原片窗口与反事实均用RGB无损编码，逐条核验像素和时间。Origin重新运行固定官方RAFT与原二值判据，**不是原450组分数复用**。
5. Repair保持全帧双线性抗锯齿缩放384²、冻结V-JEPA与现有连续头。仅实验适配器允许矩形输入；先验证方形TRAIN参考的预处理、特征哈希和头输出与旧缓存一致。原适配器源码不修改。
6. 先在源内平均三个窗口与两个种子，再对128源等权。报告Δ、逐窗口/种子MAE、按prompt聚类20,000次bootstrap的95%CI，以及逐条大变化、分数分布和Origin饱和情况。
7. 结果名称为**windowed Origin/Repair**：Origin均分是采样窗口的动态比例，不冒称官方整段30秒视频得分。若Origin本来已全为1，不用“未继续增分”宣称Repair修复成功。

新视频短边480px，因此8px相对幅度小于此前短边256px的数据；不能忽略这一跨分辨率差异。固定窗口实验也不验证30秒级长期轨迹理解。

## 运行与复现

H200独立输出根：`/data/chenjiayu/dynamic-generalization-forcing128-20260923/`。不改源视频和历史实验目录。运行脚本：[evaluate_dynamic_generalization.py](../../scripts/counterfactual/evaluate_dynamic_generalization.py)，子命令`select/build/score/summarize`。

阶段输入/输出：`selection/sources.jsonl`与`selection.json` → `construction/shard-*/inputs.jsonl`及无损媒体 → `scores/shard-*/scores.jsonl`与provenance → `analysis/pairs.jsonl`、`summary.json`。所有输出使用新目录，覆盖失败与不完整状态明确保留。

本地新增及相关回归测试：`tests/test_dynamic_generalization.py tests/test_vjepa_aligned.py tests/test_vjepa_aligned450.py tests/test_local_texture_jitter.py tests/test_local_texture_stress.py`，**40 passed**；四个新增脚本编译及`git diff --check`通过。

## 用户转向时的保留状态

- Deep首轮前台构造因连接中断保留132条输入；随后独立`construction-retry1`四分片完成1536/1536，零构造失败。两份产物不混合。
- GPU评分已自动开始，停止时`scores/shard-0/scores.jsonl`与`shard-1/scores.jsonl`分别保留58、63条，即121/1536。没有完整汇总，不从这部分样本推断总体效果。
- H200任务控制进程组3283815和本任务传输进程3298254已发送SIGTERM并确认停止，原片、中间视频、下载和部分评分全部保留。终止原因是用户改变研究重点，不记为模型运行失败。
- TRAIN参考检查确认方形预处理与旧实现逐位一致、V-JEPA特征NPY哈希一致；冻结头logit误差`5.960464477539063e-08`。这只验证模型加载，不代表外部泛化通过。
- 后续真实运动/静态数据实验另建协议与报告，不修改本协议或把它标为完成。
