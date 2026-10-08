# P1 原料下载与可训练性盘点

本轮授权：五类数据尽量下载标注/提示词，大文件限量；MovieGen作为未来训练候选。不下载图片、视频、模型权重，不调用teacher API，不训练，不自动commit/push。

## 实测盘点

| 来源 | 已获取原料 | 当前可用于什么 | 不能直接声称什么 |
| --- | --- | --- | --- |
| Visual Genome | 108,077图像的2,316,104条关系标注；ZIP 77,904,473 bytes | 保守词表筛选并去重后34,962条结构候选 | 不是34,962条已验证自然prompt→JSON；需模板弱监督或严格文本对齐 |
| Flickr30k Entities | 158,915条caption，559,767次实体短语提及；annotations.zip 29,284,070 bytes | 自然文本与实体短语抽取的原料 | 短语不是已归一化GRiT标签；annotation许可仍需澄清 |
| K400 | 400个唯一类别，7,446 bytes | 锁定UMT类别/ID、覆盖与输出校验 | 自然prompt监督对为0，不是现成动作解析训练集 |
| SNLI | 570,152条原始句对；train 550,152条，其中549,367条标签有效；validation/test各10,000条，约20MB | 通用三分类NLI辅助；完整统计见子报告 | 不是场景专用或真实Tag2Text三分类 |
| teacher 试标（pilot-01） | 19次`deepseek-flash`请求；15条候选、4条quarantine | 三解析任务schema与长度桶验证；Scene三标签探针；候选仅供工程冒烟 | 不是金标准、不能估计准确率、未人工审核 |
| MovieGenBench | VideoBench 1,003行，精确去重998个prompt；两个audio文本文件另贡献527个唯一video_prompt；合计1,525个精确唯一视频prompt | 用户指定的未来复杂prompt训练候选池 | 无我们的四维目标标签；音频文件行数不能直接与video相加当独立prompt |

所有原料在本地 `data/raw/`，Git忽略；来源、revision、SHA256与真实计数记录在各目录manifest/stats。部分SNLI检查复用远端现有pyarrow，不改变包锁或占用GPU。五类下载原料（含来源说明文件）合计128,616,460 bytes，约128.6MB，未下载媒体。

## 子报告

- [Visual Genome / SNLI](vg-snli.md)
- [MovieGen / Flickr30k](moviegen-flickr.md)
- [K400](k400.md)

## 关键结论

1. VG结构候选极不平衡：left376、right220、above12,707、below21,659。直接均匀抽样会几乎没有左右关系；under等词也不保证满足现有box geometry判据，必须再审核。
2. Objects有大量自然标注，但bodyparts、clothing、scene、非视觉短语及同类重复如何处理需先定协议，不直接把所有短语塞入entities。
3. K400词表有固定非字母序ID，必须与现有UMT输出一致；仍缺自然动作表达和未知动作监督。
4. SNLI中premise作为证据、hypothesis作为待验证条件，方向不可倒置；通用标签映射只可作为辅助，不冒充Scene金标准。
5. MovieGen官方许可CC-BY-NC 4.0，已审阅文本未发现明文禁止训练条款；但非商业/归属要求仍需遵守。将其加入训练候选不等于已经生成训练标签。
6. 使用MovieGen prompt训练后，不能再拿整套MovieGenBench当独立泛化测试。先去重、来源划分，再标注；论文披露使用范围。未制作train/dev/test，也未擅自把官方test直接混进train。
7. SNLI训练集中有785条未定标签、547条重复已标注行、64组标签冲突文本对；剔除冲突组后有548,677个无歧义唯一文本对（仍只是NLI原料）。训练/验证还存在1组完全相同的premise+hypothesis。SNLI与Flickr30k存在来源关联，须跨数据源去重及来源隔离，不能各自随机划分后宣称无泄漏。

## 复现

```bash
uv run --no-sync python scripts/inspect_k400.py
uv run --no-sync python scripts/inspect_moviegen_flickr.py --fetch
uv run --no-sync python scripts/inspect_vg_snli.py --download --download-only
uv run --no-sync python scripts/inspect_vg_snli.py --source visual_genome
# SNLI Parquet统计需要已安装pyarrow的训练环境，见子报告；基础开发环境不额外装包。
```

后两脚本网络下载显式开关、源大小上限及检查命令以其--help和子报告为准；默认不会触发媒体或模型下载。已缓存文件校验hash。VG使用作者大学主页而非已变更用途的visualgenome.org域名。

## 本轮验证

- 本地 `uv lock --check`、CLI `--help`、`git diff --check`通过；依赖锁未变。
- 离线单元测试16项通过，包含新增K400、MovieGen/Flickr、VG/SNLI工具测试。
- raw数据被根级 `/data/` 忽略；修正旧 `data/` 规则误伤 `docs/data/` 的问题，报告可被Git跟踪。
- 本轮未commit/push；没有改动相邻audit仓库的冻结研究文件。

## 本轮新增：小批量加工与 teacher 试标

- `scripts/prepare_data.py` 已产出忽略构建 `data/processed/local-0001/`：spatial 400（左右上下各100，另21条多关系）、objects 400（Flickr，许可待核）、action 400（K400模板弱监督）、MovieGen 1,525条无标签清单；`data/smoke/local-0001/` 另存工程fixture（12/12/12/90）。计数闭合，重复运行同输入同哈希。
- teacher 试标按[试标报告](https://github.com/winbeau/vbench-faithful/blob/89396909927e22ee00291ed562ae344a7a2a3ad1/docs/reproduction/semantic-models/docs/teacher-pilot-report.md)执行：19次请求、15候选、4条因 `action_not_in_k400` 被拒；空间5/5、Objects 4/4、Scene 5/5 合法，Action 仅1/5。
- 结论：Action 需要"K400受限提示+人工确认别名"，不能直接把 teacher 自然动作当标签；Objects 需要摄影/后期词汇排除表；Spatial 自然 prompt 的正则线索不等于显式关系，弱监督仍以 VG 结构化边为主。

## 下一步（未执行）

- 核验Flickr annotation再分发/训练许可，建立可用来源清单。
- 冻结跨数据集split与去重规则；审核 teacher 15条候选并统计一致率。
- 冻结空结果/未知类别/非场景对象协议（`Camera` 类误报已记录）。
- 201+词与401+压力桶、真实Tag2Text场景数据仍缺；teacher 预算已用满，扩标需重新授权。
