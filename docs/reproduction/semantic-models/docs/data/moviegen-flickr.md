# MovieGen / Flickr30k Entities 原料审计

## 结论与边界

完整下载下述小型**文本/标注**，不是抽样；未下载图片、视频、音频、权重，未调用 teacher、安装依赖或训练。文件位于已被 `.gitignore` 的 `data/` 排除的 `data/raw/moviegen/` 与 `data/raw/flickr30k/`。没有创建项目训练划分、标签或训练集；原始 `train.txt` 是上游图像 ID 列表，不代表本项目已放行。

- MovieGen：用户希望未来训练使用，登记为 **training candidate, unlabeled**。核验到的官方许可是 CC-BY-NC 4.0，**不是“禁止训练”的 evaluation-only 许可**；没有发现明文 ML 训练禁令。仍需落实非商业用途、归属及改动说明等许可义务，冻结泄漏协议，再进入训练。若训练使用其 prompt 或派生样本，不能再宣称在完整 MovieGen benchmark 上获得独立、未见测试成绩。
- Flickr30k Entities：可用于实体短语原料调查，但官方仓库没有独立 LICENSE，README 对图片明确限定非商业研究/教育，未明确给出独立的标注再分发/训练许可。**不能将“公开可下载”当作宽松授权**；训练、再发布或商业使用前核实标注和原始 caption 权利。未核实部分保持门禁。

## 官方来源与许可依据

1. [MovieGenBench 官方仓库 README（固定版本）](https://github.com/facebookresearch/MovieGenBench/blob/bab7753fce1108def167a1efb4c1ae1d69b8f03a/README.md)，revision `bab7753fce1108def167a1efb4c1ae1d69b8f03a`。
2. [固定版本 LICENSE](https://github.com/facebookresearch/MovieGenBench/blob/bab7753fce1108def167a1efb4c1ae1d69b8f03a/LICENSE)：CC Attribution-NonCommercial 4.0 International；§2(a) 非商业复制/分享/改编，§3 分享时归属、保留声明、标记改动。README 虽写 “model”，[官方 Hugging Face 数据卡](https://huggingface.co/datasets/meta-ai-for-media-research/movie_gen_video_bench) 明确将数据许可链接到同一 LICENSE，并将数据命名为 `test`/`test_with_generations`。数据卡不是禁止训练条款。未下载 HF parquet 或 generations。HF 独立 LICENSE URL 返回 404，依照其数据卡链接核查 GitHub 正文。
3. [Flickr30k Entities 官方 README（固定版本）](https://github.com/BryanPlummer/flickr30k_entities/blob/68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b/README.md)，revision `68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b`。固定版本根目录无 LICENSE；README 要求引用 Plummer et al. (IJCV 2017) 与 Young et al. (TACL 2014)，解释图片权利不归发布者并须遵守 Flickr 条款。[原始 Flickr30k 页面](https://shannon.cs.illinois.edu/DenotationGraph/) 同样载明图片非商业研究/教育说明；不能据此推定所有标注具有 CC/MIT 许可。没有发现明文“禁止所有训练”，但也未获得清晰独立标注许可。

## 下载清单：URL、大小、SHA256

所有 URL 均按 `https://raw.githubusercontent.com/{repo}/{revision}/{path}` 构造；repo 与 revision 见上文，下面的 path 精确对应远端对象。完整 URL、大小、SHA256 同时存于各 raw 目录的 `manifest.json`。不克隆仓库。下载体总量：MovieGen **600,513 bytes**；Flickr **29,630,701 bytes**，各自低于十进制 100 MB。

### MovieGen

| path | bytes | SHA256 |
| --- | ---: | --- |
| README.md | 3801 | `390f8f1955d285296ea7ab3cd1533b7cc28f7aac311e83a2583f8c0d093294a5` |
| LICENSE | 19334 | `7c6858d0e71dc8b858255270ad4b5004e80b5b058bf5bbc408a28430d6ddbf4b` |
| benchmark/MovieGenVideoBench.txt | 106686 | `ae02a4bbebcb55067dbeea369e43d52a4817d57fd0ac223071b9aa236ac027c4` |
| benchmark/MovieGenVideoBenchWithTag.csv | 132624 | `9d893204f271c7db1ed13c6513c8cc6f4630e9bb62de448c6e93d49e37653fbf` |
| benchmark/MovieGenAudioBenchSfx.jsonl | 121743 | `6e86e6ef12bb6c8f0f1410262a626386d4a539b0000075fc73ba345d02e38acc` |
| benchmark/MovieGenAudioBenchSfxMusic.jsonl | 216325 | `e560c60c07094dbff26f2448ae4aefefce8a1ef216fc88d3898a7d539faa2fed` |

### Flickr30k

| path | bytes | SHA256 |
| --- | ---: | --- |
| README.md | 4493 | `7875ef2687442a208b8b1f13a79b17e243fbb10bff3c1cdb15248ea769188977` |
| UNRELATED_CAPTIONS | 303 | `160c7f08433af2a24f762e936c54b276b082f88a8e8aa3a1621ef30a3758d39d` |
| annotations.zip | 29284070 | `1bdde439e41fa936e31e6d72898f1886d49bb4298f2abcdb50771de4f516b026` |
| train.txt | 320344 | `7fac836ec29afdbda7396fe5473c4156908d60d248cae75892f5a94f08016516` |
| test.txt | 10733 | `58c576ebf3af64f346cc3e2ede7eb66523ab172b45c8b42414daa9f2914047e1` |
| val.txt | 10758 | `764e1d2b2bae38bde75f8edda435a7ad20d404e83b77233934844f6a7dbccd9a` |

ZIP 留在原处，不解压写入文件树。统计仅在内存读取 `Sentences/*.txt` 与 `Annotations/*.xml`；其解压文本总大小 **70,965,654 bytes**。

## 实测统计与样例

### MovieGen

- Video TXT：**1,003** 非空行，精确字符串去重后 **998** 个 prompt（5 个重复超额行）。CSV **1,003** 行，列名 `prompt,concept,motion_level`；不是额外 1,003 个独立样本。
- Sfx JSONL：**527** 条，字段 `id,video_prompt,audio_prompt`。
- SfxMusic JSONL：**527** 条，字段增加 `music_prompt`。实测两个audio文件的video_prompt集合完全相同（527个），与VideoBench的998个prompt精确交集为0，跨文件并集为 **1,525** 个精确唯一视频prompt。两种audio任务不可视作1,054个独立视频来源；精确唯一也不保证语义家族独立。
- Video 第一条开头为 `A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.`，对应 `concept="human - activity"`, `motion_level="low"`；完整 prompt 在 raw 文件和 stats 中保留。
- Sfx 第一条结构：

```json
{"id":0,"video_prompt":"A person slowly takes a big bite of a crunchy apple, their teeth sinking into the juicy flesh.","audio_prompt":"juicy crunches of the apple being bitten into and chewed on."}
```

这些是自然语言输入及 benchmark 概念/运动标签，**不是**已对齐 K400 的动作真值、GRiT 对象真值、空间三元组或 Scene 三分类真值。不得用概念列自动冒充这些输出。

### Flickr30k Entities

| 实测项 | 数量 |
| --- | ---: |
| sentence TXT / annotation XML | 31,783 / 31,783 |
| captions | 158,915 |
| 去掉实体标记后精确字符串唯一 captions | 158,439 |
| phrase mentions（出现次数，不是唯一物体） | 559,767 |
| 非零 `(image_id, chain_id)` 对 | 243,891 |
| chain ID 0 mentions | 30,695 |
| XML bndbox 元素 | 275,775 |
| scene=1 object entries | 4,892 |
| nobndbox=1 object entries | 47,932 |
| UNRELATED_CAPTIONS 行数 | 22 |

`UNRELATED_CAPTIONS` 上游明确说列表可能不完整；本轮没有过滤它，统计是原始条数。ID 0/notvisual 按上游应视为各自单例，不能全合成一个实体。上游约数 244k chains / 276k boxes 与上述实际计数口径相容。phrase type assignments（允许一个 phrase 多类型，不能简单相加当 mention 总数）：people 182,924；bodyparts 18,106；scene 86,317；clothing 71,538；other 138,658；notvisual 30,695；instruments 4,900；animals 17,372；vehicles 13,173。

`Sentences/1000092795.txt` 第一行：

```text
[/EN#1/people Two young guys] with [/EN#2/bodyparts shaggy hair] look at [/EN#3/bodyparts their hands] while hanging out in [/EN#8/scene the yard] .
```

对应 XML 包括 `<filename>1000092795.jpg</filename>`、图像尺寸、`<object><name>1</name><bndbox><xmin>159</xmin><ymin>125</ymin><xmax>219</xmax><ymax>335</ymax></bndbox></object>`。同一个 object 可关联多个 name/chain；某些只有 scene/nobndbox 标志。计数不应把短语数、chain 数、框数混为一谈。

上游 split：train **29,783** 图像，val **1,000**，test **1,000**；各自 ID 唯一，两两交集为 0；并集与全部 sentence ID 相同；sentence/XML ID 一致。没有将上游 train 自动转成本项目训练集。

## 后续加工与泄漏门禁

1. MovieGen 原始重复文本先聚类，CSV/TXT 同一 prompt 不拆到不同 split；audio 的相同视频家族及跨 video/audio 重复也需联合检查。保持原始行号/ID、revision、文本哈希和派生家族键。已完成上述文件间精确文本交集/并集统计，尚未完成语义近重复或与VBench的相似文本比对。
2. MovieGen 未来若训练使用：先声明放弃该部分及其近似/改写版本的独立测试身份。单独划出的 held-out 家族才可能构成项目内测试，不能报告为未污染的原始全量 benchmark。非商业许可不等于所有商业下游部署已获许可。
3. Flickr 按 image ID 而不是 caption/phrase 随机分割；五条 caption、实体 chain、框及所有改写绑定同一 split。精确重复 caption 也需跨图像聚类或删除跨 split 重复；本轮原始 split 交集检查只证明 image ID 不重叠。
4. **SNLI 源自 Flickr30k captions**：不同下载来源不等于独立数据。若 Scene 用 SNLI、Objects 用 Flickr，需结合 SNLI 的来源图像/句子 ID 统一分组、查重；未完成跨源审计前不能宣称独立测试。
5. Flickr 的 people/clothing/other 等是粗粒度类型，实体短语包含属性、复数、身体部位、场景与不可视内容，**并非全部是 canonical GRiT labels**。对象映射需固定检测器版本/词表、保留不支持与歧义、人工审核并记录证据；不能把图中有但 prompt 未提及的对象加为目标。框关系也不能冒充文本明确表达的空间关系。
6. 原始可用条数不等于监督条数：尚未作任务资格筛选、许可放行、目标标注、人工审核，当前可宣称的已验收训练样本数为 **0**。没有根据媒体或真实 Tag2Text 生成 Scene 标签。

## 复现与验证

```bash
# 工作目录：vbench-prompts-compile；已存在 Python 3.11.14 环境
.venv/bin/python scripts/inspect_moviegen_flickr.py --fetch
.venv/bin/python scripts/inspect_moviegen_flickr.py  # 离线重算，检查缓存大小和既有 manifest SHA256
.venv/bin/python scripts/inspect_moviegen_flickr.py --help
```

标准库脚本固定 revision 和文件大小白名单，每源总预算 100,000,000 bytes；不使用 HF 自动下载器，不跟随 README 中媒体链接。不完整/大小异常下载报错而不写入最终文件；首次抓取计算 SHA256，后续对照现存 manifest 校验（首次哈希仍需和本文清单核对）。输出 `manifest.json` 和 `stats.json` 均在 ignored raw 目录；离线重算不会覆盖原始下载文件。

实测 Python 3.11.14；全量下载及统计成功，未抽样；离线重算成功；现有环境 `python -m pytest` **12 passed**（包括新增的 3 个离线合约测试与并行任务已有测试）；项目 CLI `--help`、此脚本 `--help` 和 `git diff --check` 通过。测试覆盖源预算、固定版本、缓存篡改拒绝、离线缺文件及合成 prompt 计数，不读取真实数据。未验证 GPU、真实模型或训练。新增本脚本、本报告及 `tests/test_moviegen_flickr.py`，未更新其他 docs/scripts。
