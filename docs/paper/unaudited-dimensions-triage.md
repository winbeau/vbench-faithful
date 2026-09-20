# 未审计维度候选：字符串匹配与命名错配（源码定位）

范围：VBench 1.0 共 16 维（`evaluate.sh:7`、`vbench/__init__.py:19`）。原 8 维审计集为
`subject_consistency`、`scene`、`human_action`、`spatial_relationship`、`multiple_objects`、
`dynamic_degree`、`motion_smoothness`、`overall_consistency`；本笔记覆盖**其余 8 维**：
`background_consistency`、`temporal_flickering`、`aesthetic_quality`、`imaging_quality`、
`object_class`、`color`、`temporal_style`、`appearance_style`。

**2026-09-15 范围决定**（[`../plans/2026-09-15-dimension-scope-11d.md`](../plans/2026-09-15-dimension-scope-11d.md)）：
下表前 4 维（`background_consistency`、`temporal_style`、`object_class`、`color`）**已入选候选**，
与在办的 7 维合计 11 维；`overall_consistency` 退出本轮范围；其余 4 维不入本轮。
入选 4 维的源码已在 `configs/upstream.toml` 钉住（sha256 见 §4）。

出处：锁定上游 `Vchitect/VBench @ fd18b3d`（`configs/upstream.toml`）。本机 checkout 的
`HEAD == origin/master == fd18b3d`，工作树干净；行号均针对该 revision。该 revision 之后只有
dev 分支动过这些文件（`origin/dev/torch2` `4427aa8`、`origin/dev/videoformat` `c2064c5` +
`4427aa8`、`origin/dev/vbench2{,-refactor}` `45c0583`），内容是 antialias/preprocessing/重构，
**没有**改动本文描述的标签匹配或命名问题。

未验证范围：本机无权重、无可用 CUDA，因此下面的"字符串漂移率""截断影响"是**设计层事实**
（比较运算符、索引、tokenizer 上限）+ 从冻结 prompt suite 重新统计的数值，不是模型实测率。
真实 GRiT/ViCLIP 输出需要 GPU 与权重才能测量。

## 0. 继续审计的推荐表（问题 / repair / 星级）

星级口径：① 缺陷是否已由源码确定（不需先做实验）；② 反事实族能否复用已有资产（官方视频、
已锁权重、现有 counterfactual 工具链）；③ 与本文主线（严格字符串匹配、命名与实现不同）的契合度；
④ 边际成本（新模型/新依赖/重新生成视频）。

| 维度 | 问题（简短） | repair 建议 | 推荐 |
|---|---|---|---|
| `background_consistency` | **命名与实现不同**：整帧 CLIP 相似度，无背景分解；`(prev,first)` 锚点 + 全局帧均值与 `subject_consistency` 逐条同构 | 背景掩码或前景抑制后再编码；锚点改 prev-only / 滑动窗口；直接复用 subject 的 `temporal_relocation`+median-box 契约（backbone 换 CLIP ViT-B/32，H100 已有本地权重） | ★★★★★ |
| `temporal_style` | **命名与实现不同**（最硬）：与 `overall_consistency` 逐行同一份代码；只读整条 prompt；风格子句被 ViCLIP 32-token 截断 | 切出风格子句单独编码（学 `appearance_style` 的 aux 口径），内容项/风格项分开报；同 base 的 10 个风格改做相对判别（softmax/ranking）；风格子句前置或分段编码，并显式记录被截断的 prompt | ★★★★★ |
| `object_class` | **严格字符串匹配**：GRiT 生成类名与目标串做 `set` 精确成员，无归一，per-instance 分数被丢 | 保留全部实例与分数；标签走语义对齐（同义/复数/词表 + 不支持显式标注）；分母固定为全部帧，"未检出"与"叫法不同"分开报 | ★★★★☆ |
| `color` | **严格字符串匹配 + 实例错位**：`[2][0]` 只比 top-1 类名、颜色用子串、12 色白名单、条件分母 | 修 `[2][0]→[2][i]`（或按 box/IoU 绑定 caption↔实例）；颜色词拼写/同义归一；全帧为主分母、条件率降为诊断；对象未检出不再整条丢视频 | ★★★★☆ |
| `temporal_flickering` | **命名与实现不同**：全局像素 MAE = 运动能量，不是局部高频闪烁；静态前提只由 prompt suite 保证 | 运动补偿或只对高频残差打分（可与 motion_smoothness 共用 flow）；报 fps/分辨率对照，非静态输入拒绝出分；修 `calculate_mae` 的 `None` 与只捕 `AssertionError` | ★★★☆☆ |
| `appearance_style` | 命名基本成立：报 `logits/100` 而非论文的归一化 cos；**单卡与多卡聚合不同**（多进程用末帧 `cur_sim`） | 用归一化 cosine 或显式 `logit_scale`；多进程改用 per-video 均值；补单/多卡一致性测试（这是可复现性缺陷，最容易做） | ★★★☆☆ |
| `aesthetic_quality` | 无字符串、命名基本成立：视频加权 vs 论文"all frames"；LAION 代理效度 | 固定并报告聚合口径；分辨率/编码扰动不变性作为诊断；说明代理训练分布 | ★★☆☆☆ |
| `imaging_quality` | 无字符串、命名基本成立：per-video 记录 0–100 而总分 0–1；预处理模式改变分数 | 统一量纲；把 preprocessing mode 写入 provenance；分辨率/压缩扰动契约 | ★★☆☆☆ |

**推荐组合**：只加 2 维 → `background_consistency` + `temporal_style`（一个补
"命名错配"主线，一个成本近乎为零且证据是源码级同一）；能加 4 维 → 再加
`object_class` + `color`，把 GRiT 严格字符串族从 2 维补到 4 维（`multiple_objects`、
`spatial_relationship` 已审计）。`aesthetic_quality`/`imaging_quality` 建议只作为
"命名基本成立"的对照写进讨论，不单独开审计线。

## 1. 字符串匹配问题

### 1.1 `object_class`：集合精确成员

- `vbench/object_class.py:32`：每帧 `set(model.run_caption_tensor(frame)[0][0][2])`。
  `[2]` 是 `det_obj.data`，即该帧**全部**检测到的类名列表
  （`vbench/third_party/grit_src/image_dense_captions.py:57-65`，第 60 行取 `det_obj.data`）；
  `set(...)` 丢掉 per-instance 分数。
- `vbench/object_class.py:40`：`if key_info in pred`，`pred` 是 `set`，因此是**精确字符串相等**
  （哈希成员），不是子串、不区分大小写、不做单复数/同义词/上下位归一。
- 目标串来自 `vbench/VBench_full_info.json` 的 COCO 风格标签（79 条，含 `tv`、`couch`、
  `potted plant`、`hair drier`、`remote`、`sports ball`、`dining table` 等长尾写法）；
  被比较的另一侧是 GRiT ObjectDet 文本解码器**逐 token 生成后 decode** 的自由文本
  （`vbench/third_party/grit_src/grit/modeling/roi_heads/grit_roi_heads.py:296-303`；beam 路径
  `:259-289`），检测阈值 0.5（`image_dense_captions.py:88`），随后 NMS/top-k。
- 后果：帧被判 0 有两种不可区分的原因——"没生成"和"生成了但叫法不同"；`try/except` 把
  GRiT 失败也变成空集（`object_class.py:33-34`），于是**检测失败 = 类别失败**（与
  `spatial_relationship` 的"无配对记 0"同一族）。
- 该维度与已审计的 `multiple_objects`（`multiple_objects.py:33,40-45` 同样 `set` + `in`）、
  `spatial_relationship` 共享同一套 GRiT 标签串契约，因此语义适配器（见
  `docs/semantic-adapter-constraints.md` §3.4）应把 `object_class` 与 `color` 一并覆盖。

### 1.2 `color`：四种字符串操作叠加 + 标签与实例错位

- `vbench/color.py:37`：`cur_pred.append([cap_det[0], cap_det[2][0]])`。`cap_det[2]` 是该帧
  全部类名列表，取 `[0]` 意味着**每一段 caption 都被贴上"第 0 个（最高分）检测"的类名**。
  DenseCap 与 ObjectDet 两路 head 在同一组 boxes 上按同一顺序解码
  （`grit_roi_heads.py:254-303`），所以第 i 段 caption 自己的类名是 `[2][i]`；用 `[0]` 就
  把第 i 段文本归给了 top-1 实例。目标物体不是该帧 top-1 时，这一帧根本不会被计入
  （分母缩小而不是记失败）。
- `vbench/color.py:46`：`object_key == pred[1]`——对象身份用**精确相等**（与 `object_class`
  同一约定）。
- `vbench/color.py:50`：`if color_key in pred[0]`——颜色用**子串**匹配 dense caption：
  `red` ⊂ `colored`/`hundred`，`blue` ⊂ `blueprint`，`white` ⊂ `whiteboard`，存在系统性假阳性。
- `vbench/color.py:47-49`：分母的"物体出现"判据是**硬编码 12 个颜色词**
  `["white","red","pink","blue","silver","purple","orange","green","gray","yellow","black","grey"]`
  的子串命中；`crimson`/`navy`/`maroon` 等写法直接把该帧踢出分母，而 `gray`/`grey` 两种拼写
  只出现在这个列表里，目标 `color_key` 只有 metadata 里那一种拼写。
- `vbench/color.py:65-67`：对象名由 prompt 字符串手术得到
  （`prompt.replace('a ','').replace('an ','').replace(color_info,'')`）。对现有 85 条 prompt
  我重新算过：全部正确解析为 10 个单词对象（`bicycle,bird,bowl,car,cat,chair,clock,suitcase,
  umbrella,vase`），**当前 suite 不触发**；但它不是解析器——任何以 `a` 结尾并后接空格的词都会被
  破坏（`a red pizza box` → `pizz box`），颜色串在 prompt 中任意位置被删也可能命中对象词。
- `vbench/color.py:81-89`：整条视频若从未匹配到对象类名则被**丢弃**（`if cur_object>0`），
  因此总分是"在目标物体被检出"条件下的均值，分母不是全样本。论文 App. G.2 同时写了
  "in all frames" 与 "Among the frames where …" 两种口径，代码实现的是后者。

### 1.3 共享管线：文件名 ↔ prompt 精确匹配（16 维共有）

- `vbench/__init__.py:151-161`：`f'{prompt}{special_str}-{i}{postfix}'` 与 `os.listdir` 做
  **精确文件名相等**；缺失只打印 `WARNING` 并静默缩短 `video_list`，不报错。默认
  `vbench_standard` 模式下 16 维全部走这条路径。
- `vbench/__init__.py:128-136`（`vbench_category`）：`Path(filename).stem.startswith(prompt)`
  ——前缀匹配。
- `vbench/utils.py:375-384`（`custom_input`）：`re.sub(r'-\d+$', '', stem)` 反解 prompt。
- `vbench/utils.py:226-233`：`dimension in prompt_dict['dimension']`、`dimension in
  prompt_dict['auxiliary_info']` 都是列表/字典键的精确成员判定；`background_consistency`
  因此拿不到 `scene` 的 auxiliary_info。
- `evaluate.sh:7-10` 的 dimensions↔folders 平行数组固化了共享视频关系：
  `background_consistency → scene/`、`aesthetic_quality|imaging_quality|overall_consistency →
  overall_consistency/`、`motion_smoothness|dynamic_degree → subject_consistency/`
  （`dimension_to_folder.json` 同）。

### 1.4 没有目标字符串匹配的维度

- `background_consistency`、`temporal_flickering`、`aesthetic_quality`、`imaging_quality`：
  纯像素/特征统计，**完全不读 prompt**（prompt suite 只用于选视频集合）。
- `temporal_style`、`appearance_style`：把原始字符串直接送进 ViCLIP/CLIP 的文本编码器，
  是 embedding 层软匹配，不是标签比较；但两者都有下面的字符串层面问题（风格子句截断、
  无模板、CLIP logits 标度）。

## 2. 命名错配

### 2.1 `background_consistency`：`subject_consistency` 的镜像，没有背景分解

- `vbench/background_consistency.py:39` 用 `load_video(video_path)`，无 `num_frames`，
  因此取**全部帧**（`utils.py:154-160`）；`:42-43` 用 CLIP 编码**整帧**。模块内没有任何
  crop/mask/box 逻辑，前景主体必然进入特征。
- `:49-51` 与 `subject_consistency.py:58-60` 是同一个
  `(cos(prev,cur) + cos(first,cur))/2`、同样 `max(0,·)` 截断、同样把**每一帧锚到第一帧**；
  `:64` 同样是全局帧加权均值 `sim/cnt`。也就是说，已审计出的 subject_consistency 缺陷
  （首帧锚点 + 变更位置依赖 + 全局均值稀释）**逐条适用于 background_consistency**，
  差别只有 backbone（CLIP vs DINO）与 prompt suite（scene 的 86 条，与 `scene` 共用视频）。
- 论文 App. G.1 的 Eq. 2 与正文"Beyond the focus on the foreground subject"是**解释**，
  代码里没有实现"foreground/background"的分离；Fig. A9 的分数也只是整帧一致性。

### 2.2 `temporal_flickering`：测的是运动能量，不是闪烁

- `:16-27` 用 cv2 读全部帧，`:30-42` 相邻帧全局 MAE，`:45-49` `(255 - mean)/255`。
- 没有运动补偿，因此得分是"整体像素变化率"：平滑快速 pan 与静态闪烁画面无法区分；论文点名的
  "local and high-frequency details" 被全局像素平均按分辨率稀释。
- 前提不写在度量里：`:46` 的 docstring 是 `please ensure the video is static`，论文 App. G.1
  用"静态 prompt suite + Fig. A11 跨动态度排名稳定"来辩护——即**前提由数据保证，不由度量保证**。
  一旦在非静态视频上调用，该维度与 `dynamic_degree`/`motion_smoothness` 同族地纠缠采样率
  （相邻帧 MAE 随帧间隔增大）与分辨率。
- 附带健壮性问题：`calculate_mae` 形状不一致时 `return None`（`:37-42`），使 `np.array(ssds)`
  变 object 数组、`np.mean` 抛错，而 `:58-59` 只捕获 `AssertionError`。

### 2.3 `temporal_style`：与 `overall_consistency` 是同一份代码

- `diff vbench/temporal_style.py vbench/overall_consistency.py` 只差函数名与
  `load_dimension_info` 的 dimension 串；函数体逐行相同。因此该维度实际是"在
  temporal_style prompt suite 上再算一次 overall consistency"。
- `temporal_style.py:48`：`query = info['prompt']`——**整条 prompt**（= 内容子句 + ", " +
  风格子句），`:59-61` 用 ViCLIP 文本特征与视频特征算相似度。风格标签从未被读取，
  `VBench_full_info.json` 里 `temporal_style` 也**没有** auxiliary_info（与
  `appearance_style` 不同）。
- suite 结构是 10 个 base 内容 × 10 个风格子句（每个子句各 10 条）：同一 base 的 10 个变体
  共享同一段内容描述，内容项主导相似度，度量无法把"内容生成了"与"风格跟上了"分开，
  也没有与其他 9 个风格的对照。
- 文本截断正好切在风格子句上：ViCLIP `max_txt_l = 32`（`viclip.py:25,152`）且
  `truncate=True`（`viclip_text.py:122,151-152`，截断**尾部**并把末位换成 eot）。
  最长的 prompt 恰好是长风格子句那几条：31 词（`…, featuring a steady and smooth
  perspective`）、30 词（`…, with an intense shaking effect`）、29 词（`…, in super slow
  motion`）；31 词加 2 个特殊 token 已 ≥33 > 32，**必定被截断**，风格子句在文本侧不可见
  （精确 token 数需 bpe 词表，本机未下载）。

### 2.4 `object_class` / `color`：存在性与条件化分母

- `object_class` 是"帧的检测类名集合里出现过目标串"，不是"prompt 指定的**主体**属于该类"：
  背景里任一同类实例、任意位置的一次检出都算成功；per-instance 置信度被 `set` 丢弃，
  也没有计数/实例身份要求。论文 App. G.2 的 "success rate of generating the specific class
  of objects" 本身就只说 presence，所以这是**较弱**的错配，但与 subject_consistency
  同属"没有关注目标"的一族。
- `color` 报的是"命中 top-1 类名且 caption 含白名单颜色词的帧中，目标颜色词子串出现的比例"
  （`:46-55`），再对"至少有一帧命中"的视频取均值（`:81-89`）。它不是"视频里目标物体颜色的
  逐帧正确率"，而是条件率；分母口径、灰度拼写与 `[2][0]` 错位都会直接移动这个数。

### 2.5 `aesthetic_quality` / `imaging_quality`：命名基本成立，风险在代理与尺度

- 两者都不读 prompt、都不做字符串匹配；命名与论文一致（LAION aesthetic predictor；
  MUSIQ-SPAQ）。
- 聚合口径：都是"先对所有帧取 per-video 均值，再对视频取无权均值"
  （`aesthetic_quality.py:49,66-68,72`；`imaging_quality.py:46,51-53,54-55`）。论文写的是
  "average of all synthetic frames"，代码实际是**视频加权**而非帧加权，只在各视频帧数不等时
  有差异——引用论文口径时需要注明。
- 尺度不一致：`imaging_quality.py:53` 的 per-video 记录保留 MUSIQ 的 0–100，`:55`/`:72`
  的总分才除以 100，因此落盘的 per-video JSON 与 headline 不同量纲。
- 效度风险：两者都是学习到的无参考代理（LAION 美学评分、SPAQ 手机照片 IQA），论文关于
  "artistic quality" / "distortion" 的表述继承代理的训练分布。

### 2.6 `appearance_style`：命名基本成立，两处实现细节

- 用的是 metadata 里的风格串（对应论文"extract the style description"），但打分是
  `logits_per_text/100`（`appearance_style.py:61-63`），而论文写的是"normalized features 的
  mean cosine similarity"。CLIP logits = logit_scale × cosine，除以 100 等于假设
  logit_scale ≈ 100；分数无界且可为负，换 checkpoint/改标度会静默改变量纲。
- 聚合不一致：单进程 headline 是全局帧均值（`:73` `sim / cnt`），多进程分支却对
  `cur_sim` 取均值，而 `cur_sim` 在帧循环结束后等于**最后一帧**的分数（`:62-72,83`）。
  单卡与多卡会报出不同数字（同一族问题已在审计里记录）。

## 3. 推荐理由（补充 §0 表）

1. **`background_consistency`**：收益最高。`temporal_relocation` / 位置不变性契约可直接复用，
   缺陷结构已由 subject_consistency 侧证明；另外它与 `scene` 共用 86 条 prompt × 5 视频
   （`evaluate.sh:10`），任何基于 scene 视频的反事实族都会同时扰动它，做 scene 结论时必须声明。
2. **`temporal_style`**：最便宜且可判定。代码与 `overall_consistency` 相同，契约是
   "度量必须读风格子句"——只需 prompt 侧反事实（只替换风格子句，分数应随子句变化），
   不需要新模型；32-token 截断是同一族的可复现证据。
3. **`object_class` / `color`**：属 GRiT 字符串族，必须与"标签漂移率"一起报——对 suite 跑一次
   GRiT，统计生成类名与目标串不一致的比例（本机无权重，未测）；`color` 还要先修
   `[2][0] → [2][i]` 的实例绑定，否则任何评分对比都不干净。
4. **`temporal_flickering`**：与 `motion_smoothness` 同族的运动能量问题，需要 fps/分辨率对照
   与"静态 vs 运动"对照。
5. **`aesthetic_quality` / `imaging_quality` / `appearance_style`**：无字符串匹配；若要审计，
   问题是代理效度、量纲与聚合（`appearance_style` 另有单/多卡不一致）。

## 4. 供引用的事实与待补项

- 本笔记引用的 8 个源文件在 `fd18b3d` 下的 sha256。**2026-09-15 起，入选候选的 4 维
  （`background_consistency`、`temporal_style`、`object_class`、`color`）已写入
  `configs/upstream.toml` 并通过 `verify_upstream()` 校验**；未入选的 4 维只在本文列出，
  不写入该文件（以保持"钉住的即本轮范围"这一不变式）：

| 文件 | sha256 |
|---|---|
| `vbench/object_class.py` | `8d4c4a7feee52503e3d27f8aba0ed467cbc356ed75da2598367b572237e05db2` |
| `vbench/color.py` | `03baffc12508fd7ce6c70a227d6794bbc7152c24ba2d2c447a816974466db3d7` |
| `vbench/temporal_style.py` | `9f9e950efca4673721d0a07ea83137be9ebdc9aef2364f412300ba979beef57a` |
| `vbench/appearance_style.py` | `166b291abe55b0ac28ff4c937469e42cc2aa135c308ab06665518837956deb2a` |
| `vbench/background_consistency.py` | `7def92057819df0dad01c857eecf3b39b8b555c8cdfe881a672cc2b19dc226b4` |
| `vbench/temporal_flickering.py` | `c2c892034d1652c9390f213308c18b77ae20a87bd52fc50a125294ce83af4e83` |
| `vbench/aesthetic_quality.py` | `baa1894c2543483901a444af9aabe0701b1f937bf304141c53019a45b27abc57` |
| `vbench/imaging_quality.py` | `ac6c648cb4d5e82cd29c63be04977e9288c3540558cf12f979e66bef3e1f9e3a` |

- 待补（需要权重/CUDA）：① GRiT 生成类名与 COCO 目标串的实际不一致率；
  ② `temporal_style` 32-token 截断的精确条数（需 `bpe_simple_vocab_16e6.txt.gz`）；
  ③ `color` 中 `[2][0]` 错位在真实视频上的帧占比与分数偏移。
