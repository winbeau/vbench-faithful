# S1：数据加工与清洗子计划

状态：待实现。输入使用现有data/raw快照；不改写raw，不调用teacher。

## 1. 输出契约

训练JSONL使用统一外层记录；模型看到的输出仍保持最小格式，不把provenance塞进模型答案。

```json
{"sample_id":"unique-id","task":"spatial","input":{"prompt":"A cat is left of a dog."},"target":{"relationships":[{"subject":"cat","relation":"left","object":"dog"}]},"source":"synthetic_fixture","source_id":"fixture-1","group_id":"fixture-family-1","quality":"engineering_only"}
```

| task | input | target |
| --- | --- | --- |
| spatial | prompt | relationships数组，每项subject/relation/object |
| action | prompt | actions数组，K400合法名称 |
| objects | prompt | entities数组，标准对象名称 |
| scene | prompt、caption | 单字符串supported/contradicted/insufficient |

来源revision、hash、参数和split种子写manifest；每条记录保留源ID与group_id。固定任务名spatial/action/objects/scene。拒绝缺字段、未知任务、非字符串实体、不合法标签、任意额外输出字段；明确统计拒绝原因。

## 2. 目录与拟实现入口

- scripts/prepare_data.py：按来源清洗、输出counts/manifest/quarantine；默认不产生研究train split。
- src/vbench_prompts_compile/data.py：记录校验、task renderer、去重与分组工具。
- tests/fixtures/：少量手写、明确engineering_only的合成样本。
- data/processed/<build-id>/：ignored清洗产物，含candidates、quarantine、manifest、counts；新运行不覆盖既有build。
- data/smoke/<build-id>/：专门工程样本，禁止混进论文test。

这些路径是实现计划，不代表文件已存在。

## 3. 各来源加工策略

### Visual Genome → Spatial

- 复用已校验关系档案，保留image_id、对象ID和原谓词。
- 采用白名单规范化，不把on直接标above；under等关系保留原词并加弱监督标签。
- 除去无端点、相同对象ID、自关系、重复边，输出清洗统计。
- 从结构化边生成显式模板句仅作为synthetic_weak，不声称是人写自然prompt。
- 第一批最多400条单关系工程候选，尽量方向均衡；不足不复制补齐。多关系句仅拼接原图中明确保留的边，保留图内实体身份，歧义样本隔离；首批人工确认约50条多关系样本。
- 同图/反转表达/对象重命名等变体绑定同group；左右稀缺必须单报。不得以不同模板实现伪独立测试。

### Flickr30k → Objects

- 解析标记得到plain caption、实体短语、类型、chain/image ID，不把全图对象当caption目标。
- 隔离notvisual、scene、无法对齐标签的词；bodyparts、clothing是否计入须形成显式规则，不能静默删除后称完整目标集合。
- 第一批200条仅输出candidate供核对；词表映射采用固定别名表并人工复核，不利用LLM临时猜词。
- 保留上游image split、过滤UNRELATED_CAPTIONS，跨图重复文本继续隔离。
- 在annotation许可和GRiT词表未核实前不放行训练；Objects工程smoke先用手写合成fixture。

### K400 → Action

- 使用锁定的400标签及原始ID；生成400条标准句“A person is {label}.”作工程/弱监督，不宣称自然语句流畅或全面。
- 机械套句造成语法异常的类别单列，不以过滤造成类别遗漏后隐瞒。
- 第一轮仅有唯一可映射动作；多动作、隐含动作、词表外与歧义先进入quarantine，正式支持协议后再标注。
- 按canonical action分组可构建专门未见类诊断，但不能与词表内表达泛化指标混为一谈。
- teacher扩写未获本计划自动授权。

### SNLI → 辅助NLI

- 保持原train/dev/test，删除label=-1、空文本、重复已标注行；删除所有标签冲突组。
- train/dev完全相同的句对从train剔除，保留测试身份；再做premise来源聚类与跨源泄漏检查。
- 只生成auxiliary_nli候选，不直接写scene任务训练JSONL：通用动作蕴含不等于场景判别，尤其Scene会忽略prompt动作要求。
- SNLI缺原始来源ID时，文本去重只是补充，不能证明与Flickr来源独立。

### MovieGen → 无标签prompt池

- VideoBench去重998条；加audio文件video_prompt共1,525条精确唯一候选。
- 保留原文件/行号，合并TXT/CSV/audio重复和近重复家族，不使用concept列伪造动作/实体标签。
- 先生成unlabeled候选及特征统计，不写target、不自动投入SFT。
- 在许可非商业要求下，先冻结未来训练使用范围；使用的prompt及派生样本不再参与独立MovieGen测试。

## 4. Scene工程样本与真实样本分开

- 先手写至少30个场景家族，每家族各含支持、矛盾、不足，共约90对；只作为engineering_only。
- 覆盖ocean/river、bedroom/living room、同义表达、宽泛water/room和缺场景证据。
- 按场景家族分组，禁止同一模板的几种标签分到不同split制造泄漏。
- 真实研究数据需后续导入已获许可、含源视频ID的实际Tag2Text输出，与原prompt配对并人工标注。当前无这批数据时明确标记research_data_missing，不用SNLI替代验收。

## 5. 划分与金标准

- 工程smoke可以全量小集过拟合用于排错，但必须标明无泛化结论。
- 研究split先按原来源、图像/视频、近重复家族联合分组，再标注扩写；既有官方dev/test维持保留身份。
- 来源许可、未知类别规则、GRiT规范词表未通过则candidate/quarantine，不阻止用纯合成fixture做工程测试。
- 每任务先审核约50条，记录一致率与分歧；独立测试双人标注仲裁，不复用调试样本。

## 6. 验收

计数恒等式raw=retained+quarantined+deduplicated等分类必须互斥可追溯；重复运行同配置、同输入得到相同样本和split；每条target由输入支持；schema100%合法只是入口门槛，不代表语义正确。
