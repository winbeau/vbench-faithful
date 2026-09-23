# Dynamic Degree：官方原视频局部纹理抖动与运动 Repair 开发

最新状态（UTC 2026-09-23）：**真实VBench反事实已从90扩至450源/900 CF，两后端
各1800/1800，零失败；旧冻结联合头的不变性数值检查通过，但静态/轻微运动诊断
发现排序反例，不能称为合格的绝对运动强度Repair。**450组Origin为
**0.680000→0.857778**，旧联合Repair为**0.518089→0.516319**；同一静态图片
重复帧得0.456649，轻微平移8px反而降至0.442835。
用户要求的[锚点监督模型重训练](#dynamic-vjepa-anchored)已实际完成300步：新头将
该单例静止/8px平移/32px平移评分改为**0.008533/0.134990/0.293819**。
但DEV60自然偏好由20/23降至18/23，未过预定≥19/23；**训练完成，整体验收未通过**。
不使用该单例/450测试源训练或选检查点，不改公开默认，也不把新结果冒充450组测试。

本页是**局部纹理8px干预族**的唯一报告入口，不替换历史 FPS-resampling 主表。
当前证据见[450组扩展](#dynamic-vjepa-expansion450)及[静态反例](#dynamic-vjepa-static-frame)；此前
[90组冻结验证](#dynamic-vjepa-validation)、
[小头训练](#dynamic-vjepa-probe)、[SAM分区负结果](#dynamic-outer-support)及
[CoTracker3五源实测](#cotracker3-实测双查询臂完成但仍未修复)均保留为历史。
旧分区候选的0.560659→0.678265仍是该候选的实测结果，不重标为V-JEPA。
450组验证阶段未再训练/调尺度；后续模型训练是用户的新授权，版本和结果单列。
45条校准预留仍未读取；旧方法的正、负结果均不改写。

<a id="dynamic-vjepa-static-frame"></a>

## 静态图片与轻微位移诊断：旧小头存在真实运动排序反例

用户质疑分数一直在0.5附近，并要求用一张静态图片及轻微运动测试。按既有
DEV32原片UID字典序取第一条`v_0bc30f02053a39df2334`的首帧，不按模型分数选片。
原生512²、16帧、8FPS、2秒；同一图片重复、首末帧右移8px/32px、以及静止图上的
8px局部抖动，另用完整官方原片参照。四个新视频均为RGB无损编码，逐值解码核验通过。
全局平移采用反射边界，是已知几何位移的合成镜头式运动，不是自然关节运动。
这**五条独立诊断不混入450组正式官方原片反事实**。

| 输入 | Origin | 旧联合Repair | 仅自然偏好头 |
|---|---:|---:|---:|
| 同一图片完全静止 | 0 | **0.456649** | 0.481237 |
| 同一图片首末平移8px | 0 | **0.442835** | 0.477736 |
| 同一图片首末平移32px | 0 | 0.480346 | 0.518978 |
| 静止图片＋8px局部往返抖动 | 0 | 0.428737 | 0.436294 |
| 完整官方原片（参照，不是静态构造） | 1 | 0.605576 | 0.611576 |

结论有两层：静止0.456649说明当前显示分数没有静止零点；而8px平移相对静止
**−0.013813**，是已知运动控制的排序错误，**不能仅归因于未标定尺度**。
原始latent也从−0.173842降到−0.229663；任意单调平移/缩放都不能修复此倒序。
32px比静止高0.023698，原片高0.148927，故也不能说模型完全不读取运动。
这是一个明确反例，不给单例配总体CI，不外推所有自然视频均失败。

源码依据：[旧训练目标](../../metrics/dynamic-degree/src/dynamic_degree/learned_probe.py)
只监督自然视频之间的差值与抖动一致性，另以0.01权重约束原片平均latent接近0；
`sigmoid(0)=0.5`，却没有静止零点或已知运动幅度监督。它提供相对排序，不是
有绝对零点的运动强度。当前应分别保留“不变性改善”和“运动强度能力不足”，
不能用450组的低分差抵消这个反例。

证据：[独立像素/分数核验](../../output/dynamic-static-jitter/vjepa-static-frame-v1/analysis/summary.json)、
[静止视频](../../output/dynamic-static-jitter/vjepa-static-frame-v1/construction/videos/still.mp4)、
[8px平移](../../output/dynamic-static-jitter/vjepa-static-frame-v1/construction/videos/pan8.mp4)、
[32px平移](../../output/dynamic-static-jitter/vjepa-static-frame-v1/construction/videos/pan32.mp4)、
[源帧](../../output/dynamic-static-jitter/vjepa-static-frame-v1/construction/source_frame.png)。
5/5两后端均完成，静止版帧间RGB差精确为0，整数平移端点像素核验通过；原片
特征和分数与前一批精确相同。远端`vjepa-static-frame-v1/`与前述任务同根，
构造入口为`probe_vjepa_static_frame.py`，核验入口为`audit_vjepa_static_frame.py`。

<a id="dynamic-vjepa-expansion450"></a>

## 450源扩展：旧冻结模型的不变性及完整配对响应

用户要求扩大数据量后，从同30个已打开TEST prompt的450个官方MP4全部构造
反事实；不是另一批独立未见prompt。原90源的270个控制/CF视频字节保留，另为
360源新增1080个无损MP4；模型、两头和映射不变，全部**450原片＋450编码控制＋
900 CF＝1800输入/后端**重新推理，0构造/运行失败。
保持8原生像素局部坐标往返与两个固定种子，不加RGB噪声、不静态化原视频。
全部1200 qualified / 600 legacy flagged均评分，无质量标记选择性排除。
此前30项GIF仍按原生MP4协议NOT SCORED；45校准预留未读取。

| 队列 / 后端 | base → CF | 有符号Δ | MAE |
|---|---:|---:|---:|
| 全450 / Origin | **0.680000 → 0.857778** | +0.177778 | 0.177778 |
| 全450 / 旧联合Repair | **0.518089 → 0.516319** | **−0.001770** | **0.013358** |
| 全450 / 仅自然偏好头 | 0.516299 → 0.528713 | +0.012414 | 0.022800 |
| 新增360 / Origin | 0.675000 → 0.847222 | +0.172222 | 0.172222 |
| 新增360 / 旧联合Repair | 0.518333 → 0.516698 | −0.001635 | 0.013694 |
| 原90 / 旧联合Repair复测 | 0.517112 → 0.514804 | −0.002308 | 0.012016 |

双种子源等权、30prompt聚类20,000次bootstrap沿用冻结协议。全450的OriginΔ
CI为[0.133306,0.224444]；RepairΔ CI为[−0.004038,0.000598]，MAE CI为
[0.012129,0.014696]。固定相对尺度的允许涨幅0.017778，数值通过；
`ΔRepair−0.1ΔOrigin` CI [−0.024598,−0.014759]。
仍有个体最坏降分0.066883和最大绝对分差0.073489；base标准差0.083845，
范围[0.210709,0.694912]。不能以批量小变化推导绝对零点或小运动单调性。

全部450个人类配对现在两端都有CF，不再只是原90子集的6个有序对。
171有序、279人类平局分别报告，下面concordance对预测平局给半分：

| 后端 | 原片：正确/预测平局/错误；concordance | seed1701 | seed2904 |
|---|---|---|---|
| Origin | 114/52/5；81.87% | 70/91/10；67.54% | 71/89/11；67.54% |
| 旧联合Repair | 147/0/24；85.96% | 146/0/25；85.38% | 145/0/26；84.80% |
| 仅自然偏好头 | 148/0/23；86.55% | 146/0/25；85.38% | 148/0/23；86.55% |

联合头原片→两个CF分别损失1/2个净正确对；配对变化CI为[−3.42,+1.80]、
[−3.85,+1.02]个百分点，不能写成逐对无损。Origin两种CF均下降14.33个百分点，
CI分别[−21.43,−7.47]、[−21.02,−7.72]。原片联合头比Origin高4.09pp，CI仍
[−0.97,+9.12]pp；不能宣称自然集显著胜出。279个人类平局的平均绝对分差
（原片/两个CF）：联合头0.064436/0.067117/0.069233，Origin
0.240143/0.164875/0.172043，各自量纲诊断，不用于事后定阈值。

版本与证据：

- [冻结扩展配置](../../configs/dynamic-static-jitter/vjepa-expansion450-v1.json)，SHA
  `83fb066dfcd0e3334df47a216f4bb2469bcf63920efb0c3e4f3658a40466b72a`；
  扩展入口SHA `87e9a6a6187335a649c6d7bed5616f3cf1928fb6a7af7dbba7640b41ac1f4de8`。
  直接调用90组阶段的冻结评分实现；模型及两个头SHA与下节完全相同。
- [主结果](../../output/dynamic-static-jitter/vjepa-expansion450-v1/analysis/summary.json)，SHA
  `8b43ef4ab5e1ecd52593022d6c8035c3ca2632e8e2708c3aa3a23a1153019d25`；
  [逐源表](../../output/dynamic-static-jitter/vjepa-expansion450-v1/analysis/pairs.jsonl)、
  [1800输入](../../output/dynamic-static-jitter/vjepa-expansion450-v1/inputs/inputs.jsonl)、
  [构造记录](../../output/dynamic-static-jitter/vjepa-expansion450-v1/construction/)。
  输入SHA `afe145e4dc0df238126201d9b55abecfba2ed095fb59f40556e1822f9b6e1380`。
- [独立核验](../../output/dynamic-static-jitter/vjepa-expansion450-v1/independent-audit.json)：
  1800个Origin判据、3600个sigmoid值、450个精确编码控制、全部统计及bootstrap
  区间复算通过，最大差1.11e−16；既有720个重叠输入的特征、头输出与原始光流
  全部精确不变。全部450对×3视图的标签、预测和分母核验通过。
- H200根`/data/chenjiayu/dynamic-structural-motion-20260922/vjepa-expansion450-v1/`；
  本地同名`output/dynamic-static-jitter/`目录仅保存清单/代码/评分，不复制大特征。
  物理4–7卡、单进程逻辑cuda:0；四片452/452/448/448，各输入16帧/8FPS/2秒，
  合计3600媒体秒。V-JEPA分片96.67/89.95/96.11/89.78秒，整体约97秒；
  Origin分片312.44/315.94/308.08/305.88秒。Origin第4片为让出静态诊断延后约2分
  20秒，整体跨度约456秒，不能把它报成理想同时启动的四卡吞吐。

<a id="dynamic-vjepa-anchored"></a>

## 模型重训练完成：静止及位移控制改善，自然偏好门槛未通过

用户明确要求调整模型，不再靠评分后处理。已固定
[训练配置](../../configs/dynamic-static-jitter/vjepa-anchored-v1.json)和
[goal §0.4](../plans/2026-09-22-dynamic-structural-motion-repair-goal.md#vjepa-anchored-model-plan)。
相同冻结编码器，重新训练51,393参数小头，推理仍是视频→模型→连续分数，
没有新增光流门槛、输出偏置或缩放。训练在旧DEV210/60分组，排除刚才单例和TEST450。

### 实际训练方法与数据流

```text
训练：14个DEV prompt / 210官方原片及其训练干预
        → 冻结V-JEPA 2.1 ViT-B时空特征
        → 可训练注意力池化＋MLP头（51,393参数）→ sigmoid分数
        → 自然偏好＋抖动一致性＋静止零点＋已知位移排序监督
        → AdamW反向传播，实际更新头参数300步
推理：待测视频 → 同一冻结编码器 → 新头权重 → 同一sigmoid
```

不是用视频类别标签选择公式；推理不读取prompt、干预类型、原片参照或人类标签。
编码器保持冻结，**本轮训练的是小头，不是全量微调V-JEPA**。
原训练/开发验证按prompt分为210/60源（14/4 prompt）；45校准预留未读取。
自然有序/平局监督为训练67/143对、验证23/37对，来自Dynamic自身的人类标注。
复用270源的810份原片及两种8px局部抖动特征；每源第7帧另构造静止、首末平移
8px、平移32px、32px往返平移、静止＋8px局部抖动五视图，共**1350新特征、零失败**。
全部2160视图保留原生16帧/8FPS/2秒，送编码器前统一缩放384²；训练1680视图，
开发验证480视图。五种新增控制是合成监督，不冒充官方自然视频。

新loss在连续sigmoid分数上约束自然偏好差±0.2、人类平局差0、原片/抖动一致，
静止及静止抖动目标0，以及弱位移高于静止、强位移高于弱位移、往返位移高于静止。
原片只要求不低于自己的静止参照，不将全部自然视频强标为运动；去掉旧平均latent
接近0的gauge项。AdamW、lr=0.001、weight decay=0.01、seed=20260927，固定300步，
从旧joint初始化，**只用末步，不扫参数、不挑验证集最优检查点**。
配置在提取特征/训练前冻结；所有权重和验收阈值保持不变。

首次整块大张量训练触发非有限loss检查，未产出检查点，准确失败步未记录。
`training/`和[失败记录](../../output/dynamic-static-jitter/vjepa-anchored-v1/training-failure.md)
保留。使用24视图块计算、拼接全1680视图的输出后仍作一次完整loss/更新，保持
相同目标、初始化与300步；等价输出/梯度单测通过，初始630个自然视图latent与旧头
最大差1.19e−7。此恢复完整完成300步，无非有限输出/梯度。
大张量路径的故障根因尚未确证，不能把分块恢复当作根因证明或另一次调参。

### 用户指定静态图片案例：新权重保存后才评分

以下同一个DEV32源不在训练或DEV60验证中，不用于选检查点。原始视频/像素/编码
仍是上节的五个输入，只替换实际学到的小头权重。

| 输入 | 旧联合头 | 新锚点头 |
|---|---:|---:|
| 同一图片完全静止 | 0.456649 | **0.008533** |
| 同一图片首末平移8px | 0.442835 | **0.134990** |
| 同一图片首末平移32px | 0.480346 | **0.293819** |
| 静止图片＋8px局部往返抖动 | 0.428737 | **0.013851** |
| 完整官方原片（参照） | 0.605576 | **0.240548** |

静止→弱位移→强位移的倒序在此例已修复；局部抖动相对静止仍有+0.005318，
是小幅增分，**不是精确不变或降低**。不能用此单例证明所有自然运动均正确。
完整原片分数降低也不能单独解释成能力提升，需看下面的自然偏好退步。

### 60源开发验证：预定11项中10项通过，整体未通过

| 验证项 | 旧联合头 | 新锚点头 | 预定要求 / 判断 |
|---|---:|---:|---|
| 静止均分 / p95 | 0.443295 / 0.543806 | **0.008077 / 0.009107** | ≤0.05 / ≤0.1，通过 |
| 静止＋8px抖动均分 / p95 | 0.392174 / 0.501408 | **0.010941 / 0.014610** | ≤0.05 / ≤0.1，通过 |
| 弱平移高于静止 | 39/60 | **60/60** | ≥90%，通过 |
| 强平移高于弱平移 | 60/60 | **60/60** | ≥90%，通过 |
| 往返平移高于静止 | 60/60 | **60/60** | ≥90%，通过 |
| 原片自然偏好严格正确 | **20/23** | **18/23** | ≥19/23，**失败** |
| 两种原片抖动下自然偏好正确 | 19/23、19/23 | 19/23、19/23 | 相对各自原片至多损失1对，通过 |
| 原片→两种抖动平均分 | 0.566153→0.564112 | **0.179246→0.173609** | 新头Δ −0.005637，描述值 |
| 原片/抖动MAE | 0.009302 | **0.011906** | ≤0.02，通过，但比旧头增加 |
| 原片分数标准差 | 0.069419 | **0.056946** | ≥0.05，通过，非恒定输出 |

新头各平移控制均分：8px为0.143239，32px为0.389110，32px往返为0.389397。
静止加抖动平均增加0.002865；新头原片范围[0.022161,0.309121]，原片加抖动的
最坏个体降分0.068074（旧头0.035526），不能只报平均分小变动。
37个人类平局的平均分差（原片/两CF）为0.047893/0.047701/0.048202，单独报告，
不混进23个有序对的分母。训练67个有序对旧/新头均67/67，**不充作验证能力**。
这四个验证prompt已用于开发，以上为描述性点估计，不提供伪独立视频级CI，
不宣称新的独立留出验证或统计显著改善。

自然偏好退步不是净数抵消：18对保持正确、3对仍错误、**2对由正确变错误，0对改善**。
两对同属`a bicycle accelerating to gain speed`，下面是按人类正确方向签名的分差：

| 配对（A / B；人工标签A更强） | 旧头 signed gap | 新头 signed gap |
|---|---:|---:|
| `v_00e7a6a70a6ae9a46073` / `v_063211fcdfe26a49159f` | +0.005951 | −0.046817 |
| `v_973107bc9da80800fa14` / `v_bec36cb1921a7d560a43` | +0.021860 | −0.008354 |

因此**本版是静止/位移反例的部分修复，尚不能晋升合格Repair**。保持≥19/23门槛，
不删除失败对、不事后改阈值、不据此循环尝试多版。新头TEST450评分为**NOT RUN**；
前节450组0.518089→0.516319仍专属旧冻结joint头，不能继承到新头。
后续重点是让自然运动监督与零点监督兼容，而不是再调输出映射。关节、细小物体、
镜头与周期运动的分类人工核验、独立留出、绝对物理量纲标定仍未完成。
控制平移含反射边缘，不能排除模型利用这种合成特征，不能用控制的100%代替自然泛化。

### 权重、执行与可追溯证据

- 配置SHA：`3dcb93cd365ac0b7023890e1200dbe13e4a09919fe6e7ee12ac0869cc7a38712`；
  原提取/训练入口SHA `516778d9789d66f5e79a82a657fb4942099098f699e0754a98819998115fad73`；
  实际分块训练入口SHA `aca8ff6a21e47f4bea1413aae6566881fa65704a1f79a61404d5f7a1ac53ab84`。
  旧模型/编码器/上游版本沿用后文已锁定身份，未覆盖原权重。
- 新权重：`vjepa-anchored-v1/training-chunked/anchored.pt`，SHA
  **`8e10add01e050417baeccd191d80525afa3a7f6d60754de549703427823dd045`**。
  [实际训练结果](../../output/dynamic-static-jitter/vjepa-anchored-v1/training-chunked/summary.json)，SHA
  `e5a6404f41b00e5efe335a640f43ee6af80c906e64a2fe42d23a985dabf4df8c`；
  [完整逐源/视图分数](../../output/dynamic-static-jitter/vjepa-anchored-v1/training-chunked/scores.jsonl)，SHA
  `0252adf270747a85a98ca3e744c8b324ae20c842307044fd26bde5b5f67f65ac`；
  [300步曲线](../../output/dynamic-static-jitter/vjepa-anchored-v1/training-chunked/training_curves.json)。
- [独立统计核验](../../output/dynamic-static-jitter/vjepa-anchored-v1/independent-audit.json)：
  1350新特征台账、4320新旧模型分数、300步loss分项及全部预定门槛复算通过，
  最大统计差2.22e−16；该处“核验通过”不等于模型验收通过。
- [磁盘权重重载实测](../../output/dynamic-static-jitter/vjepa-anchored-v1/reload-verification/receipt.json)：
  新旧头在全部2160缓存视图上的4320个预测、以及用户五个诊断输入重载复测，
  latent和sigmoid分数最大差均**0**；七个参数张量均确实改变，总计51,393参数。
  单卡49.87秒，没有训练更新、编码器推理或TEST/校准媒体读取。
  重载入口SHA `55a41a8324f500f678d533fac6c04ad70db2a5257de27dedc5917fa9f91ca202`。
- H200根`/data/chenjiayu/dynamic-structural-motion-20260922/vjepa-anchored-v1/`；
  本地`output/dynamic-static-jitter/vjepa-anchored-v1/`保留小权重/清单/训练结果，
  稠密特征仅在H200，不入Git。四卡4–7提取340/340/335/335份新增特征，
  分片52.17/57.34/64.26/63.42秒，整体跨度约64.42秒（含加载/旧特征核验）；
  实际头训练单独使用物理4卡/逻辑cuda:0，105.74秒（含加载和评估），
  UTC 2026-09-23 09:09:39完成，**不是四卡训练耗时**。
  编码器没有训练更新，训练16帧/8FPS/2秒视图数1680，验证480；没有新下载。
  [复现命令](../../configs/dynamic-static-jitter/README.md#model-retraining-after-the-still-frame-counterexample)。
- 本地全仓测试：**1237 passed / 3 skipped**（19.68秒）；三个skip为需要真实
  CUDA/RAFT、UMT、GRiT的独立parity测试，不虚报已通过。Dynamic本轮H200实测及
  上述重载证据另列。`uv lock --check`通过；冻结`data/results/splits/runs`无改动。
  本轮未提交或推送，既有不相关工作区改动保留。

<a id="dynamic-vjepa-validation"></a>

## 首轮：真实 VBench 1.0 的90源冻结模型反事实验证

### 方法、数据流与事先冻结范围

用户要求“在实际vbench1.0上构造反事实，然后开始验证”。沿用已训练完成的
联合头（自然偏好＋抖动一致性）作为 Repair，另保留仅自然偏好头作消融。
编码器、两个头、预处理及输出映射全部冻结，不再训练、不按测试分数挑样本。
协议：[vjepa-validation-v1.json](../../configs/dynamic-static-jitter/vjepa-validation-v1.json)；
事先记录：[goal §0.2](../plans/2026-09-22-dynamic-structural-motion-repair-goal.md#vjepa-validation-plan)。

```text
官方完整原片 / 同片的8px局部坐标干预（原生16帧、8FPS、2秒）
  ├─ Origin：原版VBench解码/RAFT/阈值规则 → 单视频0或1 → 源等权动态比例
  └─ Repair：全帧384²缩放、不裁剪/跳帧 → 冻结V-JEPA 2.1 ViT-B
              → 4608×768时空tokens → 冻结51,393参数注意力池化/MLP头
              → latent q → 原训练时固定sigmoid(q)，连续相对分数
```

Repair推理只读待测视频，不读prompt、生成器、原片配对、种子或是否抖动。
`sigmoid(q)`未经绝对运动强度标定；不套用旧CoTracker的τ，也未按本轮均值
向Origin对齐。下面的10%比较是**冻结相对分数上的数值诊断**，不是共同物理量纲证明。

### 实际构造、评分分母与排除项

| 队列 | 官方原片 / prompt | 实际输入与状态 |
|---|---:|---|
| 已有DEV32复测 | 32 / 8 | 原样128输入：32原片＋32编码控制＋64条8px CF；不是新独立测试 |
| 预留TEST反事实 | 90 / 30 | 从原120源清单固定保留90个MP4，不补选；90原片＋90控制＋180 CF＝360输入 |
| 同prompt自然偏好 | 450 / 30 | 3生成器×5重复×30；含上述90原片，额外360原片仅作自然评分 |
| 预留GIF排除 | 30 / 30 | **NOT SCORED**：不属于固定原生MP4/16帧/8FPS协议；不是此次重新检查后认定全部缺时长 |
| 合并去重后 | 482原片 | 482原片＋122控制＋244 CF＝**848输入/后端**；Origin和V-JEPA各848/848，零失败 |

实际新生成**270个MP4**（90编码控制＋180 CF），原始官方视频字节不改。
8px是**原生像素的局部纹理坐标往返位移**，不加RGB噪声、不静态化、不作整帧平移、
裁剪、删帧或变速。复用已固定的平滑位移场/边界收敛方案，种子1701、2904；
原生256²/512²、全部帧和时间轴保留，编码为libx264rgb CRF0。
180条新CF均通过预期像素逐值与原生时间轴检查；不以插值存在等同语义无效，
也不声称这些新片已获用户逐片人工确认。

新360输入保留**240 qualified / 120 legacy rejected**，0构造失败。120个旧质量
标记均为`local_displacement_geometry_failed`，源于原Jacobian≥0.5筛查；最小实测
Jacobian为0.442275，仍无折叠。**全部180条CF均评分**，未删除这些警示项。
连同DEV32，848输入共686 qualified / 162 flagged；标记不是运行失败或零分填充。

TEST的30个prompt与训练/开发验证/校准预留21个及DEV32的8个prompt均不重叠；
与前期270个训练/开发验证原片无字节重复。官方媒体可能此前用于项目E0，
不能写成全项目从未见过。全部预测结束后才读取Dynamic自己的TEST偏好标签，
不复用Subject标注。这30个prompt现已暴露，不得作为未见留出反复调参。

### 不变性主结果与必要消融

先平均每源两个种子，再对源等权；MAE为各源/各干预的绝对分差均值。
95% CI均为20,000次prompt聚类bootstrap（seed20260926）；DEV32只有8个已暴露
prompt，区间只作开发描述；留出90源为30个prompt。

| 队列 | 后端 / 训练方式 | base → CF均分 | 有符号Δ及95% CI | MAE |
|---|---|---:|---:|---:|
| DEV32 | Origin原版 | 0.593750 → 0.781250 | +0.187500 [0.062500, 0.343750] | 0.187500 |
| DEV32 | 仅自然偏好头 | 0.528987 → 0.530246 | +0.001259 [−0.007035, 0.010166] | 0.021715 |
| DEV32 | **联合Repair** | **0.515726 → 0.508504** | **−0.007222 [−0.015714, 0.001426]** | **0.016736** |
| 留出90 | Origin原版 | 0.700000 → 0.900000 | +0.200000 [0.122222, 0.283333] | 0.200000 |
| 留出90 | 仅自然偏好头 | 0.514328 → 0.526435 | +0.012107 [0.006305, 0.018147] | 0.020825 |
| 留出90 | **联合Repair** | **0.517112 → 0.514804** | **−0.002308 [−0.005608, 0.000964]** | **0.012016** |

联合Repair在两队列均无平均虚假增分；允许涨幅分别为0.018750、0.020000，点估计
通过数值门槛。`ΔRepair−0.1ΔOrigin`的CI分别为[−0.044808,−0.010139]、
[−0.032088,−0.013394]。这不是逐片不变：Repair MAE的CI分别为
[0.011399,0.022837]、[0.010167,0.014003]。

相对自然偏好头，联合头的MAE在DEV32/留出90分别降低**22.93% / 42.30%**。
但自然偏好头的平均涨幅也低于本次数值上限，**不能声称只有一致性训练才能通过
均值门槛**；其作用证据是更小的绝对变化及较少的平均正漂移。
联合头seed1701/2904均分：DEV32为0.509679/0.507330，留出为0.513776/0.515832；
Origin两个种子的队列均分均相同。

未映射q也保留：DEV32为0.062284→0.031990（MAE0.069270），留出为
0.067294→0.056816（MAE0.050263）；原片q标准差分别0.355393、0.365521。
留出Repair原片范围[0.210709,0.653535]、标准差0.088163，**不是常数输出**，
但该范围不能解释为低/中/高绝对运动标尺。

个体负结果不隐藏：DEV32最大降分0.072612（`v_d763ec50c6af177e60ed`），留出
最大降分0.047728（`v_b994806ac2791e6e951d`）。批量稳定不等于所有真实运动
都被保留；这些病例没有事后排除，也没有据此继续调头或缩分。

### 自然运动偏好：单独报告，不能与反事实均值混算

450个人类配对中**171有序 / 279平局**，有序对覆盖26个prompt。
严格正确/预测平局/错误三项完整列出；concordance对预测平局给半分。
这不是全部450对的分类准确率，也未事后调平局阈值。

| 后端 | 有序正确 / 预测平局 / 错误（分母171） | concordance及95% CI | 279个人类平局的平均绝对分差 |
|---|---:|---:|---:|
| Origin原版 | 114 / 52 / 5 | 81.87% [75.82%, 86.80%] | 0.240143 |
| 仅自然偏好头 | 148 / 0 / 23 | 86.55% [78.38%, 93.10%] | 0.067238 |
| 联合Repair | 147 / 0 / 24 | 85.96% [78.09%, 92.11%] | 0.064436 |

联合Repair−Origin为**+4.09个百分点，配对CI [−0.97,+9.12]个百分点**，区间
跨零，不能宣称显著优于Origin。联合头较自然偏好头少排对1对；不变性改善存在
排序代价。Origin的52个平局已给半分，不以连续模型较少平局夸大优势。
人类平局分差是各自固定分数下的诊断，两种分数尚非同一物理量纲。

其中两端均有CF的仅**20对：6有序 / 14平局**。联合头原片、seed1701、seed2904
均为5/6严格正确；Origin对应5正确＋1平局、1正确＋5平局、2正确＋4平局。
不能把171个有序对全部写成加抖动后仍保持排序；这一小子集不足以证明各种运动
均被保护。独立绝对运动锚点与物体/镜头/关节/周期/细小运动的人工分组审核仍为**NOT RUN**。

### 执行身份、核验与证据

- 实验运行于H200物理4–7，各进程仅见本卡逻辑`cuda:0`；Python3.10、
  torch2.6.0+cu124、NumPy1.26.4、OpenCV4.11.0。四片输入数220/220/183/225。
  每输入16帧、8FPS、2秒：848评分输入对应1696媒体秒，482个原片对应964秒。
  V-JEPA四片含加载47.08/58.15/39.60/78.58秒，整体跨度约79.15秒；Origin
  162.79/161.84/115.85/149.25秒，跨度约162.85秒。跨度由逐片完成UTC和墙钟推算，
  不是端到端构造/传输耗时或单卡吞吐。两后端顺序运行，未再训练。
- 仓库基准HEAD为`fdf4890c67bee5881e53fc43926d555a63f9cef8`，运行含未提交新增代码，
  不能只靠该commit复现；实际执行快照和逐文件SHA见各片`provenance.json`。
  验证入口SHA：`b2619965627e1ea554b11fe5acbd0e8713cc893ef5ac12019d9cf955b638b317`；
  协议SHA：`aef55f4fbb6d31a334de014b2528dc01e1a373a315006bd559f6c2c9dbd38c09`。
- VBench固定`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`；RAFT权重SHA
  `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`。
  V-JEPA固定`204698b45b3712590f06245fbfba32d3be539812`；编码器权重SHA
  `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`；
  联合头SHA `2ec2a5d98e5555b6cab3c409c250e163181687626fafc1aee15aae45c4911b18`；
  自然头SHA `8b77f37517861fc9fe10de17d5eeeaaa2f4b57598bcdac916d93672bd94a4914`。
- 预测前四卡均用既有训练样例核验加载：特征精确一致，头重载/单条与批量推理
  最大差2.98e−8，小于预定1e−6。**122个编码控制**的分数、特征SHA精确相同；
  原128条Origin的分数和原始光流统计与旧记录精确一致，未偷换Origin或输入。
  每个Origin分片首条另与上游`dynamic.infer()`核验一致；运行后再次只读检查，
  两个上游checkout保持原pin且干净，编码器/两小头/RAFT权重SHA全部不变。
- [独立CPU核验](../../output/dynamic-static-jitter/vjepa-validation-v1/independent-audit.json)
  重算848条Origin判据、1696个头的sigmoid值、全部配对统计，并核对450个人类
  标签及执行快照哈希，最大统计差1.11e−16。验证入口与审计器是分开实现；
  这不等于已经复现V-JEPA官方动作识别benchmark。
- [完整机器结果](../../output/dynamic-static-jitter/vjepa-validation-v1/analysis/summary.json)，
  SHA `ad67cea4164b4450c484feffde753834e44bb7ceb04aa1f9e7dd1498cc58c6f8`；
  [逐源配对](../../output/dynamic-static-jitter/vjepa-validation-v1/analysis/pairs.jsonl)、
  [人类配对](../../output/dynamic-static-jitter/vjepa-validation-v1/analysis/human_pairs.jsonl)、
  [原始评分及provenance](../../output/dynamic-static-jitter/vjepa-validation-v1/scores/)、
  [构造清单与质量记录](../../output/dynamic-static-jitter/vjepa-validation-v1/construction/)、
  [30项GIF排除清单](../../output/dynamic-static-jitter/vjepa-validation-v1/selection/excluded_sources.jsonl)。
  [848输入清单](../../output/dynamic-static-jitter/vjepa-validation-v1/inputs/inputs.jsonl)的SHA为
  `fecc98496ab1617ed98fbfeb4a9c21e5802d7b326a95fbcc0f76796f256ce5ff`。
- 远端独立根：`/data/chenjiayu/dynamic-structural-motion-20260922/vjepa-validation-v1/`。
  `code/`保留实际执行源码，`construction/`保留270个新MP4及位移场，`scores/`
  保留特征；本地`output/dynamic-static-jitter/vjepa-validation-v1/`仅同步清单、
  小型评分与核验记录，不复制大缓存入Git。复现命令见
  [配置入口](../../configs/dynamic-static-jitter/README.md#frozen-validation-on-actual-vbench-10-user-authorized)。
- 本轮全仓纯算法/合约测试：**1225 passed, 3 skipped（21.01秒）**；锁检查106包、
  验证/审计两入口`--help`、`git diff --check`通过。冻结目录无工作树改动，索引摘要
  仍为`0648e4ebf351ebc7268f01830615f8e7d964d6eb8c627f84ed4e1ef9084115a2`。
  保留其他任务已有修改，未暂存、commit或push；模型和大体积输出不入Git。

结论：**固定8px局部纹理抖动下，原版动态比例明显虚增，冻结联合Repair的批量
分数基本保持；独立自然偏好提供了非恒定运动排序的支持。**仍存在个体降分、
偏好优势未显著、CF偏好覆盖仅6个有序对、未标定绝对强度等限制。按用户要求的
本次验证已完成；完整联合goal尚未完成，公开默认不变，不追加训练或调参。

<a id="dynamic-vjepa-probe"></a>

## 冻结视频编码器开发试验：V-JEPA 2.1＋小连续头

UTC 2026-09-23：用户明确授权下载官方 V-JEPA 2.1 ViT-B 权重，且仅做冻结
编码器＋小评分头的有限开发试验。协议在特征提取和训练前写入
[`vjepa-probe-v1.json`](../../configs/dynamic-static-jitter/vjepa-probe-v1.json)，
方法、预定开发门槛及边界见[goal §0.1](../plans/2026-09-22-dynamic-structural-motion-repair-goal.md#vjepa-probe-plan)。
本节保留开发阶段记录；后续主32组及留出已在[新冻结验证](#dynamic-vjepa-validation)
中完成，不改写本节当时分母、输出和限制。这不是默认评分变更。

数据流为：官方原片全部16帧/8FPS → 全画面384²缩放（无裁剪/跳帧）→ 冻结
V-JEPA 2.1 ViT-B → 4608×768时空tokens → 注意力池化/MLP小头 → 连续latent。
sigmoid仅用于显示相对分数；自然偏好监督没有给出绝对运动强度标尺，不能沿用
CoTracker强度的τ，也不能直接宣布达到相对Origin的10%验收。两帧tubelet的
时间压缩仍可能损失快速运动信息，不等同于已证明鲁棒性。

开发阶段结束时状态：**270原片/810份特征全部完成，零失败；两个51,393参数的小头均完成
固定100步训练，编码器86,833,152参数全部冻结。预定开发可行性条件通过，
但绝对强度校准、主32组新模型验收及正式留出未运行，不是最终Repair成功。**

固定输入为剩余21个官方DEV prompt的315个MP4；训练/验证/校准预留分别
14/4/3个prompt、210/60/45条。只处理训练与验证，810份特征（原片及两个
不同相位方案的新8px坐标干预）。训练210偏好对中67有序、143平局；验证60对
中23有序、37平局。现有主32组的全部prompt及正式留出prompt均排除；45条
校准预留不读取。该DEV群体含已用于旧校准的63源，不是全新自然视频群体。
新干预质量标记全部保留，未声称得到用户逐片视觉确认。
新干预在原生解码RGB内存中构造，未另写540个MP4；保留像素摘要、位移场摘要
和逐帧相位以供复现，没有新增视频编码差异。这不是主128个MP4的重新评分。

只比较自然偏好监督与相同监督＋latent抖动一致性两个训练臂，同初始化、固定
100步、同一小头，不挑检查点；无训练的同初始化头和常数分数另作诊断基线。
推理阶段不读prompt、生成器、干预种子或原片配对。

运行身份：源码SHA `204698b45b3712590f06245fbfba32d3be539812`，
[官方仓库](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812)；
[官方权重](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt)
实测1,664,223,428字节，SHA-256
`848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`。
这是HTTPS下载后的固定摘要，不是发布方签名摘要。上游此pin的自动下载地址
指向localhost；适配器仅使用 `pretrained=False` 构造及本地安全EMA严格加载，
没有改写上游源码。严格加载/重复性不冒充动作识别官方基准parity。

H200独立根目录：`/data/chenjiayu/dynamic-structural-motion-20260922/vjepa-probe-v1/`；
本地清单/执行快照：`output/dynamic-static-jitter/vjepa-probe-v1/`。
配置SHA `6c5e1464f903f6434cfa3a707ff2b6f93d10142a57fc0f586ac4cea82f3116af`；
输入清单SHA `b2bbc3fd06d9bcb56ba4e11a3486184b2a32da13caa626001b7add394c6df15b`；
训练/验证偏好清单SHA `79a04371644af1a8a1ef706acdb856c29f94867845866daadb9210d6797755bb`。
本地全仓纯算法测试结果见下方最终核验；新实现不接入公开默认CLI。
在本开发阶段结束时，正式留出、校准锚点及主32组冻结模型验收为NOT RUN；
后续测试状态见[新冻结验证](#dynamic-vjepa-validation)，不宣告goal完成。

### 有限试验实测结果

下表均为新数据分组上的**未校准sigmoid相对分数**，不是旧CoTracker强度或
新的Origin测量。每源两种8px干预等权，再对源等权；MAE为逐源/逐干预绝对
分差平均，另报有符号均分变化。验证只在两个最终检查点固定后读取。

| 数据 / 训练方式 | 原片数 | base → 两种CF均分 | 有符号变化 | 平均绝对变化 | 有序正确：原片 / 交替 / 不规则 |
|---|---:|---:|---:|---:|---:|
| 训练 / 仅自然偏好 | 210 | 0.502059 → 0.516680 | +0.014620 | 0.024381 | 67/67 · 67/67 · 67/67 |
| 训练 / 偏好＋一致性 | 210 | 0.506073 → 0.506006 | −0.000067 | 0.008238 | 67/67 · 67/67 · 67/67 |
| 开发验证 / 仅自然偏好 | 60 | 0.569694 → 0.574984 | +0.005290 | 0.013949 | 20/23 · 21/23 · 20/23 |
| 开发验证 / 偏好＋一致性 | 60 | **0.566153 → 0.564112** | **−0.002042** | **0.009302** | **20/23 · 19/23 · 19/23** |

验证集绝对分差降低**33.31%**；未映射latent MAE也从0.057845降到0.038620
（降低33.24%），不是仅由sigmoid显示尺度产生。原片latent标准差从0.310172到
0.286988，保留92.53%；带符号有序margin从0.508800到0.412713，说明运动差异
没有完全塌缩，**但margin有所减小，干预后的排序也弱于自然监督臂，不能隐藏**。
联合臂每种CF比自己的原片少排对1对，恰好满足预定允许范围，不等于逐对无损。

37个人类平局对的latent绝对差：自然臂原片/两CF为0.231860/0.263646/0.262290，
联合臂为0.214071/0.230417/0.227053。没有事后选平局阈值，不将23个有序对的
86.96%写成全部60对准确率。未训练头只排对13/23且latent标准差仅0.036070；
常数基线MAE为0但严格有序0/23，均不能作为成功Repair。

逐验证prompt保留局限（每prompt15原片/30CF）：

| prompt | 自然臂→联合臂：原片有序正确 | 联合臂有符号分差 | 联合臂绝对分差 |
|---|---:|---:|---:|
| a bicycle accelerating to gain speed | 6/7 → 7/7 | +0.002808 | 0.008073 |
| a motorcycle accelerating to gain speed | 5/5 → 4/5 | −0.002885 | 0.007660 |
| a motorcycle turning a corner | 6/6 → 6/6 | +0.001292 | 0.009077 |
| an elephant taking a peaceful walk | 3/5 → 3/5 | −0.009381 | 0.012399 |

CPU独立复算全部810行/2430个输出值、分母、均值、MAE、排序和特征清单，通过；
最大浮点差1.11e−16。按4个prompt聚类、2000次bootstrap（seed20260925），联合臂
验证有符号分差95%区间为[−0.006713,+0.002050]，绝对分差区间[0.007764,0.011317]；
有序正确率区间[0.70,1.00]。联合减自然的latent MAE区间[−0.027302,−0.010791]；
有序正确率差区间[−0.142857,+0.111111]。**只有4个已暴露开发prompt，区间只作
开发描述，不能外推总体或当独立正式测试。**

另作事后只读混淆检查：只用训练偏好拟合3个“生成器身份常数”，验证仅排对
9/23，低于联合臂20/23。这排除了该简单生成器偏置解释，不证明模型没有利用
其他外观/质量捷径，也没有补足低/中/高绝对运动锚点。

### 覆盖、计时与证据

- 特征分片204/204/201/201，共810，所有输入16帧、8FPS、2秒。270原片SHA
  均唯一；训练/验证无字节重复，与原32组也无字节重叠。540干预全部纳入，
  285条保留旧Jacobian<.5标记（交替145、不规则140），没有按分数或质量标记
  排除；最小Jacobian0.290744，无折叠/坐标越界，新构造仍未人工语义审核。
- H200物理7/6/5/4分别执行四片，各进程只见本卡逻辑cuda:0。包含加载的分片
  时间47.61/37.76/35.82/38.31秒；先单片验证再启动后三片，总跨度约96秒。
  两个小头的训练、缓存读取及评估共95.20秒，物理7。源视频总媒体时长540秒，
  810个完整输入视图对应1620秒；没有宣称这些视图是810个独立自然样本。
- 四个分片重复提取完全相同；权重运行后SHA不变，上游checkout仍干净。
  默认评分、主128条输入和Origin记录均未更改；无正式留出或45条校准媒体推理。
- 两个小头从磁盘`weights_only=True/strict=True`重载后，全部1620个已学习评分
  与原输出**逐值精确一致**，最大latent差0；没有重新调用编码器。复核使用
  物理5，共91.49秒（含缓存读取/哈希），见
  [重载检查](../../output/dynamic-static-jitter/vjepa-probe-v1/head-reload-check.json)。
- [机器可读主结果](../../output/dynamic-static-jitter/vjepa-probe-v1/training/summary.json)，
  SHA `9cf1593c5d7a28209a783235a5bde5a26a74d53fff62c676307bd3fab40d6b37`；
  [逐视频输出](../../output/dynamic-static-jitter/vjepa-probe-v1/training/scores.jsonl)、
  [固定训练曲线](../../output/dynamic-static-jitter/vjepa-probe-v1/training/training_curves.json)、
  [独立核验/区间/混淆检查](../../output/dynamic-static-jitter/vjepa-probe-v1/audit-with-confound.json)。
  首次不含混淆检查的`audit.json`及当时脚本快照保留，没有覆盖旧审计。
- 最终检查：全仓`1221 passed, 3 skipped`（18.13秒），新测试9项；`uv lock --check`
  106包、两个实验入口`--help`、`git diff --check`通过。冻结树无工作树改动，Git
  索引摘要仍为`0648e4ebf351ebc7268f01830615f8e7d964d6eb8c627f84ed4e1ef9084115a2`。
  权重/特征/小头仅在忽略的输出目录；未暂存、commit或push其他任务改动。

结论：**这一次有限训练给出了值得继续验证的信号：抖动响应下降、自然有序
准确率保持，且不是常数评分。**但大象仍只有3/5，干预排序有损失，独立运动
强度锚点/运动类型审核/主32组/正式留出在本开发阶段都未完成。当时建议下一步先补绝对运动标尺与
错误样本的实际运动核验，再冻结尺度测试主32组；不是解冻大模型或继续搜索
损失权重。本轮有限试验到此收束，不更换默认Repair、不宣告联合goal完成。

<a id="dynamic-outer-support"></a>

## 联合目标恢复执行：较大区域支持消融与真实运动复核

UTC 2026-09-23，本轮按“抖动不敏感、真实运动敏感”联合目标恢复执行。
固定32个官方DEV MP4、同一8px/双种子、全部128输入及另外63原片；无新模型
调用、无新构造、无尺度重拟合、不修改Origin或公开默认、不打开正式留出。
以下仍是已暴露开发集上的机制消融，**未验证成功的Repair**。

### 方法假设与完整测量

原分区让每个网格点归属于包含它的最小SAM掩码。很多细小区域只有≤3点或共线点，
不足以区分区域仿射与任意局部位移，保护约束因而保留整个区域。这可能同时保护了
纹理抖动。本轮只将所有权改为**最大包含掩码**，同面积按原顺序；未覆盖点仍为−1。
这不是新语义分割，也不证明大掩码对应完整物体。后续仍使用相同残余往返分解、
跨区域/时间检验、全144点top-5%运动强度与冻结 `I/(I+0.12684953311437752)`。

```text
同一视频缓存的CoTracker3全起点轨迹 + 中间帧SAM全部掩码
  → 最大包含掩码分组（唯一改变；原版为最小包含掩码）
  → 相同区域仿射/时间均值保护与两支路往返检查
  → 未知点仍保留原估计；不按可见点重新改变评分分母
  → 完整网格连续运动强度 → 相同冻结尺度
```

代码：[outer_support.py](../../metrics/dynamic-degree/src/dynamic_degree/outer_support.py)；
重放前固定的[配置](../../configs/dynamic-static-jitter/scoring.outer-support-dev32-v1.json)。
分区改变意味着只保护**新大区域**的瞬时仿射，不能沿用“每个原小部件均受保护”的
说法，尤其要核验关节、周期与小目标运动。

| 方法 | 32组Base均值 | 32组CF均值 | 有符号涨幅 | 涨幅/Origin |
|---|---:|---:|---:|---:|
| Origin原版 | 0.593750 | 0.781250 | +0.187500 | 100% |
| 前一残余往返候选 | 0.560659 | 0.681470 | +0.120811 | 64.43% |
| 最大区域支持消融 | 0.560659 | 0.678265 | +0.117607 | 62.72% |

全部191条完成、0失败，32组编码控制逐数组精确相同，191条Origin原样保留。
CF中的退化区域完全保护点对从14,490降到7,689（121,056个可靠点对），说明该
机制确实存在；但对比前一残余候选，仅22/64条CF降分、37不变、**5条升高**，
最大额外降分0.075847、最大升分0.025814。平均CF仅降0.003204，不能通过10%门槛。
最新候选23/64条CF实际发生分解抑制；基线支路45条未通过跨区域/时间预测，
残余支路60条未通过该检查，各原因可重叠。**仅改变分区不足以完成修复。**

两种子CF均分0.676745/0.679785；原始强度为0.255291→0.362046短边/秒。
新涨幅95% prompt-cluster CI为[0.070050,0.185002]；
`ΔRepair−0.1ΔOrigin`区间为[0.059471,0.156044]，仍明显大于0。
区间是同一冻结尺度下8 prompt、20,000次bootstrap，仅为开发诊断。
95条原片分数全部精确不变，不能证明原有对应/运动强弱估计已经正确。

已知列车原片依然为0.302752，两条CF为0.685087/0.722744；其中后一条比上轮
0.699496更高。原片车厢连接处运动漏检也未被分区修改恢复，**不能为更低的整体
CF均值忽略这一负结果**。

### 人类偏好与可见运动：有辅助证据，但覆盖严重不足

新增[自然响应审计](../../scripts/counterfactual/audit_natural_motion_response.py)仅读取
`dynamics_degree/dev`标注；不使用Subject标签、不读取test标签、不调整平局容差。
95条原片在全部870个DEV对中仅覆盖23对/20 prompt（2.64%），其中19对人类平局，
仅4对有序。较大分区与残余候选原片分数相同，得到相同结果：

| 有序自然对（A/B） | 人类偏好 | Origin A/B | Repair A/B |
|---|---|---|---|
| car turning，ModelScope / VideoCrafter | B | 0 / 1 | 0.085991 / 0.807519 |
| motorcycle turning，LaVie / ModelScope | A | 1 / 0 | 0.854249 / 0.524751 |
| motorcycle accelerating，LaVie / ModelScope | A | 1 / 0 | 0.837278 / 0.129299 |
| giraffe walking，LaVie / ModelScope | B | 0 / 0 | 0.127358 / 0.550158 |

有序严格排序Origin为3/4，候选4/4；一对Origin预测平局按不正确计。
配对差为+0.25，四prompt bootstrap区间[0,0.75]。候选4/4的经验bootstrap退化为
[1,1]，**这不是总体准确率无不确定性**，不能用于宣布自然运动非劣或总体优秀。
19个人类平局中，Origin严格同分9对，连续候选严格同分0对；平均绝对分差分别为
0.526316/0.257901。没有独立冻结的语义平局容差，不能把严格同分计数单独作为
连续评分优劣，更不能只挑4个有序对而隐藏其余19对。
完整结果：[前一候选](../../output/dynamic-static-jitter/natural95-response-residual-v1/summary.json)、
[新分区](../../output/dynamic-static-jitter/natural95-response-outer-support-v1/summary.json)。

95个MP4已按原评分输入SHA逐条核对，全部原生1520帧渲染为隐藏提示词、生成器和
分数的检查图；**渲染完成不等于全部审核完成**。本轮实际查看四个有序对的8条原片
加已知列车共9条、每条全部16帧，保留[代理核验记录](../../configs/dynamic-static-jitter/natural-motion-proxy-review.focus9-v1.json)。
检查者已看过分数/配对和部分旧帧，标记`blinded=false`，不冒充独立人类盲审。
旧32条代理检查中有24个MP4身份与当前批次相同；8个新替换MP4不能继承旧GIF审核。

复核看到了低运动长颈鹿与明显腿部/姿态运动的差别，也确认列车相对桥面持续通过。
另有三条汽车/摩托序列是“大部分时间稳定，只有末尾或中途1–2帧快速通过”，
**不能当成静态真值**。部分高运动片段还有生成变形/场景跳变，不能用它们的
高分认证精确刚体速度。周期、小尺度、纯镜头运动及所有新CF的完整人工审核
仍未完成；四个偏好对只提供有限的低/高可见运动排序证据。

### 来源、验证与下一步

结果：[完整统计](../../output/dynamic-static-jitter/dev32-outer-support-v1/summary.json)、
[191条记录](../../output/dynamic-static-jitter/dev32-outer-support-v1/scores.jsonl)。
CPU重放28.10秒，Python3.11.14/NumPy1.26.4，所有视频16帧、8FPS、媒体时长2秒，
相邻运动观测跨度1.875秒；没有四卡新增推理。执行快照为
`output/dynamic-static-jitter/code-outer-support-v1/`，HEAD仍为`fdf4890`，工作树含
候选代码，以逐文件SHA绑定，不把HEAD冒充完整版本。配置SHA
`052a519aad96068db7803aa0dfd46bd08e728cb133a5c2c78804d41d31cae034`，结果SHA
`18855cd985a840fd937adc07a93330d4c2de47ab8248f6ffd09dae5d433a0b11`。
独立核对191条输入/输出hash、最大掩码所有权、top-5%强度和映射，最大分数误差
1.11e−16、区域仿射矩误差8.76e−7、时间均值约束误差1.07e−14，未知点删除量为0。
完整测试**1212 passed / 3 skipped**；锁检查、两个研究入口help、diff检查通过，
冻结目录未改，不提交或推送。

下一步不再只调分区或门槛：当前评分每时刻重置固定网格，只取该起点轨迹的
紧邻位移；它的时间分解不等于同一物质点的完整轨迹。要检验跨起点与跨时间
对应能否恢复可见移动结构，同时保留短时快速、周期和小幅运动，再联合测8px。
可扩展的DEV MP4偏好清单有435对（134有序/301平局）；目前未新增完整自然评分。
真正运动响应、CF人工核验与冻结后的独立留出仍未通过/未完成，goal保持active。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.counterfactual.replay_residual_reversal \
  --config configs/dynamic-static-jitter/scoring.outer-support-dev32-v1.json \
  --output output/dynamic-static-jitter/dev32-outer-support-replay
.venv/bin/python -m scripts.counterfactual.audit_natural_motion_response \
  --scores output/dynamic-static-jitter/dev32-outer-support-replay/scores.jsonl \
  --output output/dynamic-static-jitter/natural95-response-replay
```

<a id="dynamic-residual-reversal"></a>

## 残余往返分解：小幅降分，未达标

用户要求进一步降低上一版的 CF 均分 0.68304。本轮固定已经校准的
`τ=0.12684953311437752`、全部 8px 视频及 Origin，不重新缩分或改造数据。
先核对上一版：64 条 CF 中只有 21 条发生抑制，42 条不满足跨区域/时间预测
门槛，30 条最强分量没有重复反向；这些原因可以重叠。大幅区域/镜头运动可能
支配最强分量，因此增加一条**先分离受保护运动，再分析剩余量**的支路。

新候选数据流：

```text
同一视频的既有 CoTracker3 全起点网格轨迹 + SAM 区域
  → 原有保护分解（保持不变）
  → 将余下速度分解为“区域瞬时仿射 + 每点时间均值”与其正交残余
  → 只在正交残余上检验原有往返/占比/跨区域-时间预测门槛
  → 只移除通过检查且不改变受保护分量的成分
  → 原有全网格 top-5% 连续运动强度 → 同一冻结 0–1 映射
```

门槛没有放宽；未知点不删除，较小区域的保护不变。不读取配对原片、文件身份、
构造种子、位移场或预知的抖动频率。这是基于已看过的开发诊断设计、重放前固定
的一次候选比较，不是预注册的独立验证；同步的真实非仿射运动仍可能被误认为干扰。
实现见 [residual_reversal.py](../../metrics/dynamic-degree/src/dynamic_degree/residual_reversal.py)，
配置见 [scoring.residual-reversal-dev32-v1.json](../../configs/dynamic-static-jitter/scoring.residual-reversal-dev32-v1.json)。

| 方法 | 32 组 Base 均值 | 32 组 CF 均值 | 有符号涨幅 | 相对 Origin 涨幅 |
|---|---:|---:|---:|---:|
| Origin 原版 | 0.593750 | 0.781250 | +0.187500 | 100% |
| 上一版连续 Repair | 0.560659 | 0.683035 | +0.122377 | 65.27% |
| 新残余往返候选 | 0.560659 | 0.681470 | +0.120811 | 64.43% |

CF 均分只下降 **0.001566**，原误增减少约 **1.28%**，改善很小。64 条 CF 中
7 条下降、57 条不变、无新增升分；逐条最大额外降分 0.023248。两个种子的 CF
均分为 0.680000/0.682939。原始物理强度均值为 0.255291→0.362530 短边/秒。
按 8 个 prompt、20,000 次配对聚类 bootstrap，新涨幅的 95% CI 为
[0.071202, 0.188791]；相对上一版涨幅变化 CI 为 [−0.002746, −0.000537]。
这些区间条件于已冻结尺度和已观察过的开发集，不是泛化效果的独立证据。

**保护检查：**32 条开发原片及另外 63 条自然开发原片的分数全部精确不变。
后 63 条均值仍为 0.571429，逐条最大损失和平均绝对变化都为零；32 编码控制
逐数组精确一致。这只证明本批分数保留，不等同于真实运动感知或人类偏好非劣。
按原片均值 0.560659，10% 门槛要求 CF 均分 ≤0.579409，本候选仍远未达到。
**不宣称修复成功、不晋升公开默认，不继续通过本批结果调门槛。**

产物：[汇总](../../output/dynamic-static-jitter/dev32-residual-reversal-v1/SUMMARY.md)、
[完整统计及逐源结果](../../output/dynamic-static-jitter/dev32-residual-reversal-v1/summary.json)、
[191 条记录](../../output/dynamic-static-jitter/dev32-residual-reversal-v1/scores.jsonl)。
本地 Python 3.11.14/NumPy 1.26.4，191/191 完成、0 失败，CPU 墙钟 27.19 秒；
无新模型调用、无新 GPU 进程、无新 CF。输入模型证据及其来源 SHA 全部复核。
独立重算全部 191 条 top-5%/积分/映射，最大分数误差 1.11e−16；最大区域仿射
矩误差 5.40e−7、最大时间均值约束误差 7.11e−15，未知点删除量为零。
全部 191 条 Origin 原样保留。源码快照为
`output/dynamic-static-jitter/code-residual-reversal-v1/`，配置 SHA
`4d9e27da8375db7a20ce65dfc5b6efaf6b45cb3ea00cfea1e53d7924339ca134`，结果 SHA
`24ee9fdcc411872c871f93269f4dfeb3e86759145636229f468b4485d3d39e44`。
完整测试 **1203 passed / 3 skipped**，锁检查、研究入口 help、diff 检查通过；
冻结目录无改动，未提交或推送。新 CF 人工审核、真实运动非劣性及正式留出仍为 NOT RUN。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.counterfactual.replay_residual_reversal \
  --config configs/dynamic-static-jitter/scoring.residual-reversal-dev32-v1.json \
  --output output/dynamic-static-jitter/dev32-residual-reversal-replay
```

<a id="dynamic-calibrated-scale"></a>

## 最新尺度校准：Origin 不改，Repair 连续 0–1

以下保留首次尺度校准结果；不改尺度的后续残余候选见上一节。

用户要求量纲对齐，并希望 32 组原片的 Repair 均分与 Origin 接近、略高。
本轮完成了**数值尺度校准**，没有通过当前 32 组的均值或反事实涨幅拟合参数。
实测原片均分接近但略低，不能把“希望略高”写成已经实现；尺度对齐也不等于
动态视频比例与运动强度具有相同物理含义。

### 固定映射与独立校准数据

保留上节定义的原始连续运动强度 `I`（短边/秒），增加：

`Repair = I / (I + τ)`，`τ = 0.12684953311437752` 短边/秒。

单视频零强度映射为零，严格单调、连续、无运动阈值或硬裁剪；`τ` 是半饱和
强度，不是动/静决策阈值。公式、唯一尺度参数、零偏置和固定指数 1 均在新
模型评分前声明。只解“独立校准原片的 Repair 均值 = 这些原片的 Origin 均值”，
没有为了让当前批次略高而添加偏置，也没有按 CF 涨幅选择映射或尺度。

校准池来自只读 `data/processed/e0_scoring_manifest.csv`：排除当前 32 组全部
8 个 prompt、原清单的预留测试 prompt/UID，使用余下全部 **21 个 DEV prompt**。
每 prompt 的 LaVie、ModelScope、VideoCrafter 各选 1 个官方 MP4，按
`SHA256("20260923:" + video_uid)` 取首项，共 **63 条原片**。不读分数选样、
不构造新视频、不读取正式留出媒体；这些是既有研究开发输入，不是全新的独立测试。
每片仍为 16 帧、8 FPS、2 秒，运动积分时间 1.875 秒；模型、SAM、网格、保护
分解、全部原生帧及 Origin 公式不变。63 条均无被分解删除的分量。

63/63 评分成功，0 失败/排除；Origin 均值 **36/63 = 0.571429**，原强度均值
0.343394 短边/秒，映射后校准均值 0.571429。按 21 个 prompt 聚类的 20,000 次
bootstrap，`τ` 的 95% CI 为 **[0.081846, 0.190399]**。冻结点估计后，将同一映射
用于当前 32 组的原片、编码控制、两个 CF 种子及无抑制消融；应用阶段不再拟合。

实现：[intensity_scale.py](../../metrics/dynamic-degree/src/dynamic_degree/intensity_scale.py)；
预定协议：[scoring.natural63-calibration-v1.json](../../configs/dynamic-static-jitter/scoring.natural63-calibration-v1.json)；
冻结参数：[calibration.natural63-intensity-v1.json](../../configs/dynamic-static-jitter/calibration.natural63-intensity-v1.json)。

### 当前 32 组结果

先逐视频映射，再平均每源两个 CF，最后对 32 源等权平均。仍报告有符号均值差，
不替换为 MAE、正部均值或逐条通过率。

| 方法 | Base 均值 | CF 均值 | 有符号变化 | 变化的 prompt 聚类 95% CI |
|---|---:|---:|---:|---|
| Origin 原版 | 0.593750 | 0.781250 | +0.187500 | [0.062500, 0.343750] |
| 同尺度无抑制轨迹消融 | 0.560659 | 0.697890 | +0.137232 | [0.080466, 0.211971] |
| Repair 连续校准分 | 0.560659 | 0.683035 | +0.122377 | [0.073126, 0.190604] |

Repair 两种子 CF 均值为 0.680808/0.685262。Base 的 Repair−Origin 为
**−0.033091**，配对 prompt 聚类 CI **[−0.148431, 0.095693]**；本批次未实现
“略高”，此 CI 含零也不构成统计等效性证明。Repair 原片逐视频范围
0.072910–0.861554，仍是连续分数，不是重新计算二值视频比例。

在冻结后的共同数值尺度上，Repair 涨幅是 Origin 的 **65.27%**，大于 10%；
允许涨幅为 0.018750，实际 0.122377，**未通过**。`ΔRepair − 0.1ΔOrigin`
的 prompt 聚类 CI 为 **[0.061596, 0.161674]**，全为正。以上区间来自 20,000 次
固定尺度重采样，仅条件于拟合后的 `τ`；校准参数不确定性单列，不冒充联合区间。
无抑制消融同样使用该映射，原始短边/秒结果完整保留，避免把饱和映射本身当成
抖动识别改善。对应关系可靠性、自然运动非劣性、新 CF 人工审核与正式留出仍未验证。

### 证据、运行和检查

- [校准源清单](../../configs/dynamic-static-jitter/sources.natural63-calibration-v1.jsonl)、
  [选择记录](../../configs/dynamic-static-jitter/sources.natural63-calibration-v1.selection.json)；
  [63 条完整拟合/核验记录](../../output/dynamic-static-jitter/natural63-calibration-fit-v1/calibration.json)。
- [32 组总表与逐源表](../../output/dynamic-static-jitter/dev32-cotracker3-calibrated-intensity-v1/SUMMARY.md)、
  [summary.json](../../output/dynamic-static-jitter/dev32-cotracker3-calibrated-intensity-v1/summary.json)、
  [128 条原始/映射分数](../../output/dynamic-static-jitter/dev32-cotracker3-calibrated-intensity-v1/scores.jsonl)。
- H200 4–7 卡，每 worker 仅见一个 UUID；63 次 RAFT 主评分、每 shard 首条官方
  `infer` parity（4/4）、1008 次全视频 CoTracker3 查询、63 次 SAM 中帧调用。
  worker 计算窗口 **100 秒**，含启动的调度墙钟 **111.37 秒**，最大单进程
  已分配显存 6,168,425,984 bytes。所有本任务 GPU 进程已结束。
- 远端根 `/data/chenjiayu/dynamic-structural-motion-20260922/`；输入台账
  `natural63-calibration-inputs-v1/`，模型输出 `natural63-calibration-scoring-v1/`，
  不可变源码 `code-natural63-calibration-v1/`，本地 `output/dynamic-static-jitter/`
  有同名副本。HEAD 仍 `fdf4890c` + dirty tree，以逐文件 SHA 为真实运行身份；
  上游仍 `fd18b3d`，权重同此前固定版本，没有下载新模型。
- 63 条输入/配置/源码/缓存 SHA、全部 1008 组轨迹、区域分解和官方决策重放
  通过。首次本地核验早于 rsync 完成，因缺缓存而安全退出、未写拟合结果；
  取回完毕后全量核验通过，不补选/丢弃任何样本，也未重跑模型。
- 32 组应用零新增模型调用；128 条 Origin 和未缩放 Repair 逐条原样保留，
  128 条映射独立公式校验通过，32 条编码控制仍精确相等。
- 完整测试 **1198 passed / 3 skipped**；`uv lock --check`、八个公开 CLI 与
  四个相关研究入口 help、`git diff --check` 通过。冻结目录 Git 索引 SHA 仍为
  `0648e4ebf351ebc7268f01830615f8e7d964d6eb8c627f84ed4e1ef9084115a2`。
  公开默认未晋升、未提交或推送本轮工作。
- 拟合记录 SHA：`618eaf576a26ebd8e8366482e5353bf39361dfa1f89df0f5938bf157c7f8475f`；
  当前结果 SHA：`3ea288df89e56966871d977afcae34e8ab0236ba21d073f1efdcc3b11cf98f8f`。
  复现命令见[配置入口](../../configs/dynamic-static-jitter/README.md#current-score-scale-independent-natural-calibration)。

## 最新：32 组官方 MP4，Origin 原版、Repair 连续运动强度

以下是尺度映射前的物理强度与完整模型证据，原始数值保留；当前 0–1 评分见上一节。

用户确认替换缺少原生时长的 8 条 GIF：保留原开发清单中的 24 个 MP4，在同一
8 个 prompt 的官方开发池中按固定生成器轮转和 UID 哈希补选 8 个 MP4，未读取
分数来补选，未打开预留 120 条测试视频。清单为
[sources.local-texture-dev32-mp4-v1.jsonl](../../configs/dynamic-static-jitter/sources.local-texture-dev32-mp4-v1.jsonl)，
[选择记录](../../configs/dynamic-static-jitter/sources.local-texture-dev32-mp4-v1.selection.json)
保留被替换的 GIF 身份。共 32 源、8 prompt，每 prompt 四源；LaVie/ModelScope/
VideoCrafter 各 11/11/10 源，并非四生成器平衡数据。

构造不变：官方完整原片，16 帧、8 FPS、2 秒，原生 256²/512²；局部纹理最大
8px 往返位移，种子 1701/2904。每源原片、无干预编码控制、两条 CF，共 128 个
输入。原片直接评分；编码对照与 CF 使用 RGB 无损编码。64 条 CF 全部逐像素
验证、时间轴一致、采样坐标未越界，最小 Jacobian 为 0.442275，无折叠。
旧自动筛查保留 42 条 rejected：40 条仅 Jacobian 门槛，2 条还触发结构相关性
门槛（全批最小 0.803341）。**全部保留并评分，没有筛选更有利的子集**；此前
用户对五源十条的视觉确认不冒充对全部新 CF 的人工审核。

### 评分口径

**Origin 完全不改。** 使用固定上游 `fd18b3d` 的 RAFT、最强 5% 光流位移及
官方阈值/计数规则，单视频仍为 0/1，批量均值是动态视频比例。128 条官方记录
在本次连续重放中逐条保留，不替换成“Origin 连续强度”。

**只有 Repair 改为连续量。** 当前研究入口默认连续强度；旧二值头仅供显式
历史复现，不再作为当前 Repair 主结果。评分数据流为：

```text
单个视频全部原生帧
  → CoTracker3 offline：每帧起点的固定 12×12 网格
  → 同视频中帧 SAM 区域 + 既有往返分解（保护区域仿射和每点平均速度）
  → 每个相邻帧对：修正位移幅值最大的 floor(144×5%)=7 点取均值
  → 累计上述位移 /（画面短边 × 全部相邻帧间隔总和）
  → 连续运动强度，单位：short-side lengths / second
```

实现：[motion_intensity.py](../../metrics/dynamic-degree/src/dynamic_degree/motion_intensity.py)，
配置：[scoring.dev32-cotracker3-intensity-v1.json](../../configs/dynamic-static-jitter/scoring.dev32-cotracker3-intensity-v1.json)。
公式为 `I = Σ_t top5mean(||d_t||) / [min(H,W) × Σ_t Δt_t]`。
16 帧间有 15 个区间，观测运动时长为 1.875 秒，不把容器时长 2 秒错作运动
积分分母。不设运动阈值、不做 sigmoid、不裁剪到 [0,1]、不按本批 base/CF
或 Origin 拟合缩放系数。0.25 表示该空间统计量下每秒约四分之一个短边的移动，
不是“25% 视频动态”。全部 144 点预测保留固定分母，可见性另报，不以删除点
或缺失填零换低分；它仍是模型估计，不是可靠物理对应的认证。

“最强 5%”的空间侧重与 Origin 一致，但 Origin 是稠密 RAFT，Repair 是网格
CoTracker3，**并非估计器完全对齐**，也不能靠把自然 base 均值强行调成一样
来证明对齐。原片真实运动漏检及自然运动保护仍需独立验证。

### 32 组批量均值（主结果）

每源先平均两个固定种子的 CF，再对 32 源等权平均；不是绝对差、正部均值或
逐条通过率。两种单位分别列出，禁止直接把两行涨幅作同量纲比例。

| 方法 | 单位 | Base 均值 | CF 均值 | 有符号变化 | 变化的 prompt 聚类 95% CI |
|---|---|---:|---:|---:|---|
| Origin 原版 | 动态视频比例 | 0.593750 | 0.781250 | +0.187500 | [0.062500, 0.343750] |
| Repair 原始轨迹消融（无抑制） | 短边/秒 | 0.255291 | 0.374620 | +0.119329 | [0.092800, 0.155512] |
| Repair 连续强度（保护分解） | 短边/秒 | 0.255291 | 0.365000 | +0.109709 | [0.087244, 0.140319] |

Repair 两种子 CF 均值分别为 0.362890/0.367109。分解相对**同估计器、同单位**
的无抑制消融只减少 8.06% 的平均误增；32 组平均 CF 均高于各自 base，不能
称为不变性成功。21/64 条 CF 有非零分量被删除，所有原片无删除。用户要求的
“不超过 Origin 涨幅 10%”保留为目标，但在原版二值 Origin 与连续 Repair
之间未建立独立、合理的尺度映射，因此此比值验收记为 **NOT EVALUATED**，
不是通过，也不以事后缩分解决。自然运动非劣性、新 CF 人工审核及正式留出仍
为 NOT RUN。

此前新增二值 Repair 的 0.687500→0.781250 是 22/32→25/32 的预测动态比例，
并非连续强度；用户已明确取消该主口径。其 50% 涨幅比例只属于该历史二值头。
旧结果及证据保留在原 run，不改写、不升级成当前结果。

### 产物与验证

- [连续结果、32 组逐项表](../../output/dynamic-static-jitter/dev32-cotracker3-intensity-v1/SUMMARY.md)、
  [summary.json](../../output/dynamic-static-jitter/dev32-cotracker3-intensity-v1/summary.json)、
  [128 条评分记录](../../output/dynamic-static-jitter/dev32-cotracker3-intensity-v1/scores.jsonl)。
- [源模型批次](../../output/dynamic-static-jitter/dev32-cotracker3-boolean-v1/launch.json)及
  [完整证据核验](../../output/dynamic-static-jitter/dev32-cotracker3-boolean-v1-audit/summary.json)：
  128/128，0 运行失败，2048 次全视频网格查询、128 次中帧 SAM，32 编码对照
  逐数组完全一致；每分片首项真实官方 `infer` parity 通过。四卡进程计算窗口
  180 秒，含解释器启动的调度墙钟 193.29 秒。每个源视频长 2 秒。
- H200 输出根：`/data/chenjiayu/dynamic-structural-motion-20260922/`；源执行快照
  `code-dev32-scoring-v1`，本地同名快照位于 `output/dynamic-static-jitter/`。
  HEAD 为 `fdf4890c` 加未提交研究代码；完整执行文件 SHA 在各分片 provenance，
  不把 HEAD 单独当作运行身份。权重/上游保持前述固定版本。
- [连续重放脚本](../../scripts/counterfactual/replay_mp4_dev32_intensity.py)再次核验所有
  缓存/输入/来源 SHA，重放区域分解和原评分证据，再只更换 Repair 的数值汇总。
  **无新增模型推理、无新视频构造、无阈值调参；Origin 128 条记录原样保留。**
- 本轮完整测试 **1182 passed / 3 skipped**，`uv lock --check` 通过；未改依赖或
  同步覆盖 CPU torch overlay。新增测试覆盖连续性、原 6px 阈值附近无跳变、
  空间/时间单位、强度可大于 1、Origin 保持二值，以及缺失源不能删掉后取均值。

复现连续重放：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m scripts.counterfactual.replay_mp4_dev32_intensity \
  --run-root output/dynamic-static-jitter/dev32-cotracker3-boolean-v1 \
  --audit output/dynamic-static-jitter/dev32-cotracker3-boolean-v1-audit/summary.json \
  --manifest output/dynamic-static-jitter/local-texture-dev32-mp4-8px-v1/candidates.jsonl \
  --config configs/dynamic-static-jitter/scoring.dev32-cotracker3-intensity-v1.json \
  --output output/dynamic-static-jitter/dev32-cotracker3-intensity-replay-new
```

以下章节均为此前各轮证据及局限，不能把旧 `score=null` 条目改称新强度结果。

## 机制复核：先确认真实运动，再设计 Repair（2026-09-22）

用户要求先分析自然运动与局部像素抖动的光流轨迹。本轮因此**停止追加评分候选和
阈值调整**，复用相同二十条输入的 2× RAFT 缓存，做 CPU 机制诊断并查看原片的
全部十六个采样帧。没有生成新视频、减弱 8px、追加模型推理或读取留出结果。

结论不是“找到一个反向阈值即可修好”，而是发现两个必须分开解决的问题：

1. 抖动的对应关系可以很可靠，仍不应累加为真实运动增益。
2. 原片中明显的实体运动也可能没有被当前光流捕获。**把 CF 压回一个错误的低
   base 分数不是成功；高对应覆盖也不证明运动已经被测对。**

### 先看原片，不以 Origin=0 判静止

以下为本轮代理对逐帧显示的内容核验，不是新的人类偏好标注或盲审真值。图像
直接取自核验 SHA 的官方原 MP4；16 帧均显示，评分采样保持原有 8 FPS/256 边长。
这里只确认可见运动和后续需要保护的成分，不据此量化真实物理速度。

| 官方源 | 逐帧可见内容 | 对设计的约束 |
|---|---|---|
| [自行车减速](../../output/dynamic-static-jitter/flow-mechanism-dev5-8px-v1/video_000000_frames.png) | 自行车/车轮轮廓持续向画面左侧移动，路面大体稳定 | 不能让大量静背景抹去局部持续位移 |
| [桥上列车](../../output/dynamic-static-jitter/flow-mechanism-dev5-8px-v1/video_000004_frames.png) | 列车沿桥通过，车头、接缝和车尾位置明显改变，桥体/树林大体固定 | **不是静态片**；窄小运动区域必须被捕获，不能以 Origin=0 定义低运动真值 |
| [雪地自行车](../../output/dynamic-static-jitter/flow-mechanism-dev5-8px-v1/video_000008_frames.png) | 车轮、身体/腿部姿态及背景相对位置变化，并含明显生成形变和水印 | 不能将反向多、循环失败直接等同于静止或干扰；目前轨迹证据仍不足 |
| [马](../../output/dynamic-static-jitter/flow-mechanism-dev5-8px-v1/video_000012_frames.png) | 头部、腿部及身体位置有小幅变化，背景较稳定 | 应保留局部、小幅与关节运动，而非只保留全画面同向平移 |
| [长颈鹿](../../output/dynamic-static-jitter/flow-mechanism-dev5-8px-v1/video_000016_frames.png) | 腿部姿态改变，身体/颈部在画面内移动 | 姿态与部件运动应受保护，不能用单一全局仿射代替所有运动 |

**这五条不能充当已审核的“纯静态”组，也不足以证明真实周期运动已经覆盖。**
既有自然开发 32 源/8 prompt 的类别仍需逐视频确认；不能用 prompt 的“走路、飞行”
直接认定周期运动，更不能把数学周期测试当作自然视频验证。

### 轨迹诊断定义与全部五源结果

实现：[analyze_flow_mechanism.py](../../scripts/counterfactual/analyze_flow_mechanism.py)。
同前轮的局部随动轨迹，不重新跑光流。每个 lag 的每个起始时刻只取一个固定
最大内部裕量窗口，保留全部时间相位；1/2/3/4 帧间隔对应 125/250/375/500 ms。

- 轨迹路程 `L=Σ‖q(t+1)−q(t)‖`；两步效率 `E₂=‖q(t+2)−q(t)‖/L₂`。
  汇总按路程加权，避免把近零运动的任意方向放大。持续位移通常 E₂ 较高；
  返回出发处会使 E₂ 下降，但真实周期运动也会下降。
- 相邻速度方向余弦 `C` 按两个速度模长之积加权。接近 +1 表示同向延续，
  接近 −1 表示反向；零运动的方向为 null，不强行定义成稳定。
- 高频比例 `H` 为**固定空间网格速度场**的一侧频谱中，Nyquist 上三分之一的
  能量比例；包含 DC，修正单边能量权重。这不是物体运动频率、不是滤波器。
  所有原生相位都参与；非均匀时间戳不偷偷重采样。只有 16 帧，频谱分辨率有限。
- 全轨迹与通过每一步对应核验的条件子集分别保存。后者不是完整视频的无偏
  估计，缺失不填零；全部记录仍为 `diagnostic_only, score=null`。

下表每格为 **base → seed 1701 / seed 2904**，是全轨迹/场诊断，不是 Repair 分数。
雪地四条来源记录仍为证据不足，其他来源的 8/10 有效配对也没有被升级成 10/10。

| 官方源 | 相邻方向 C | 两步净位移/路程 E₂ | 高频比例 H |
|---|---|---|---|
| 自行车减速 | 0.993 → 0.934 / 0.933 | 0.991 → 0.949 / 0.960 | 0.052 → 0.127 / 0.101 |
| 桥上列车 | 0.233 → −0.927 / −0.925 | 0.787 → 0.175 / 0.169 | 0.309 → 0.854 / 0.866 |
| 雪地自行车（对应不足） | 0.128 → −0.110 / −0.027 | 0.810 → 0.662 / 0.704 | 0.312 → 0.410 / 0.387 |
| 马 | 0.918 → −0.796 / −0.771 | 0.948 → 0.296 / 0.316 | 0.133 → 0.811 / 0.792 |
| 长颈鹿 | 0.876 → −0.726 / −0.755 | 0.954 → 0.401 / 0.378 | 0.182 → 0.753 / 0.773 |

对列车的**可靠条件子集**，CF 的 C 为 −0.991/−0.989，E₂ 为 0.099/0.096。
因此强往返不是仅由少量未核验轨迹造成。自行车原片对应子集的 E₂=0.990。
但不能据此将 C<0 或 E₂ 小写成扣分器：周期运动会触发同样现象，雪地原片也
没有简单的全局同向性。列车 base 的 H=0.309 对应的是极小运动能量，不能仅凭
这个比例将其判作“高频动态视频”。完整数值/条件子集见下述 JSONL/CSV。

### 为什么时间滤波和局部一致性都不够

几何留一点检验：用邻居光流拟合仿射场，预测没有参与拟合的中心点；检查
半径为短边的 1/8、1/4、1/2 和全局。解释能量允许为负，未知为 null。这里
没有语义区域或物体边界，**不把局部可预测性叫成语义结构支持**。

列车 seed 1701 在四个尺度的解释能量为 **0.403 / 0.138 / 0.028 / 0.016**；
长颈鹿同种子为 **0.457 / 0.193 / 0.094 / 0.070**。这支持研究跨尺度结构
支持，但还不支持一个通用尺度阈值：自行车原片 1/8 尺度仅 0.254，马为 0.288，
低于列车 CF；雪地 base/CF 则都约 0.75。**平滑构造本就能在小邻域内协调，
真实关节/小物体运动则未必符合固定邻域仿射模型。**

机制上的区分应是：

| 现象 | 时间轨迹可能怎样 | 还必须核对什么 |
|---|---|---|
| 持续平移/相机移动 | 短时方向有延续，跨间隔仍有位移 | 位移是否由可见区域/结构共同支持；缩放和转动不要求全图同向 |
| 真实周期/关节运动 | 也可往返、高频、净位移为零 | 部件/关节/旋转结构是否持续、协调地改变；保留其完整路径长度 |
| 内部纹理往返抖动 | 相邻方向反复翻转，局部弧长大、跨步位移小 | 实体/部件位姿是否真的变化，还是内部纹理相对结构反复重采样 |
| 跟踪失败 | 漂移、跳变、循环不一致，甚至错误地接近零 | 独立结构锚点、跨帧匹配和可见性，而不是仅看光流自洽 |

上述是需要联合检验的倾向，不是互斥真值标签。仅凭相同 RGB 序列不能保证
区分不同物理成因；这个限制不改变本批 8px 的用户确认，也不允许弱化构造。
[RAFT 原论文](https://arxiv.org/abs/2003.12039)解决的是像素对应/位移估计，并没有
提供“该位移是否应计入语义运动”的判别器；本轮结果正要求把这两层分开。

### 更根本的漏检：列车原片的稠密光流也接近零

逐帧显示已确认列车在通过。随后直接检查**完整稠密场**，排除“只是 12×12
采样网格漏点”这一单独解释。同一 2× 模型，单位均为评分像素/相邻帧：

| 输入 | 全像素平均流长 | 144 点网格平均流长 | 每帧 top-5% 再平均 |
|---|---|---|---|
| 列车原片 | 0.035886 | 0.034777 | 0.122781 |
| 列车 8px / 1701 | 3.747058 | 3.663217 | 9.840737 |

原片全场最大值也仅 1.367527。Origin 的旧 RAFT 各帧 top-5% 仅约
0.175–0.422，未达其 6px 门槛。因此并非只需重调最后的运动阈值：**至少当前
像素流不足以描述这个可见的实体通过过程**。窄目标、重复纹理、遮挡/生成变化
等可能解释漏检，但本轮未分离其因果贡献，不能把某个原因写成已证实。
当前大量背景点的可靠/静止判断不能为这一区域的运动作证。

### 对下一版设计的具体改变

先恢复自然原片的可解释运动，再讨论抑制 CF。不能直接继续在现有全图轨迹
上加滤波，也不能假定新的 DINO/分割模型一接入就会成功。候选数据流为：

```text
单视频 RGB + 原生时间戳
  ├─ 多尺度像素对应：局部位移、双向/多帧证据
  └─ 独立结构锚点：物体/部件与背景区域的位置、轮廓及稠密特征对应
          ↓
  区域运动解释：持续位移 / 镜头运动 / 协调旋转与关节运动
          ↔ 同时检验局部可逆纹理形变能否解释额外位移
          ↓
  跨时间间隔、跨空间尺度与未参与拟合锚点的验证
          ↓
  有证据的结构运动轨迹 → 合成位移向量后累计路径长度
  对应/模型冲突 → 显式不确定性与缺失，不能假静止或宣称通过
```

结构锚点必须由单个视频自动得到，不读取配对 base、seed、位移场或本轮手工
观察。仅全局 DINO 相似度不够：它不能直接给出可解释位移，也可能忽略小物体。
周期运动需要部件/轮廓及内部结构支持，不能只跟物体质心。前景、镜头/背景
运动均计入，不能悄悄将 Dynamic 改成前景相对背景运动或画质评分。

下一步的优先检验：**列车实体位移能否恢复、小幅马/长颈鹿运动能否保留、
雪地复杂运动能否获得足够证据**；再用真实周期/小区域/镜头开发控制约束分解。
只有“时间上往返且缺少与目标结构一致的运动证据”才有资格被解释为干扰；
单一频率、净位移、局部平滑或全局仿射投影都不能单独作判据。
这条新结构证据路径尚未实现或通过验证；G1–G5 门槛、完整十对分母均保持不变。

### 产物和验证

- [完整二十条诊断与 CSV](../../output/dynamic-static-jitter/flow-mechanism-dev5-8px-v2/)，
  含全网格/稠密场、全部相位、可靠条件子集、原来源失败状态；所有 score 为 null。
- [十五条非重复显示输入的逐帧图/轨迹图](../../output/dynamic-static-jitter/flow-mechanism-dev5-8px-v1/)，
  共三十张 PNG；编码控制仍参与数值分析，不重复绘图。本地 PNG 是诊断显示，
  不是新视频或评分输入。图中九个展示 query 按固定几何位置选取，不挑成功轨迹。
- 首次 CPU 诊断 **16:37:41–16:38:28 UTC，47.53 秒**，包括绘图；隔离源码
  `/data/chenjiayu/dynamic-structural-motion-20260922/code-mechanism-v1/`。
  首次 JSONL SHA256：`17da050be50a6b8ff20c177a4ca69a505f0c6b2fa5109afb5e78db86ecaad96c`。
  后续同二十缓存补充全稠密场检查，保留独立 `code-mechanism-v2/` 快照与运行身份：
  **16:44:12–16:44:38 UTC，25.97 秒**，JSONL SHA256
  `0550c68596eb867c9a9c4ee434c5e0591e7ebf63a808cef51f36e01c70a0adb4`。
  原有全部诊断与首次运行逐项相同，不是新增二十条视频或新的模型实验。
- 输入二十个 MP4 SHA、二十个模型缓存 SHA 均绑定前轮身份；源码和分析脚本
  另存哈希。五组原片/编码控制的时间、空间和频率诊断完全相同。
- 新增数学测试验证同向/周期轨迹、全相位唯一统计、条件缺失不填零、留一点
  预测不偷看中心、频谱能量守恒、稠密/网格漏检区分。数学测试不证明自然运动
  非劣或独立不变性；本轮没有新的有效 Repair 分数，更没有完成联合验收。
  该阶段完整 CPU 测试 **919 passed, 3 skipped**；代码/脚本哈希、锁文件、
  编译、diff 与冻结目录检查通过。

## 独立结构证据对照：DINO 描述子仍不能直接计分（2026-09-22）

按上述机制分析接入现成的 DINO ViT-B/16，目的仅为检查独立结构对应能否补充
光流漏检。相同五原片、五编码控制、十干预均完成，**没有新建视频、重跑 Origin、
改变 8px 或打开留出**。所有记录为 `diagnostic_only, score=null`，不是最终 Repair。

### 实际实现与预设配置

模型适配：[dino_dense.py](../../packages/audit-models/src/vbench_audit_models/dino_dense.py)；
对应诊断：[structure_correspondence.py](../../metrics/dynamic-degree/src/dynamic_degree/structure_correspondence.py)；
运行入口：[probe_structure_correspondence.py](../../scripts/counterfactual/probe_structure_correspondence.py)。

```text
单视频全部 16 个原生时间采样帧
  → RGB/255 + ImageNet 归一化；完整画面缩放到 512 长边、16 的整倍数
  → 已有本地 DINO ViT-B/16：第 9 块 keys / 最后一块 patch tokens（两个对照）
  → L2 描述子、明确像素坐标的 32×32 网格、单独记录 CLS attention
  → 全部起点的 1/2/3/4 帧跨间隔余弦匹配，互为最近邻与歧义/峰曲率检查
  → 双向对称亚像素细化；相邻匹配组合与跨两帧直接匹配的一致性检查
  → 条件位移、匹配比例、量化零、三帧一致性及可视化；不输出评分
```

`9`、`11` 是从零开始的块索引。模型严格载入本地 checkpoint，无 hub 下载、
微调或源码改写；权重 SHA256
`bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e`。
输入不是先缩到 256 再放大：本轮保留最高 512 的原生解码图；模型内部统一
512，坐标还原到相应输入像素。两类官方分辨率对应不同原生网格步距，均记录。
描述子推理 float32、L2 归一化后以 float16 缓存；匹配时恢复 float32 再归一化。

初始配置为 [structure.dino-dev-v1.json](../../configs/dynamic-static-jitter/structure.dino-dev-v1.json)。
[Dense ViT descriptors 原论文/项目](https://dino-vit-features.github.io/)为 DINO 的部件/对应
特征提供研究依据，但本轮使用既有 **ViT-B/16**，不声称复现其 ViT-S/8、binning
或全部对应算法，更不以该论文证明本反事实鲁棒。CLS attention 不是语义掩码，
PCA 颜色不是物体分割；当前并未实现区域运动模型。

### 数值纠正与中断记录

单向二次峰细化有一个独立于视频结果的反例：相同特征的左右邻居相似度不对称
时，二次插值也可能偏离自身位置，制造非零位移。已增加**双向相反位移对称化**：
只有互为最近邻且正反峰都可细化时才使用两侧偏移差的一半，否则保留粗匹配并
明确记为未细化。精确相同的描述子产生零位移，时间反转对应相反位移，均有测试。
不删除粗量化为零的记录，也不把“可细化”或“互为最近邻”当成真实运动真值。

修正配置为 [structure.dino-reciprocal-dev-v2.json](../../configs/dynamic-static-jitter/structure.dino-reciprocal-dev-v2.json)。
**复用同二十份模型特征，不追加模型推理**。首次 CPU 重放因 SSH 连接异常中断，
PID 389305 随后只读检查确认已不存在；仅完成 3/20，原目录和记录完整保留。
其 `runtime.json` 末值仍为 running，不能据此宣称进程仍在运行或重放已完成。
终态确认后用新目录 `...-v2b` 完成二十条 CPU 重放，未覆盖旧输出或重复 GPU 推理。
后者是下表的来源；第一次单向版本仅作为历史数值诊断，不作为当前方法结果。

GPU 初次匹配与 CPU 重放在极近相似度峰的末位浮点可能有少量粗匹配差异，
未宣称跨设备逐位 parity，也不把本次校正作为科学效果提升消融。首次实际模型
最后层 patch 输出与原生 `get_intermediate_layers` 首条首帧比对，缓存精度最大
差值为 0；这只证明该提取入口，不证明跟踪/运动正确。

### 五源全部结果：直接描述子位移不具备所需不变性

下表使用 keys（第 9 块）相邻帧**互相匹配子集的条件平均速度**，每个帧对先求
条件均值，再对 15 个帧对平均，单位为短边长度/秒。它的分母随匹配变化，**不
是完整视频运动量，不可与前轮全轨迹 MAE 直接排序**。每格两个 CF 对应 1701/2904。

| 官方源 | 条件位移诊断：base → 两条 CF | 互为最近邻比例：base → 两条 CF |
|---|---|---|
| 自行车减速 | 0.060339 → 0.255927 / 0.277237 | 78.29% → 41.83% / 39.86% |
| 桥上列车 | 0.035940 → 0.228470 / 0.244333 | 85.20% → 49.86% / 46.09% |
| 雪地自行车 | 0.271263 → 0.378009 / 0.402956 | 39.34% → 30.57% / 29.06% |
| 马 | 0.024874 → 0.098107 / 0.104764 | 87.94% → 58.62% / 58.07% |
| 长颈鹿 | 0.086560 → 0.235098 / 0.248323 | 60.57% → 42.00% / 40.23% |

十对均误增。最后层 tokens 对照也未解决：条件诊断均值为
`0.093123 → 0.242481`，keys 为 `0.095795 → 0.247322`。
这不意味着换用 DINO 本身一定不可行，而是明确否定了**直接逐点最近邻位移
累计**已经满足目标的说法。匹配比例不是旧 RAFT 的几何/光度覆盖，不能把本表
改称新的“有效视频率”，也不能用不同定义绕过原有验收门槛。

列车原片部分描述子对应的方向与可见行进方向一致，但还没有独立位置真值
或完整轨迹证明其速度恢复正确。其 attention 加权条件量在 1/2/3/4 帧间隔为
`0.0555 / 0.0365 / 0.0200 / 0.0119`，显示跨间隔不一致；不能以相邻帧的非零
值就宣布找回了真实列车速度。雪地的互相匹配三帧链仅约 12.5%，对应难题仍在。

后续应在现有缓存上检验：保留多个可能对应，在单视频自动取得的物体/部件区域
内建立共同运动解释，并用多帧几何闭合、边界/内部结构移动核验。**不能把最近邻
跳跃当成运动，也不能只取偶数间隔消掉抖动。** 区域提取、运动多假设解算、真实
周期/小区域/镜头运动对照尚未完成；完整分解与标量评分继续不晋升。

### 执行与产物

- 特征推理：H200 物理 **7**，UUID `GPU-5e07f671-a7e1-b38d-4d14-30316dfd9ce8`，
  逻辑 cuda:0，PID 388191；**16:59:57–17:02:18 UTC，141.24 秒**，二十条
  各 16 帧（每条媒体 2 秒、采样跨度 1.875 秒）。当时已有其他进程，使用空余
  显存，没有 kill；启动检查 ≥20 GiB 可用，进程最大 allocated/reserved
  为 512,241,664 / 606,076,928 bytes。Python 3.10.20、torch 2.6.0+cu124。
  隔离代码 `.../dynamic-structural-motion-20260922/code-dino-structure-v1/`。
- 完成的 CPU 重放：PID 390323，**17:05:56–17:07:58 UTC，121.84 秒**，
  20/20 完成，0 失败，源码快照 `code-dino-structure-v2/`。复制该快照时 rsync
  曾报告连接异常，但运行所需文件已落盘；随后已逐项检查全部 60 项执行源码
  与脚本哈希，未改动执行中的快照。部分中断输出与完整 `v2b` 分开保留。
- 本地[首轮文本/模型身份](../../output/dynamic-static-jitter/dino-structure-dev5-8px-v1/)、
  [三条中断重放](../../output/dynamic-static-jitter/dino-structure-dev5-8px-v2-partial/)、
  [完整二十条重放与所有相位/匹配诊断](../../output/dynamic-static-jitter/dino-structure-dev5-8px-v2b/)。
  首轮 JSONL SHA256 `b0e24712948895738d1f9385a7799cd83391180a6b3a57d4c9269061373c8d68`；
  完整重放 `48214c719fcec0832afa3bc715e009ab94396966a2bae71d3fbf28126d001488`。
- 二十份特征缓存及二十份重放匹配缓存保留在远端各自 `output/<run>/evidence/`。
  均逐条核验 SHA、有限数值和全部十六个时间戳；没有声称本地已复制这些大文件。
  两份执行源码各 60 项 SHA 与各自快照匹配；v2b 另与当前本地代码核对一致。
  五组原片/编码控制的全部描述子匹配诊断完全相同。
- [对应图与 attention/PCA 显示](../../output/dynamic-static-jitter/dino-structure-dev5-8px-display-v1/)
  由[绘图脚本](../../scripts/counterfactual/render_structure_correspondence.py)读取绑定
  原视频/特征/匹配缓存生成，不是新评分输入；固定规则取展示点，不按效果挑点。
  没有任何正式新 Repair 分数、自然偏好/周期保留或留出通过结论。

该阶段完整 CPU 测试为 **945 passed, 3 skipped**。十五张显示图已全部同步到本地，
逐张 SHA 与远端绘图清单一致；此前网络中断只影响复制/资产目录读取，不是未完成
模型任务。DINO 大特征缓存仍在远端。

## 自动区域与联合位移候选（2026-09-22，开发中）

下一步不把 DINO 最近邻跳跃直接累计，而是让自动区域中的多个位置共同提出
运动解释。当前已完成 **SAM 20/20 视频、320 帧、0 运行失败**；区域联合平移
的全相位 CPU 重放已完成 **20/20 条、1080 帧对、0 运行失败**，四分片完整取回并合并核验。它们仍是
`diagnostic_only, score=null`，不能改称
20 条有效 Repair，也不替代五源/十 CF 的原有门槛。

### 实际数据流

```text
单视频 RGB + 全部原生采样时间（不读取 prompt、配对原片、seed 或构造场）
  ├─ 本地 SAM ViT-H：逐帧规则点网格 → 自动区域候选，包括背景与小区域
  └─ 已有 DINO block-9 keys 缓存：同一解码画面、明确的描述子网格
       ↓
区域占据每个描述子 patch 的面积比例（不是只取落在掩码内的网格中心）
  → 同一区域全部位置共同拟合平移，完整画面范围搜索，保留三个分离匹配峰
  → 双线性描述子直接对齐、连续位移细化；另用两组棋盘格位置独立拟合
  → 输出候选位移、可观测秩、可见支持、替代峰差与交叉核验误差
  → 全起点 1/2/3/4 间隔诊断；尚未完成跨帧身份、关节/旋转与最终标量
```

模型适配：[sam_regions.py](../../packages/audit-models/src/vbench_audit_models/sam_regions.py)；
区域提取：[probe_motion_regions.py](../../scripts/counterfactual/probe_motion_regions.py)；
联合平移：[regional_motion.py](../../metrics/dynamic-degree/src/dynamic_degree/regional_motion.py)；
重放：[probe_regional_motion.py](../../scripts/counterfactual/probe_regional_motion.py)。

SAM 权重在 H200 已存在，本轮未下载/训练权重。为加载它，在本任务独立 `deps/`
下取得 [SAM 官方源码](https://github.com/facebookresearch/segment-anything)，固定 revision
`dca509fe793f601edb92606367a655c15ac00fdf`；源码 clean，未修改官方 VBench 或现有共享环境。
checkpoint 为 `sam_vit_h_4b8939.pth`，SHA256
`a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`。
显式本地加载、严格 state dict；各 SAM 源文件哈希写入运行身份。

[区域配置](../../configs/dynamic-static-jitter/structure.sam-regions-dev-v1.json)使用官方
ViT-H AMG 默认质量/去重设置：32×32 点、predicted-IoU 0.88、stability 0.95、
box-NMS 0.7、无多层裁剪；仅点批量改为 32 限制显存。不使用类别、建库掩码、
原片掩码传播、最大面积筛选或强制选一个主体，返回后的全部候选均保留。
**SAM 的 stability 是二值化阈值变化稳定性，不是时间稳定性或跟踪置信度。**
逐帧规则点只是模型提示，不是语言 prompt；没有把“8px”变成判分阈值。

### 已完成的区域证据，不是运动覆盖率

下表为每条视频十六帧的候选掩码 **union 面积均值**，每格 CF 为 1701/2904。
这是空间分割覆盖，不是区域身份正确率，更不能与旧 RAFT 对应覆盖直接互换。

| 官方源 | 原片 union 面积 | 两条 8px union 面积 |
|---|---:|---:|
| 自行车减速 | 99.35% | 99.56% / 99.56% |
| 桥上列车 | 60.38% | 60.61% / 68.90% |
| 雪地自行车 | 85.92% | 82.13% / 85.88% |
| 马 | 95.93% | 95.91% / 95.95% |
| 长颈鹿 | 60.83% | 57.34% / 53.63% |

每帧均有候选，但存在未覆盖区域、碎片、重复/嵌套区域和水印区域。加入全画面
运动对照仅为保留背景/镜头解释，**不能把它当作自动填满的可靠运动覆盖**。
已目视检查[列车首帧全部区域](../../output/dynamic-static-jitter/sam-regions-dev5-8px-display-v1/video_000004_regions.png)
和[长颈鹿首帧全部区域](../../output/dynamic-static-jitter/sam-regions-dev5-8px-display-v1/video_000016_regions.png)：
可见列车整体、长颈鹿身体/部分腿部候选，也有大量非物体碎片。这只是代理可视
复核，不是新的人类偏好、盲审掩码真值或跨帧身份标注。

本轮 H200 物理 7 / 逻辑 cuda:0，PID 398795，**17:29:14–17:34:02 UTC，288.14 秒**。
二十条仍是五原片、五编码控制、十已确认 CF；每条 16 个采样帧，未改变输入。
启动要求空闲显存 ≥24 GiB，使用已有进程外的余量，没有终止别人的进程；
峰值 allocated/reserved 为 6,043,027,968 / 6,603,931,648 bytes。
执行快照 `/data/chenjiayu/dynamic-structural-motion-20260922/code-sam-regions-v1/`。

本地[全部区域缓存与身份](../../output/dynamic-static-jitter/sam-regions-dev5-8px-v1/)已同步：
二十个缓存 SHA、无损 bit-pack 还原面积、有限质量值和全部十六时间戳核验通过；
五组编码控制与原片的全部数组逐位相同，61 项执行源码与本地一致。
JSONL SHA256 `c456f464d54fa8a7d7f3f5c83951c2196f26249792dabd76f1a5a67c3f8684b5`。
[十五张全候选区域图](../../output/dynamic-static-jitter/sam-regions-dev5-8px-display-v1/)亦已完整同步、逐张验 SHA。

### 边界偏差纠正与当前重放

首个区域解算版本使用固定出画惩罚，首条原片已显示自行车的左移，但大区域
易停在零：任意亚像素位移都可能让边缘网格点出界，高常数罚项造成零位移处的
不连续壁垒。原来仅测试内部物体平移，没有覆盖全画面微移，测试范围不足。
因此**主动中断这个有偏候选**，不把背景被锁零当作鲁棒性。PID 401175 的身份
核验后发送 SIGINT，随后确认进程不存在；保留[旧部分输出](../../output/dynamic-static-jitter/regional-motion-dev5-8px-v1/)
和不可变 `code-regional-motion-v1/`：2/20 完整记录，144/1080 帧对已计算，第三条
未完成。末次 runtime 的 running 是中断遗留值，不是活进程证据。

当前 [v2 配置](../../configs/dynamic-static-jitter/structure.regional-motion-visible-dev-v2.json)
改为显式可见支持上的条件对齐损失，仍要求至少 80% 源区域支持在界内；全出画
为 null，不填零。不再为出界直接添加大常数。新增全画面 0.15/−0.2 描述子格
平移测试验证微小镜头运动可恢复；这个数学测试不等于自然镜头/周期运动验证。
覆盖条件与条件损失仍需一起读，不能凭较低条件误差声称完整运动已经正确。

v2 在独立 `code-regional-motion-v2/` 中以四个 CPU 分片完成计算，全部使用相同模型缓存，
**17:42:51–17:56:53 UTC**，并行墙钟约 842 秒（不含复制）。四个 PID 为
405414/405426/405421/405417，执行句柄均已返回 exit 0，随后确认进程不存在。
[分片输出](../../output/dynamic-static-jitter/regional-motion-dev5-8px-v2/)已完整取回，
[合并输出与身份](../../output/dynamic-static-jitter/regional-motion-dev5-8px-v2-merged/)
核验为 20/20 条、0 运行失败：每条 54 帧对，共 1080 帧对，全部时间相位均保留。
合并 JSONL SHA256 为 `b5188f630d736e148290a16d71bd715fec0afacaf56b45676576b7908eb9d099`。
五组原片/编码控制的全部区域诊断逐项相同。共 37,653 条区域—帧对记录，包含
嵌套候选与全画面对照；其中 12,588 条完整区域位移为 null，全部保留，不填零。
这些记录不是独立样本数，也不是有效视频评分率；二十条最终 `score` 均为 null。
同步期间网络超时只影响复制；[合并脚本](../../scripts/counterfactual/merge_regional_motion.py)
曾拒绝一次未更新完整的本地分片，完整复制后才成功合并，没有因此重启评分。
最终分数、跨帧
共同轨迹、自然周期/镜头/细小运动响应及联合验收仍未完成，旧默认 Repair 不变。

完成的列车原片已出现一个**足以否定“分出物体＋两组描述子一致就一定正确”**的
反例：首帧 region 0 对应列车整体，0→1 的选择为 `(Δx,Δy)=(+2.410,+1.197)`
原生像素，即向右下；两组独立位置拟合仅相差 0.072 像素，而[原片全帧](../../output/dynamic-static-jitter/flow-mechanism-dev5-8px-v1/video_000004_frames.png)
可见列车向左上通过。0→4 仍仅为 `(+2.482,+1.709)`，没有恢复沿桥行驶的轨迹。
三个替代峰完整保留，未手选一个正确方向作为输出。SAM 区域编号是逐帧局部索引，
例如 CF1701 的列车首帧为 region 6，不能把不同视频的同号区域冒充同一个对象。

首条自行车的背景区域还出现与前景接近的大位移候选，说明同一 DINO 表示内的
两组位置验证不是独立物理证据。接下来须加入原生 RGB/轮廓的独立位置证据及
多帧几何核验；不能仅通过调低区域误差门槛、重新选择显示区域或加一个模型名称
就认定解决。这是原片开发反例，不是二十条最终 Repair 分数的汇总结论；本轮没有输出该分数。

此阶段完整 CPU 检查为 **975 passed, 3 skipped**；锁、编译与 diff 检查通过，
冻结目录 blob identity 不变。SAM/v1 区域/v2 区域分别核对 61/62/62 项执行源码
与对应远端不可变快照一致。测试包含无损掩码、小区域保留、空结果、不同 SAM
导入来源拒绝、直接位移/周期往返、全画面亚像素相机边界、缺失不填零与分片
完整性；仍不包含已完成的自然运动非劣或正式反事实通过结论。

## 原生图像与几何交叉核验（2026-09-22，v1 完成；不是新 Repair）

上一轮已经完成二十条 DINO 区域诊断，结果改变了下一步行动：区域分割和描述子
内部一致仍可能同时给出错误方向。因此新增独立于 DINO 的原生 RGB 检验，而非
继续调整其内部一致性阈值。8px 构造、十条用户确认干预和原有验收门槛不改变。

数据流为：

```text
同一待测视频的完整原生 RGB 帧 + 既有逐帧独立 SAM 自动区域
  → 每个区域在完整平移域搜索原生 RGB 模板平方误差，保留三个候选峰
  → 双线性原生像素对齐，输出位移、可见支持、局部可观测秩
  ├─ 与目标帧全部 SAM 区域逐一计算移位后重叠（不假定区域编号恒定）
  └─ 原始图像独立 SIFT 双向 ratio 对应，保留区域内全部向量及冲突
       → 全起点 1/2/3/4 帧间隔、三帧特征身份闭合与两步路径长度
       → 诊断记录；没有以模板匹配/掩码 IoU/SIFT 数量直接定义运动分数
```

[算法](../../metrics/dynamic-degree/src/dynamic_degree/native_region_motion.py)、
[运行入口](../../scripts/counterfactual/probe_native_region_motion.py)、
[固定开发配置](../../configs/dynamic-static-jitter/structure.native-regions-dev-v1.json)。
RGB 不模糊、不降采样、不归一化局部运动幅度；除以 255 只是颜色量纲变换。
FFT 加速完整的有限域模板平方误差求和，使用显式可见像素分母，非环绕相关或
补零边界惩罚；数学定义可对照 [SciPy correlate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html)
与 [OpenCV 模板匹配公式](https://docs.opencv.org/4.13.0/df/dfb/group__imgproc__object.html)。
80% 可见支持沿用上一候选，0.001 原生像素为数值细化精度，不是运动噪声阈值。
SIFT 使用 OpenCV 默认提取器，双向 ratio=0.75；它仅提供独立对应诊断，不是
8px 分类规则，也不把特征数当视频覆盖。所有小区域、嵌套区域、缺失及全画面
对照保留；全画面对照仍不能自动填充可靠前景。身份闭合不要求速度恒定：真实
往返的两步路径不因净位移为零被删除，但闭合本身也不证明对应的物理真实性。

五条官方原始 MP4 已复制到本地
[official-dev5-native](../../output/dynamic-static-jitter/official-dev5-native/)，逐文件
SHA256 与原 manifest 一致；正式诊断直接解码这些原始字节，不使用预览或编码
控制替代原片。前置快速检验先用已核对解码像素相同的列车编码控制探查平方误差，
随后在原始 MP4 上核对：首帧列车 region 0 的最佳细化位移为
`(-37.569239, -10.678738)` 原生像素，方向与画面可见左上行驶一致，而上一轮
DINO 为 `(+2.410264, +1.196566)`。这是单帧对的方向纠正，**不是有真值标注的
位移精度或整段运动恢复**。多帧重复车厢仍可形成错误匹配峰，SIFT 向量也有冲突；
不能挑选一个向左候选就宣称完整 Repair 通过。

完整二十条/1080 帧对诊断已从本地不可变快照
`output/dynamic-static-jitter/code-native-regions-v1/` 启动：四个 CPU 进程
PID 821404/821465/821477/821501，18:28:34–18:28:38 UTC 启动，每个 OpenBLAS
2 线程、OpenCV 1 线程。输出
[native-regions-dev5-8px-v1](../../output/dynamic-static-jitter/native-regions-dev5-8px-v1/)。
四分片已结束，完成 **20/20 视频、1080/1080 帧对、37,653 个区域帧对**，
0 运行失败；[严格合并结果](../../output/dynamic-static-jitter/native-regions-dev5-8px-v1-merged/provenance.json)
已核对全部分片、输入身份、帧间隔/起点及全画面对照。五组原片/编码控制的
全部区域记录和三帧闭合记录完全相同；20 条分数仍全部为 null。
合并 JSONL SHA256 为 `028ea10afa06c6aebf7f32defc0b50e5bacc94bdd632c1786ddc53daaa75bb77`。
全部区域都存在局部满秩拟合，但这**不代表运动正确或有效评分覆盖 100%**。
四进程 monotonic 耗时分别 2558.57/2559.81/2489.65/2498.71 秒；日志 UTC
跨度为 18:28:34–19:08:59，与 monotonic 不完全相符，保留两种记录，不据此
编造计时原因。每条 16 帧、8 FPS，媒体时长 2 秒、首末采样间隔 1.875 秒。

整段诊断仍有负结果：列车原片首帧到第 2/3/4 帧的最佳 RGB 横移分别
为 −4.341/−24.377/−9.446 px，不能由首个帧对的正确方向推断整段已恢复。
首帧列车区域的 SIFT 对应数随间隔为 7/2/2/1；第一条 8px 干预中对应的
列车区域为 region 6（不是原片的 region 0），对应数为 2/0/0/0。
这些区域通过显示图辨认，仅供事后机制诊断，不能向评分器传入配对掩码或
固定语义区域编号。少量对应与重复外观仍不足以确定完整轨迹。

### 遮挡修正 v2：局部验证已做，全组诊断另行运行

自行车原片揭示了 v1 的具体几何错误：只用源帧背景掩码时，下一帧新进入
背景位置的自行车像素被当成背景外观误差，导致背景随前景一起移动。这不是
仅在 DINO 中存在的问题，原生 RGB 也出现，不能把 DINO 描述子当作唯一成因。
对首帧→第 2 帧，背景横向误位移为 −20.500 px，而区域内独立 SIFT 横移中位数
约 −0.013 px。

[v2 配置](../../configs/dynamic-static-jitter/structure.native-regions-visible-dev-v2.json)
用源、目标两个区域的共同可见像素计算光度误差：先由旧候选位移和未移位重叠
提出目标区域身份，再对每个身份重新搜索**完整平移域**，连续细化也使用目标
可见权重。未移位只提出区域身份，不是强制零运动或零位移优先规则；没有目标
证据时保留缺失。80% 支持、三个峰、数值精度及全相位检查均不改变。

实际局部检查覆盖自行车原片和两个 8px 版本的首帧到第 1/2/3/4 帧，两个区域：
**12 帧对、24 区域拟合，不是全视频评分**。原片第 2 帧间隔的背景横移降为
+0.240 px，自行车为 −21.535 px；两条干预分别为背景 +0.209/+0.212 px、
自行车 −20.673/−21.511 px。原片背景仍有亚像素偏差，不能称精确静止或已通过
不变性；轮子旋转、关节运动及完整轨迹仍未实现。该局部检查的数字来自实际
执行输出，正式逐帧归档以随后全组输出为准。

全组 v2 使用独立快照 `output/dynamic-static-jitter/code-native-regions-visible-v2/`，
输出 [native-regions-dev5-8px-visible-v2](../../output/dynamic-static-jitter/native-regions-dev5-8px-visible-v2/)，
现已全部完成。配置/执行代码/媒体/SAM 缓存 SHA 均在分片 provenance 中；不覆盖 v1。
四个 CPU 分片 PID 为 879357/879359/879360/879362，19:22:36–19:22:37 UTC
启动，OpenBLAS 2 线程/进程，20:03:09–20:03:41 UTC 结束；分片记录的单调钟耗时
分别为 2624.20 / 2637.61 / 2658.21 / 2640.91 秒。UTC 差与单调钟耗时不一致，
两种原始记录均保留，不臆测原因。全部句柄 exit 0，进程均已结束。
[严格合并结果](../../output/dynamic-static-jitter/native-regions-dev5-8px-visible-v2-merged/provenance.json)
为 **20/20 视频、1080 帧对、37,653 区域帧对、0 运行失败**，五组编码对照的区域
诊断及稀疏闭合记录完全一致。区域位移缺失 **3,570** 项；这不是运行失败，也
不是有效视频覆盖率。20 个视频 score 均为 null。诊断 JSONL SHA256 为
`e51b4c6547cd60f972ed193d0a9c011811ccca712d653b0d6f361d1c3253629c`。

已完成的自行车原片及两条干预提供了整段背景诊断：以各自首帧背景掩码为参考，
在**同一视频**内按未移位 IoU 查找后续区域，16 帧均选 region 0，最小参考 IoU
为 0.8104 以上。此选择是已知场景的事后解释，不传给评分器，不作主验收。
每条全部 54 个起点/间隔的背景位移幅值（原生 px；不是速度或视频 score）为：

| 自行车版本 | v1 平均幅值 | v2 平均幅值 | v2 最大幅值 | v2 有值帧对 |
|---|---:|---:|---:|---:|
| 官方原片 | 10.8082 | 0.2336 | 0.3701 | 54/54 |
| 8px，seed 1701 | 20.8302 | 0.4423 | 0.6669 | 54/54 |
| 8px，seed 2904 | 20.5219 | 0.4456 | 0.6844 | 54/54 |

这支持共同可见性消除了该背景的大位移错误，但干预后仍有约 0.21 px 的平均
幅值增加，且只是背景，不是完整运动分数。列车原片第 2→3 帧源区域在现有目标
SAM 区域中未找到足够支持，v2 显式缺失；未移位的最大 IoU 仅 0.0706，而沿
(−40,−12) px 的诊断位移仅 0.0100。随后已导出并逐项查看
[源帧与目标帧全部区域](../../output/dynamic-static-jitter/train-mask-loss-display-v1/frames2-3_all-target-regions.png)：
源 frame 2 / region 0 确为整段列车，目标 frame 3 原图列车仍清楚可见，但 41 个
AMG 提案没有对应整段车体，仅有若干车窗/车端小片段；目标 region 0 已是背景。
这提供了自动区域漏分/身份缺口的直接可视证据，不能硬沿用 region 编号、放宽
覆盖门槛或用缺失当静止。显示脚本/输入/缓存/PNG 哈希保存在同目录 provenance。
随后已实现并实际检验**同一视频内**由区域/位移候选提出提示的重新定位，见下一节。
不得把原片掩码传播给干预，也不把 SAM 置信度当作运动真值。它仍非当前默认评分
路径；未下载权重或修改外部 checkout。
同时去掉了每个匹配假设反复检查整组二值掩码、重复计算面积的开销，未减少区域、
帧或候选。马原片首帧对与不可变 v1 快照的共同诊断字段逐项完全一致，单进程
同次测量从 13.64 秒到 8.16 秒；这是单帧对性能检查，不是全组加速结论。

启动前完整 CPU 检查为 **991 passed, 3 skipped**；锁检查与 diff 检查通过。
新测试核对 FFT 求和与逐像素直接求和、位移符号、恒等零残差、原生亚像素镜头
位移、平坦/空区域不冒充静止、目标区域编号变化、周期往返路径保留、媒体字节
绑定和缓存时间轴。它们仍不是自然周期保留、运动敏感性或正式留出验证。

已准备预定 32 条/8 prompt 的自然开发原片全帧检查，入口为
[inspect_natural_motion.py](../../scripts/counterfactual/inspect_natural_motion.py)。
只读取 `sources.v1.jsonl` 中 `split=dev` 的视频，预留 120 条不打开；输出只作
运动机制审核辅助，不产生评分、自然锚点或自动语义标签。32 条本地媒体与远端
逐文件 SHA 完全一致，32 条的全部帧均已作代理目视复核（24 MP4＋8 GIF），记录见
[开发机制复核](../../configs/dynamic-static-jitter/natural-motion-proxy-review.dev32-v1.json)。
它不是盲审、人类偏好或运动真值标注；马/长颈鹿仍有小幅关节运动，小鸟有翼部
周期，飞机有中心位置变化不大但尺寸明显增长的案例。仅平移/质心不能覆盖这些
运动，不能按 prompt 或 Origin=0 把它们改写成静态锚点。

8 条 CogVideo GIF 的全部帧均缺少时长字段。首次严格时间检查失败的输出保留于
`natural-dev32-display-v1/`；随后显式 display-only 的
[v2 显示](../../output/dynamic-static-jitter/natural-dev32-display-v2/provenance.json)
生成全部 32 条，其中 GIF 时间戳/FPS 为 null。随后 8 条 GIF 的 33 帧也均已
检查：有真实可见拍翼、象鼻往返、物体入画，以及飞机翼尾严重生成形变等情况，
不能将“有变化”直接解释为准确物理轨迹。帧序可支持定性描述，但不能虚构
8/100 FPS 用于速度校准。此处未打开预留测试媒体，不能推断它们的实际时间轴。
人类复核、自然尺度校准及正式留出仍为 NOT RUN。

### 同视频提示重定位与共同几何证据：车体可恢复，运动仍未修好

[提示几何](../../metrics/dynamic-degree/src/dynamic_degree/prompted_regions.py)从同视频
源掩码与各位移候选生成可见框、内部点；[运行入口](../../scripts/counterfactual/probe_prompted_regions.py)
绑定媒体、原生帧像素、SAM 缓存/源码/权重。列车原片 frame 2 / region 0 → frame 3
是已诊断的事后 DEV 案例，不是预先选定的总体测试。六个提示保留 RGB 三个候选、
两条 SIFT 位移及未移位候选；每个提示的 box / box+point × single / multimask
共返回八张掩码，48 张全部保留，没有按目视挑选目标掩码。

H200 物理 GPU 5（`GPU-7d459912-b567-59fb-8abf-57c18713cb84`）实际运行 PID 446354，
2026-09-22 20:08:06–20:08:13 UTC，单调钟 7.10 秒，峰值 allocated 6,043,027,968
bytes。仅处理一对原生 256² RGB 帧，不是整视频计时。复用 SAM ViT-H checkpoint
SHA256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`、
源码 `dca509fe793f601edb92606367a655c15ac00fdf`。执行快照为 H200
`/data/chenjiayu/dynamic-structural-motion-20260922/code-prompted-regions-v1/`；
完整取回的[模型输出](../../output/dynamic-static-jitter/prompted-train-frame2-3-v1/provenance.json)
记录环境与逐源码哈希，mask NPZ SHA256
`2c2dc41e8de66774b3b0ebfac6d96d378cc686498a5eb9755ff9ac1e22f73aa1`。

[全部 48 张掩码](../../output/dynamic-static-jitter/prompted-train-frame2-3-v1-audit/all_prompted_regions.png)
已经逐项查看；一些确实恢复完整车体，另一些只含车端或桥体，均保留。为避免
SIFT 同时产生提示又验证提示的循环，另做 **RGB-only 提示消融**：仅用 RGB / 零
候选产生的 32 张已缓存掩码，保留原 41 个 AMG 提案，共 73 张输入匹配器。
这不是重跑模型，更不是从配对 clean 视频借用掩码。

结果仍不等于修复：最低 RGB 误差选 **(+0.4184,+0.3130) px**，而与可见列车运动
方向一致的候选是 **(−40.4808,−11.6158) px**。RGB MSE 分别为 0.009104 / 0.011715。
共同目标掩码内的独立 SIFT 仅有一个空间位置；其残差为 44.575 / 2.020 px，能将
左上候选排在首位，却不能验证整个车体或整段视频。此候选未收敛也照实保留。
[RGB-only 检查记录](../../output/dynamic-static-jitter/prompted-train-frame2-3-v1-rgb-only-audit/diagnostic.json)
SHA256 `0efadcf6ff1a177979559c875384b97b848c4ea180e81f964437e39f57ce1153`。
该记录 score=null / formal acceptance NOT EVALUATED。不能从一个对应点推广出
“motion 已恢复”或“8px 不变性已通过”。

共同几何检查要求匹配点落入**所有候选目标区域的交集**，防止每个候选自行删除
不利点；同一空间位置的多个 SIFT 方向只算一个见证，不根据速度/方向/8px 筛选。
在原 AMG 区域（没有把上述单例恢复传播到其他视频）上完成了
[全组重放](../../output/dynamic-static-jitter/common-sift-dev5-8px-v1/provenance.json)：
20/20 视频、1080 帧对、0 运行失败，五组编码对照逐字段一致，分数全部 null。
PID 947829 已结束；20:25:17–20:25:46 UTC，记录单调钟 32.30 秒。输出 SHA256
`cdf4a8c480e7ab25c4a871cd1c2380fd3b1418e132d864deb96af79b36ef3bdb`。

| 区域帧对诊断（包含嵌套掩码与编码对照） | 数量 / 37,653 |
|---|---:|
| 有候选且有共同空间见证，可排序但未认证 | 13,863 |
| 无法排序（缺候选或共同见证） | 23,790 |
| 共同空间见证为 0 / 1 / ≥2 | 22,236 / 4,430 / 10,987 |
| 排序首位与 RGB 首位不同 | 6,497 |

这些分母不是独立视频数；“改变候选”也不是纠错数。单例追加的 RootSIFT / AKAZE
对照将共同见证从 1 个增至 4 个，却分别选到约半位移 **(−21.472,−5.786)** 和近零
**(+0.418,+0.313)**，仍不足。两者完整输出分别保存在
[RootSIFT](../../output/dynamic-static-jitter/prompted-train-frame2-3-rootsift-v1/diagnostic.json)、
[AKAZE](../../output/dynamic-static-jitter/prompted-train-frame2-3-akaze-v1/diagnostic.json)。
RootSIFT 是相同 SIFT 描述子的重新归一化，不是独立检测器。

随后[三种空间见证与三帧身份检查](../../scripts/counterfactual/probe_sparse_identity.py)
已完成全二十条：每种 1080 帧对，共 **3240 方法×帧对、0 运行失败**。原生 RGB、
原 AMG 区域及 RGB 候选均不变；SIFT、RootSIFT、默认 AKAZE 使用同一 0.75 ratio
门槛，保留所有检测点。每条新 SIFT 对应均与已绑定的 native v2 逐项完全一致。
三帧检查比较经第三帧的特征索引组合，只保留至少一次身份相同且无已观测冲突
的对应用于排序消融；它不检查恒速、净位移或时间频率，数学往返轨迹可通过。
但这种内部闭合仍不是物理运动真值，也尚未校准为合法评分置信门槛。

执行快照 `output/dynamic-static-jitter/code-sparse-identity-v1/`；本地 CPU、OpenCV
单线程，PID 967055 已结束。记录 UTC 为 20:43:01–20:50:51，单调钟 509.31 秒，
两种原始计时都保留。每视频均保留 16 帧、8 FPS、媒体时长 2 秒。输出
[provenance](../../output/dynamic-static-jitter/sparse-identity-dev5-8px-v1/provenance.json)
的 JSONL SHA256 为 `bcaa549f3c0fd531c2825dc19c12d9ff4c9d05c2ce3f568e0864182cf48d7436`。
[完整性/控制/计数核验](../../output/dynamic-static-jitter/sparse-identity-dev5-8px-v1-analysis/counts.json)
检查二十条、全部方法和全部相位，五组编码对照完整字段完全相同；SHA256
`47f334497fcd77cbf6b8ed968bc186247440c5feccf4b2a3726420361d5b1629`。

| 对应方法 | 原始可排序区域帧对 | 三帧检查后可排序 | 原始 ≥2 空间见证 | 三帧检查后 ≥2 |
|---|---:|---:|---:|---:|
| SIFT | 13,863 | 12,635 | 10,987 | 9,422 |
| RootSIFT | 14,128 | 13,112 | 11,326 | 9,967 |
| AKAZE | 10,135 | 9,444 | 8,648 | 7,870 |

各列分母均为 37,653 个区域帧对（含嵌套掩码和控制），不能当作有效视频比例。
SIFT / RootSIFT / AKAZE 的原始对应数分别为 276,407 / 287,689 / 116,346；
三帧支持且无冲突为 204,006 / 216,325 / 92,940，存在已观测冲突为
378 / 381 / 487，其余 72,023 / 70,983 / 22,919 缺少支持。少冲突并不等于
高精度：未知点不能被当作正确或零运动，全部视频 score 仍为 null。

更关键的负结果来自同一列车补掩码案例：三帧检查把正确跟随车厢连接处的
SIFT 点也丢失了（无第三帧支持），RootSIFT 四点也全为未知；AKAZE 则保留一个
接近静止的窗部匹配，仍选 **(−1.304,+0.122) px**。所有共同点的
[并排图](../../output/dynamic-static-jitter/sparse-identity-dev5-8px-v1-analysis/prompted_case_witnesses.png)
已查看，图上编号是描述子索引，不是物理身份标注。
[逐点三帧证据](../../output/dynamic-static-jitter/sparse-identity-dev5-8px-v1-analysis/prompted_case.json)
SHA256 `4da6d8e8bf2b3c98dab539739eb28678790a3bfe03131b22ee115d38b8d9d503`。
因此不把“增加检测器＋三帧闭合”晋升为成功 Repair；它没有解决可见运动响应。

这个结果把下一步指向**连续跟踪自动结构点**：逐帧重新检测可能丢掉真实连接处，
固定网格也可能没有查询到它。已给原 CoTracker2 适配增加显式 `(t,x,y)` 查询入口，
旧 `track()` 和公共默认不变；不裁切预测出界坐标，原生全部帧参与模型。上游会
强制查询帧的位置/可见性，因此该帧的精确回到查询点不是可靠性证据；内部双向
补全也不等于独立反向循环核验。列车四版本的自动查询与实际试跑如下，仍属事后
机制开发，不代表五源或不变性验收完成。

### 自动特征点连续跟踪与区域放大：两轮实测仍未恢复列车位移

[运行入口](../../scripts/counterfactual/probe_feature_tracks.py)绑定原片、编码对照、
两个 8px 干预的四条媒体 SHA、全部解码像素、时间轴和旧 CoTracker2 源码/权重。
每条独立检测 frame 2 的全部 SIFT 点，只合并完全同位置的多个方向，不要求先
有跨帧匹配，也不手选连接处。自动查询数分别为 **482 / 482 / 486 / 486**。
每条完整 16 帧/8 FPS/2 秒输入同一个现有 CoTracker2；不使用 prompt、配对
clean 位置或构造场。frame 2 是已见反例的事后诊断起点，不能称独立验证。

第一轮直接输入未滤波原生 RGB；相较旧 G1，还改变了查询时刻/上下文及旧输入
blur，因此不是只改点选取的单因素消融。其[完整输出](../../output/dynamic-static-jitter/feature-tracks-train-v1/provenance.json)
已取回核验，四条 0 运行失败、编码对照全部缓存数组完全相同。原片中与连接处
对应的 SIFT key 272 在推理后才被选来解释结果；其 2→3 帧预测位移只有
**(−0.0071,−0.2601) px**，且除上游强制可见的查询帧外全部判为不可见。
它并未跟随连接处向左上移动。全体自动查询在该帧对的平均位移幅值为
**0.1019 → 2.1194 / 2.5426 px**，仍有干预误增；这是自适应特征点诊断，
不是完整视频 Repair 或已认证运动量。后者 score 仍为 null。

第二轮检验“细长目标/背景占比使模型难以匹配”的解释：对**各自** frame 2 的
每个 SAM 区域，由掩码主轴计算固定观察窗口，context=1.5，仿射映射到模型
384×512 输入。窗口对该视频所有帧保持同一矩阵，不能随目标运动而将其稳定掉。
输入窗口内的全部自动查询；预测再用精确逆矩阵还原原生坐标，不按窗口像素
直接计运动，不裁切出界预测。各视频区域独立，未从 clean 传播掩码；小区域、
空查询和失败均保留。[区域映射与运行实现](../../metrics/dynamic-degree/src/dynamic_degree/region_tracks.py)
的尺度、往返路径及空证据行为有纯数学测试，但尚非自然敏感性验证。

[第二轮模型输出](../../output/dynamic-static-jitter/region-tracks-train-v1/provenance.json)
共 **162 个源区域：144 个实际跟踪、18 个缺少查询/几何支持、0 运行失败**。
四视频区域总数为 45 / 45 / 34 / 38，实际跟踪为 40 / 40 / 29 / 35；不是给
162 条独立视频评分。编码对照所有数组完全相同。连接处 key 272 自动落在原片
region 0、17 两个候选内，但原生位移仍为 **(+0.2573,+0.0100)** 和
**(+0.0867,+0.0331) px**，没有恢复约向左上移动的对应。region 17 在下一帧还
被模型判为可见，说明提升模型可见性不能当作纠错。所有保留候选的
[原图轨迹显示](../../output/dynamic-static-jitter/region-tracks-train-v1-analysis/posthoc_anchor.png)
已经查看：圆点仍停在相近图像位置，没有跟随连接处。

| H200 实际运行（物理 7，逻辑 cuda:0） | PID | UTC 起止 | 单调钟秒 | 峰值 CUDA allocated bytes |
|---|---:|---|---:|---:|
| 全图自动特征查询，4 条 | 512004 | 21:01:38–21:01:58 | 20.30 | 1,960,305,152 |
| 各区域固定放大，4 条/144 次区域推理 | 520873 | 21:10:57–21:11:47 | 49.77 | 1,985,406,976 |

GPU UUID `GPU-5e07f671-a7e1-b38d-4d14-30316dfd9ce8`；未终止其他作业。既有权重
SHA256 `362f5274376d610dc987b6daf2c2fefe63e06e1835f4ec1a10d0a15c5a4eef4f` 与
旧 G1 全部 tracker 源文件哈希逐项相同；没有下载或改写上游。执行快照为 H200
`/data/chenjiayu/dynamic-structural-motion-20260922/code-feature-tracks-v1/` 和
`code-region-tracks-v1/`，两进程均已结束。环境 Python 3.10.20、torch 2.6.0+cu124、
OpenCV 4.11.0；原生视频/准备查询在本机与 H200 逐像素、逐查询完全一致。

[全图检查](../../output/dynamic-static-jitter/feature-tracks-train-v1-analysis/diagnostic.json)
SHA256 `c270173e54690b336b3701929ef55e693197d60b477508a1b395536f9d40f18f`；
[区域检查](../../output/dynamic-static-jitter/region-tracks-train-v1-analysis/diagnostic.json)
SHA256 `7d958fda0583860feadd9119908397486414b4bca5ca3fbf1fdf1255f4a2b1b8`。
对应模型 provenance SHA256 分别为
`727cf155384fba4dce9dca127e422c98bbe6d97f8202cf93d245861c1c309f93`、
`da49263f1d536d94c84e2ea219f343dc9bc5254d295ff9f5a83707c70902b707`。
结论：仅增加结构点、连续跟踪或这类区域放大都不能恢复该例运动，不能继续将
旧近零轨迹当作静态真值。下一步需直接验证独特局部外观的原生跨帧重定位，保留
多运动/多对应歧义，而不是继续用模型可见性或重复纹理的内部闭合替代正确性。
完整五源评分、真实周期运动保留、自然尺度校准和独立测试仍未完成。

自动查询/区域放大阶段 CPU 回归为 **1051 passed, 3 skipped（15.58 秒）**；`uv lock --check`、
`git diff --check` 与八个既有 CLI 的 `--help` 均通过。冻结目录的 Git 状态无改动，
其索引 blob 身份仍为 `0648e4ebf351ebc7268f01830615f8e7d964d6eb8c627f84ed4e1ef9084115a2`。
这些检查不是完整评分 parity、人类偏好或独立不变性验收。未提交/推送未完成方法，
goal 保持 active，公共默认 Repair 不变。

### 原生局部外观与形变：恢复个别连接处，但整体对应仍不足

进一步检验同一列车 **frame 2→3**，不把此事后 DEV 帧对当整段评分。四版本仍为
原片、编码控制、两条已确认 8px；所有输入字节/原生时间轴、SAM 缓存身份不变。
[局部外观实现](../../metrics/dynamic-degree/src/dynamic_degree/local_appearance.py)
独立检测每视频的全部 SIFT 点，仅合并同坐标、同尺度的多个方向。每点保留所有
包含它的源 SAM 区域和全画面对照，支持半径为特征尺度的 1/2/4 倍；在完整位移
域做逐通道去均值的 RGB 归一化相关，保留三个正向候选和每候选三个独立反向
搜索结果。可见支持低于 80% 或平坦区域不记完美匹配；不使用构造幅度/相位。
这些是未校准的开发配置，闭合不等于物理身份正确。

[完整四分片](../../output/dynamic-static-jitter/local-appearance-train-frame2-3-v1/)
已完成 **4/4、0 运行失败、4 个帧对、1,936 个自动点**，各视频点数
482/482/486/486、去重计算的支持数 2,106/2,106/2,102/2,405。
[核验和全点显示](../../output/dynamic-static-jitter/local-appearance-train-frame2-3-v1-analysis/diagnostic.json)
重新检测全部点、重建全部支持哈希、检查所有区域/尺度引用及执行源码；编码
对照完整字段与原片完全相同。分析 SHA256：
`094b6d816e42f0cb79b64748c3004fa0979217b5a4ac72ffba73da3882d332dd`。
执行快照 `output/dynamic-static-jitter/code-local-appearance-train-v1b/`，本地 CPU
四进程、每进程 OpenCV 单线程。PID 1031517/1031518/1031519/1031520 均已结束；
记录 UTC 21:41:13–21:45:24，最慢分片单调钟 270.08 秒，保留两种原始计时。

原片连接处两个事后解释点（SIFT 258/259 同位置、271/272 同位置）在区域 0、
尺度 2/4 的候选分别恢复约 **(−44,−13)/(−43,−13) px**，最佳反向也能闭合。
但重复车窗 key 310 的错误近零候选 **(+1,0)** 同样能精确闭合，因此不能把
闭合当成纠错证据，也不能把两个邻近/重叠支持当成独立区域真值。

| 全画面对照、支持尺度 4，单帧对全自动点诊断 | 原片 | 编码控制 | 8px / 1701 | 8px / 2904 |
|---|---:|---:|---:|---:|
| 最佳候选平均位移长度（px） | 3.1811 | 3.1811 | 29.2033 | 32.7610 |
| 最佳反向精确闭合点数 | 435/482 | 435/482 | 184/486 | 175/486 |

以上为自适应点集上的诊断，**不是视频 Repair、配对不变性分数或准确率**。
[全点图](../../output/dynamic-static-jitter/local-appearance-train-frame2-3-v1-analysis/all_points.png)
已查看：局部刚性外观在干预中出现大量远距离错配；不能仅挑正确连接处宣称成功。

随后[局部形变对照](../../scripts/counterfactual/probe_local_deformation.py)固定支持
尺度 4，保留各视频全部点/包含区域/全画面对照及全部三个原候选，分别细化平移
和正行列式仿射（旋转、缩放、剪切）。每次拟合固定初始可见源像素集合，不能
靠将困难像素移出画面降低误差；采用双线性采样的解析导数，不读取反事实参数。
这是局部外观拟合，不是结构运动/干扰分类器；允许变形也可能更好地拟合错误对象。

[四分片结果](../../output/dynamic-static-jitter/local-deformation-train-frame2-3-v1/)
已完成，4/4、0 运行失败、3,217 个去重支持；每模型 9,651 次拟合，其中
9,502 次有拟合结果、149 次证据不足。两模型共 19,302 次不是新增视频数。
编码对照全部字段再次相同。[完整性和事后反例核验](../../output/dynamic-static-jitter/local-deformation-train-frame2-3-v1-analysis/diagnostic.json)
保留全部缺失，不按拟合后误差重新挑候选：表中始终检查原来的 rank-0 候选。

| 全画面对照、原 rank-0 的条件平均位移（px） | 原片 | 8px / 1701 | 8px / 2904 |
|---|---:|---:|---:|
| 平移细化 | 3.3593 | 28.6521 | 32.7684 |
| 仿射细化 | 3.3647 | 28.5270 | 32.6938 |
| 缺少拟合的点（两模型一致） | 1/482 | 5/486 | 1/486 |

这些条件均值仍不能充当有效视频评分。仿射使干预平均相关系数提高约 0.08，
但没有消除全点错配；原片错误窗部候选仍为 **(+0.951,+0.210) px**，相关
系数还上升至 0.9738。正确连接处的局部结果与此负结果同时保留。
执行快照 `code-local-deformation-train-v1/`；CPU PID 1043356/1043358/1043359/1043361
均结束，UTC 21:53:12–21:54:07，最慢分片单调钟 59.69 秒。上述两轮所有
视频 score=null；没有提高 G1 有效评分数，也没有启动独立测试。

下一步需要同视频跨时间、跨区域的运动/形变解释与独立检查。单帧对中的局部
正确点、低外观误差、满秩拟合或反向闭合均不足以证明完整 Repair。

### 跨区域/跨时间共同运动检查：有可预测分量，仍非成功 Repair

复用既有 torchvision 2× RAFT 输出，**20 份稠密缓存已从 H200 完整取回本地并
逐份通过原 SHA256 核验**，位于 `output/dynamic-static-jitter/dense-tv-x2-evidence-local-v1/`。
这是补充旧报告中“未复制到本地”的后续状态，不是新的模型推理。视频、8px、
种子及原有有效评分状态均不变；预留测试媒体未读取。

[新实现](../../metrics/dynamic-degree/src/dynamic_degree/image_plane_modes.py)先检查
整段 Eulerian 光流的时变分量：32×32 固定查询网格，全部 15 个相邻帧间隔参与，
按实际时间加权并单独分析偏离常量速度的变化。每点要求至少 60% 时间具有原有
几何/循环/外观证据；这不是物理真值认证，也不降低旧视频覆盖门槛。中间原生帧
的 SAM 区域按“最小包含区域”划分点的唯一归属，未覆盖点归为 −1；此分区
不是跨帧物体跟踪或人工语义标注。逐视频独立运行，不输入配对 clean 或构造场。

第一轮[全组空间预测诊断](../../output/dynamic-static-jitter/image-plane-modes-dev5-8px-v1/)
已完成 20/20、0 运行失败。分别尝试仿射及 3×3/5×5/9×9 高斯基，每次将一个
完整区域留出，只用其他区域拟合其运动；所有尺度都保留。误差在共同可检查点
集合上比较，缺失不填零。大块区域留出后，空间外推普遍不可靠，例如列车两条
干预所有非线性尺度的解释量均为负。自行车某干预在最细尺度下仅 42/1024 点
可比较；不能忽略其余点后宣称跨区域解释成立。

时变光流自身仍提供一条线索：列车两条干预的第一共同模态占条件能量
**97.48%/97.15%**，有效相邻相位全部反向；马为 **82.81%/84.36%**，长颈鹿为
**83.20%/85.09%**，同样存在共同往返。它们的全局仿射解释较低；但“共同＋往返”
本身不是干扰真值，自然周期运动也可能共享时间模式。

随后不再要求从远处猜测未见区域的空间位移幅度，而检验**区域×时刻的交叉
预测**：时间基只由其他区域估计；目标点的系数只使用其另外两折时间拟合，
第三折从未参与该点拟合。按间隔索引 modulo 3 分折，全部相位各留出一次，
没有删掉偶数/奇数帧，也没有设定构造频率。对照 rank=0 是同样不读取被测时刻
的常量速度，另保留 rank=1/2/3；没有按视频挑最有利 rank。训练目标区域×被测
时刻的数值替换测试确认预测不读取该格数据。该协议验证的是**模型光流的预测性**，
仍不是物理因果标签，其他区域的基础光流也可能有误。

[交叉预测全组结果](../../output/dynamic-static-jitter/region-time-modes-dev5-8px-v1/)
也已完成 20/20、0 运行失败。下表固定 rank=1，数值为共同可检查集合上的
`1 − SSE(rank1)/SSE(heldout-constant)`；**不是 Repair 分数、准确率或修复成功率**。
负数表示不如常量基线，所有失败保留。每条视频总分母为 15,360 个点×间隔。

| 官方源 | 原片预测误差改善量 | 8px / 1701 | 8px / 2904 | 共同可检查点×间隔：原片 / 1701 / 2904 |
|---|---:|---:|---:|---|
| 自行车减速 | −0.0899 | −0.0682 | −0.1697 | 13,337 / 11,665 / 11,716 |
| 桥上列车 | −0.4047 | 0.9744 | 0.9503 | 14,942 / 11,340 / 11,442 |
| 雪地自行车 | −1.6804 | −1.6606 | 0.3260 | 2,867 / 1,055 / 241 |
| 马 | −0.1896 | 0.8601 | 0.8498 | 14,528 / 13,134 / 12,593 |
| 长颈鹿 | 0.1692 | 0.8667 | 0.8670 | 13,902 / 10,787 / 10,340 |

因此列车/马/长颈鹿六条干预存在可进一步核验的共同时间分量；自行车并没有相同
结果，雪地尤其不能由仅 241 个可检查点×间隔作总体判断。rank=2/3 的外推还会
严重不稳定，例如自行车第二干预 rank=3 为 −192.07；不删除这些负结果。
恒速平移、同频仿射镜头、局部周期部件的数学测试，只证明相应解释/数据隔离行为，
不等于自然视频上的周期运动保留已经验证。

[空间轮核验](../../output/dynamic-static-jitter/image-plane-modes-dev5-8px-v1-analysis/diagnostic.json)与
[交叉轮核验](../../output/dynamic-static-jitter/region-time-modes-dev5-8px-v1-analysis/diagnostic.json)
均检查执行快照、20/20 覆盖、缓存哈希，并从实际数组重新计算报告误差；五组
编码控制的全部科学字段和全部数组均完全相同。两轮所有视频 score=null，
旧有效 CF 配对仍为 8/10。**尚未应用运动抑制、生成新 Repair 分数或证明列车
完整真实运动被恢复**；自然周期运动保护、校准、独立留出、人类偏好均未完成。

| CPU 诊断 | PID | UTC 起止 | 单调钟秒 | 执行快照 |
|---|---:|---|---:|---|
| 空间留区域预测，20 条 | 1063195 | 22:11:44–22:11:52 | 7.35 | `code-image-plane-modes-v1/` |
| 区域×时间交叉预测，20 条 | 1069001 | 22:17:20–22:17:29 | 8.80 | `code-region-time-modes-v1/` |

均为本地 CPU、OpenBLAS 单线程；每视频仍是 16 帧、8 FPS、媒体时长 2 秒。
空间轮 JSONL SHA256：`c08b923a7356580021635ccc32953fe4b56e0d9b9a274a9e1a135c4622b6d8e8`；
交叉轮：`2aefc8d64226ddf52a06baecbaf80f2c6377f811331ae66c519115474bb2daf2`。
两份核验 SHA256 分别为
`76fd7c05772e7e44d8d60c7645dd9dd68ae9287adb9b64b79a0d7f0d120cfe49`、
`394893fe311737bed045d0f5ed51014d465eeae35f3970ce894783eff141e190`。

下一步是在单视频内检验可预测分量的结构/图像对应解释，同时保留其仿射镜头
成分；还须补足雪地证据、恢复原片真实运动，再进入完整五源评分。不能直接
把“可预测”当作“可删”，也不能仅将错误光流压回低值称为成功不变性。

本阶段最后 CPU 回归：**1073 passed, 3 skipped（16.34 秒）**；锁文件检查、
`git diff --check` 及八个既有 CLI 的 `--help` 均通过。冻结研究目录 Git 状态未变，
索引 blob identity 仍为 `0648e4ebf351ebc7268f01830615f8e7d964d6eb8c627f84ed4e1ef9084115a2`。
所有上述本轮进程已正常结束；没有下载权重、替换默认 Repair、提交或推送。
goal 保持 active，代码测试通过不替代完整评分和联合验收。

### 边界检查与区域运动保护分解：部分下降，仍不满足不变性

本轮继续使用上述同二十条缓存、32×32 查询网格和全部十五个时间间隔，未修改
视频、8px、Origin、对应核验阈值或有效评分分母。新增实现仍在
[image_plane_modes.py](../../metrics/dynamic-degree/src/dynamic_degree/image_plane_modes.py)，
[运行入口](../../scripts/counterfactual/probe_motion_boundaries.py)提供两个显式开发变体。

**边界假设实测。** 对前三个时间模态分别用已观测的点×间隔拟合幅度；常量速度
单独保留，缺证据的点保持 NaN。四个等间隔点的中央差分减去两侧差分的均值，
作为跨区域运动突变量：仿射和二次平滑场为零，分块平移保留跳变。中间两点位于
不同 SAM 分区、每侧两点同区时才计跨边界；步距 1/2/4 全部报告，不选最有利
尺度。分区仍来自中间帧，并非完整部件身份跟踪；未覆盖的 −1 也单列。

边界轮 **20/20 完成、0 运行失败**。最细步距下，主模态的“突变能量/中央差分
能量”如下；这不是评分或成功率，比例可以大于 1。

| 官方源 | 原片 | CF / 1701 | CF / 2904 | 有证据跨边界 stencil：原片 / 1701 / 2904 |
|---|---:|---:|---:|---|
| 自行车减速 | 1.4324 | 1.8495 | 0.7230 | 14/31；3/31；1/35 |
| 桥上列车 | 1.1683 | 0.1662 | 0.1607 | 179/185；88/188；114/191 |
| 雪地自行车 | 1.7599 | 1.1308 | null | 28/208；8/190；0/169 |
| 马 | 0.9327 | 0.5405 | 0.5197 | 157/166；119/169；135/178 |
| 长颈鹿 | 1.8425 | 0.1346 | 0.1565 | 129/144；106/170；47/121 |

列车/长颈鹿较平滑，但马在较粗步距仍有较大突变；自行车、雪地边界证据不足。
因此没有把“边界平滑”设成删除规则。数学对照还明确展示：自然柔性形变与数字
形变若给出同一个场，此统计完全相同。它不能证明物理因果。

**区域运动保护消融。** 为检验共同模态能否在保留结构运动的约束下被抑制，
实现了以下单视频分解，而非继续扩大低通平滑：

```text
当前视频全部时刻的模型光流 + 自己的 SAM 分区
  → 主时间模态 + 跨区域×时刻预测核验
  → 候选共同往返分量
  → 投影到两个约束的交集：
       每时刻、每区域的仿射运动投影不变
       每个点的原生时间加权平均速度不变
  → 仅减去交集中允许的残差；未知点不作删除
  → 条件模型运动量、覆盖和保护误差（不是完整分数）
```

小于四点或几何退化的区域完整保护。该约束保留模型场中的区域平移、旋转、
缩放、剪切以及全局仿射镜头分量；**不等于证明真实非刚性/周期运动均被保留**。
SAM 分区不完整或跨时身份改变时，保护的也只是该参考分区的代数量。
开发筛选明示为事后候选：主模态能量 ≥0.5、反向比例 >0.5、参与比例 ≥0.1、
可靠点×间隔比例 ≥0.6、跨区域×时间 rank1 预测增益 >0.5，并检查奇异值分离。
不输入配对原片、构造场、相位、8px 阈值或 seed；这些参数不是冻结独立测试协议。

v1 已运行全二十条，但独立核验发现交替投影末步会重新引入应完全保护的小区域
的极小浮点残差，故 **v1 未通过自身保护合约核验，不能用于结论**。输出和快照
保留于 `regional-reversal-dev5-8px-v1/`、`code-regional-reversal-v1/`。v2 把这些
区域直接排除在可删除支持集外，增加“时变小区域可见性”回归测试；不放宽检查。

v2 再次 **20/20 完成、0 运行失败**，五组编码对照的全部科学字段与数组完全一致。
只有列车/马/长颈鹿六条 CF 发生删除，五个原片和其他输入均未删除。六条约束
投影均在 9–13 次迭代收敛；独立 least-squares 核验实际数组中的区域仿射保护，
另核验每点平均速度、未知处零删除和小区域严格零删除，均通过。

下表是同一输入、同一可观测点×间隔分母上的条件速度，单位为短边/秒；不是
Repair score，不能与先前 12×12 网格的 G1 分数混用。各视频的可观测集合不同，
也不能用这些条件均值宣称完整不变性或计算正式 CI。

| 官方源 | 原片：分解前＝后 | CF / 1701：分解前→后 | CF / 2904：分解前→后 | 点×间隔证据比例：原片 / 1701 / 2904 |
|---|---:|---|---|---|
| 自行车减速 | 0.013786 | 0.018364→0.018364 | 0.018414→0.018414 | 89.91% / 82.83% / 83.05% |
| 桥上列车 | 0.001087 | 0.123164→0.078131 | 0.129560→0.075898 | 97.55% / 80.34% / 79.46% |
| 雪地自行车 | 0.102954 | 0.149909→0.149909 | 0.189389→0.189389 | 44.90% / 30.18% / 22.90% |
| 马 | 0.006269 | 0.060548→0.031684 | 0.061268→0.035081 | 95.27% / 88.62% / 86.22% |
| 长颈鹿 | 0.033730 | 0.130652→0.092616 | 0.136874→0.085345 | 91.88% / 76.52% / 75.26% |

六条的条件速度下降约 29–48%，**但远没有回到原片水平**。保留区域仿射分量也会
保留一部分干预，不能为了继续降分直接取消保护。列车原片真实位移仍被基础
对应关系漏掉，雪地仍缺证据；这两个关键问题没有因分解而解决。所有输出
`score=null`，旧 G1 有效配对仍为 8/10，不增加正式有效评分。自然非刚性/周期
保护、五源完整 Repair、自然标定、冻结/留出及本维度人类偏好验收仍未完成。

下一步需先恢复真实结构对应，并用图像证据核验共同形变解释；优先检查重复
车窗的自相似/身份歧义与独特连接处的跨时匹配，而非继续调整分解阈值追求低差值。
数学同步多部件、相机、时变缺失和小区域对照通过，仅证明实现约束，不替代自然验证。

| 本地 CPU 运行 | PID | UTC 起止 | 单调钟秒 | 执行快照 |
|---|---:|---|---:|---|
| 边界，20 条 | 1095055 | 22:42:18–22:42:18 | 0.294 | `code-motion-boundaries-v1/` |
| 区域保护 v1，20 条，自身核验失败 | 1099808 | 22:47:13–22:47:18 | 4.806 | `code-regional-reversal-v1/` |
| 区域保护 v2，20 条 | 1103275 | 22:50:44–22:50:46 | 4.810 | `code-regional-reversal-v2/` |

保留 UTC 与单调钟原始读数，不将不一致的时钟读数相互替换。每视频仍是 16 帧、
8 FPS、媒体时长 2 秒；均为 OpenBLAS 单线程 CPU 缓存重放，没有新增 GPU 推理。
实现基于 `fdf4890` 加工作树；实际运行源码由快照文件哈希绑定。

原始记录：
[边界](../../output/dynamic-static-jitter/motion-boundaries-dev5-8px-v1/diagnostics.jsonl)、
[分解 v2](../../output/dynamic-static-jitter/regional-reversal-dev5-8px-v2/diagnostics.jsonl)。
JSONL SHA256 分别为 `c58d30c05153bb2b2034031e6ce0cda08763b66e7ae0452c2ede84b04055f852`、
`8b60cfa9aa069361fac59e7fdfbab5597d40c647fceac623d3b73e90fbee42c3`；
v1 为 `4ee81c7ba8137b18533712991999fd5b8ccf2d6421918852e3adefe5501f9965`。
[边界核验](../../output/dynamic-static-jitter/motion-boundaries-dev5-8px-v1-analysis/diagnostic.json)
SHA256 `2218424d9dee8f54effd7fe600a6c9ff36b79bf510aef88221a202c25fab8dad`；
[分解核验](../../output/dynamic-static-jitter/regional-reversal-dev5-8px-v2-analysis/diagnostic.json)
SHA256 `04aec2cf226e3d8188a4923ca2bb8acce2514fe5124b0a6c8168983f6b9bcd7e`。

本轮全仓 CPU 回归 **1088 passed, 3 skipped（16.39 秒）**，锁文件检查、八个既有
CLI 与两个新诊断入口的 `--help`、`git diff --check` 通过。冻结目录 Git 状态与
索引 blob identity 未变；manifest/人工确认 SHA 未变。上述进程均已结束；没有
改默认评分、下载权重、提交或推送，goal 保持 active。

### 独立外观峰与全相位区域候选：真实位移仍有身份歧义

在区域保护分解仍残留显著误增后，本轮回到列车真实对应问题，未继续调整抑制
阈值。仍使用官方完整列车、编码控制及两个已确认的 8px 干预（`video_000004–7`），
各自的全部十六个原生帧与十五个相邻间隔都参与。它是五源目标中的对应关系
子实验，不是用四个列车版本代替五源/十条 CF 验收。

[local_appearance.py](../../metrics/dynamic-degree/src/dynamic_degree/local_appearance.py)
新增两个互补检查，所有输入均来自当前视频，不接受配对 clean、seed 或构造场：

```text
每帧全部自动 SIFT 位置/尺度 + 本视频 SAM 区域
  → 区域内原生 RGB patch（半径 = 4×SIFT size；另留 whole-frame control）
  → 跨帧全域 NCC：先找离散局部极大值，再保留最多 12 个独立峰
  → 源 patch 对自己这一帧的 NCC：记录远处相似外观的竞争峰
  → 全区域多位置、多个位移假设联合投票，分别报告线性/平方裕量
  → 全位置与奇/偶查询位置两折的候选、支持位置与歧义；不输出评分
```

此前 greedy 1px 抑制的三个候选可能都落在一个宽峰的肩部；新版本先作 3×3
局部极大值检测，连通平台只保留一个确定性代表。旧三个 greedy 候选同时保留。
本次全部 46,899 个唯一支持窗口的新 top1 坐标/NCC 与旧 greedy top1 完全一致：
**增加独立峰只改变备选，不会自动纠正首选。** 新候选反向搜索 **NOT RUN**，
没有继承旧候选的闭环认证。

自相似对照要求候选足迹与原支持的交集不超过其面积的 .25/.5/.75，三档都保留；
图像内可见支持仍要求至少 .8。这是相对支持范围，不是已知 8px 的阈值。
后续联合候选明确用 .5 档，权重为正的 `cross NCC − self-alternative NCC`，
线性与平方两种权重都报告，不按视频选择有利权重。相同位置的不同特征尺度取
平均、不增加投票数；一个位置的多个假设在某候选处只贡献最大权重。
1px 整数定位邻域不是运动阈值；返回的是投票格点中心，不能将 `(0,−1)` 的
近零簇解读为已测准的向上运动。奇/偶查询位置虽不重复，但 patch 仍可重叠。

**完整运行与分母。** 原生搜索四个 CPU 分片均完成；区域联合候选也完成四个
视频/60 帧对，均 0 运行失败。没有只保留正向位移、成功窗口或特定相位。

| 输入 | 帧对 | 自动位置/尺度组 | 唯一支持窗口 | 区域×帧对组（含 whole-frame control） |
|---|---:|---:|---:|---:|
| 列车原片 | 15 | 7,186 | 11,407 | 575 |
| 编码控制 | 15 | 7,186 | 11,407 | 575 |
| 8px / 1701 | 15 | 7,541 | 11,825 | 520 |
| 8px / 2904 | 15 | 7,427 | 12,260 | 522 |
| 合计 | 60 | 29,340 | 46,899 | 2,192 |

唯一支持窗口均有前向候选与可观测自匹配原点；这不是几何可靠率。两种权重共
4,384 个区域×帧对×消融组全部保留、核验。两轮的编码对照全部科学字段完全
一致。原生核验重建全部特征/区域/支持哈希；联合核验不复用栅格投票代码，
直接逐位置重算权重、支持点、邻域极大值、相位及输入/源码身份。

**保留一个明确负例。** 下表为事后检查的原片 frame 2→3、源区域 0，不是
总体准确率或固定语义区域编号。前两行连接处的向左上对应已由原图检查；第三
行重复窗部的近零对应仍不能解释列车的实体通过过程。

| 源查询位置 | top1 位移 px | 跨帧 NCC | 自帧竞争 NCC（.5） | 裕量 |
|---|---|---:|---:|---:|
| (109.10, 70.87) | (−43, −13) | 0.879407 | 0.863880 | 0.015526 |
| (119.51, 66.87) | (−43, −13) | 0.879392 | 0.871723 | 0.007668 |
| (136.06, 75.10) | (+1, 0) | 0.956435 | 0.934483 | 0.021951 |

错误近零对应的正裕量比两个正确连接处还高，说明自相似裕量不能直接充当物理
身份置信度。此区域共 41 个不同查询位置，18 个没有正裕量；它们只是不投票，
不能被判定为静止或从评分缺失分母中消失。

该例区域联合结果也保留了关键分歧：线性权重仍以近零簇 `(0,−1)` 为首，16 个
位置支持；平方权重的首位变为 `(−44,−13)`，7 个位置支持、空间设计秩为 3。
但偶数位置折选向左上，奇数位置折仍选近零。七个位置集中在 x≈100.7–120.8、
y≈61.3–80.7 的连接处邻域，不能称为七个独立物体/独立图像证据。增加了正确
位移备选的支持，不等于完整对应已经恢复，也不能据一个例子默认采用平方权重。

下一步须保留近零/移动两种身份解释，检验跨帧部件关联及未参与选择的图像证据，
而不是继续凭某个 NCC 裕量或“多点同意”强行选一个。同一个 SAM 区域不必只有
一种运动；应检查局部移动部件是否持续获支持，不能让多数近零查询直接投掉
少数移动部件，也不能把一个正确连接处外推为整区域同速。新候选的反向/跨间隔
核验、完整真实运动轨迹与标量聚合、雪地恢复和自然周期验证仍未完成。
本轮所有 `score=null`；旧 G1 有效 CF 配对仍为 8/10，五源/十 CF 目标未缩小。

| 本地 CPU 运行 | PID | UTC 起止 | 单调钟秒 | 执行快照 |
|---|---:|---|---:|---|
| 原片相邻全相位 | 1115262 | 23:01:14–23:15:24 | 922.825 | `code-appearance-ambiguity-train-v1/` |
| 编码控制相邻全相位 | 1115263 | 23:01:14–23:15:35 | 933.655 | 同上 |
| 1701 相邻全相位 | 1115264 | 23:01:14–23:15:43 | 944.361 | 同上 |
| 2904 相邻全相位 | 1115265 | 23:01:14–23:16:10 | 971.435 | 同上 |
| 区域联合，四版本/60 帧对 | 1132794 | 23:17:26–23:18:42 | 81.636 | `code-appearance-consensus-train-v1/` |

四个原生搜索进程并行、各一个 CPU 线程；每视频仍为 256×256、16 帧、8 FPS、
媒体时长 2 秒。保留 UTC 和单调钟原始读数，不互相替换。没有新增模型推理或
权重，源码仍是 `fdf4890` 加任务工作树，由各快照文件 SHA 绑定。

原始[四分片结果](../../output/dynamic-static-jitter/appearance-ambiguity-train-allphase-v1/)
的 JSONL SHA256 按 shard 0–3 依次为：

- `e5724849bb7741898991e0f466b39df5dfd820f998b8ad2da64105265f94e92b`
- `8cd76459ed1a629a489b92b0cc3ab45140d06c2eb2c37de4b75bb00a21282b9c`
- `54dd2e4f6f4e26708c67ce70b5a75bfe35d2b1e250959a6cd67e857c0e1eaaa5`
- `6b3330e7a8bb9ddf1f6e269429c22a084a4f16ba0eda409de22f2db5131b74ff`

[原生核验](../../output/dynamic-static-jitter/appearance-ambiguity-train-allphase-v1-analysis/diagnostic.json)
SHA256 `bfaf678a77ae6ae51bd7fdf5c338c9f834d4c9bf12a6e4f4c07fdbf99e0ab0a6`。
[联合候选](../../output/dynamic-static-jitter/appearance-consensus-train-allphase-v1/diagnostics.jsonl)
SHA256 `3948f0167a99e894f82bf459c5b254f64464ae4fe08b44c5a4523775ada9531d`；
[独立核验](../../output/dynamic-static-jitter/appearance-consensus-train-allphase-v1-analysis/diagnostic.json)
SHA256 `5c06489d53a423a31b20c1584ed7ad5470b0804983e4d78badda17e4ff54657d`。
复现入口在[配置说明](../../configs/dynamic-static-jitter/README.md#distinct-appearance-peaks-and-source-self-ambiguity-dev-correspondence)。

本轮最后 CPU 回归 **1099 passed, 3 skipped（16.64 秒）**；锁文件、八个既有 CLI
及四个相关研究入口的 `--help`、`git diff --check` 均通过。输入 manifest、人工
确认清单及冻结研究目录未变；所有本轮进程均已结束。未改默认 Repair，未下载
权重或提交/推送。代码与诊断通过不等于科学验收，goal 保持 active。

### 三帧关联与固定模板路径：重检断链不是唯一问题

该轮只检验跨帧对应，不把路径数、相关性或闭合率当作运动分数。输入继续是
同一列车原片、零编辑编码控制、两条 8px 干预：每条 256×256、16 帧、8 FPS、
媒体时长 2 秒。所有 14 个连续三帧起点均参与，不按成功窗口或奇偶相位筛选。
五源/十 CF 的最终目标不缩小，雪地证据与自然周期运动仍未解决。

**A. 重检关联。** 复用上一轮每块源支持的全部 12 个相邻匹配峰；将落点与
下一帧的全部自动特征位置关联，半径为 `max(1px, .5√(size_source·size_middle))`。
关联只提出身份假设，不直接认证；首步改为实际中间特征位置后重新检查 RGB。
保留每个兼容中间点的全部支持/次步假设，并检查最初模板对第三帧的匹配。
按三个 NCC 的最小值保存前三条路径，全部备选数组仍保留。一个 SAM 区域允许
多个运动分量；两步返回的路径保留两步位移，不按净位移判静止。

| 输入 | 连续三帧窗 | 全部路径 | 三项相关性均可计算的路径 | 首步假设／有中间重检点 | 首选像素复算组 |
|---|---:|---:|---:|---:|---:|
| 列车原片 | 14 | 466,056 | 436,410 | 128,736 / 20,924 | 8,751 |
| 编码控制 | 14 | 466,056 | 436,410 | 128,736 / 20,924 | 8,751 |
| 8px / 1701 | 14 | 439,956 | 404,485 | 133,872 / 20,041 | 8,580 |
| 8px / 2904 | 14 | 477,084 | 436,550 | 138,504 / 20,375 | 8,789 |
| 合计 | 56 | 1,849,152 | 1,713,855 | 529,848 / 82,264 | 34,871 |

四分片完成、零运行失败。仅约 15.5% 的首步假设有可关联中间检测；这个分母是
全部多假设，不是真实运动点，所以**不是跟踪准确率或丢失真实运动的比例**。
缺失不填零。独立核验检查所有路径的候选成员资格、无重复、完整数量、实际
位移合成、关联半径、全部排名；另从原图复算每个查询/支持组首选路径的三个
相关性，共 34,871 组。不是全部路径像素重算，也没有证明全局峰或物理身份正确。
编码控制的全部科学字段和全部数组与原片完全一致。

明确负例仍为事后选择的原片 frame 2→3→4、源区域 0：位置 `(109.10,70.87)`
的首选变成首步 `(-2.143,-0.382)`，未保住之前检查到的约 `(-43,-13)` 实体位移；
位置 `(119.51,66.87)` 的首步约 `(-42.247,-12.596)`，次步却接上 `(11,4)`，
不能解释同一连接处继续通过。重复窗部 `(136.06,75.10)` 无中间关联，不能称
静止。这些现象要求排查模板身份漂移，而不是放宽断链门槛。

**B. 固定模板对照（已完成）。** 不再要求中间帧重新检出特征：直接用原始
支持窗口对第三帧做全域搜索，保留 12 个独立极大值，与原有 12 个首步候选
做笛卡尔组合。在三个放置位置的共同可见源像素上，重新计算三个两两 RGB
NCC，避免更换为另一块中间模板。所有 SAM 支持和 whole-frame control 都保留，
相同掩码只去重计算；排名仍非物理置信度。

| 输入 | 连续三帧窗 | 唯一支持组／首选像素复算组 | 全部路径 | 三项相关性均可计算的路径 |
|---|---:|---:|---:|---:|
| 列车原片 | 14 | 10,666 | 1,535,904 | 1,522,970 |
| 编码控制 | 14 | 10,666 | 1,535,904 | 1,522,970 |
| 8px / 1701 | 14 | 11,115 | 1,600,560 | 1,584,844 |
| 8px / 2904 | 14 | 11,503 | 1,656,432 | 1,638,534 |
| 合计 | 56 | 43,950 | 6,328,800 | 6,269,318 |

四分片完成、零运行失败。所有源支持均保留 12×12 路径；独立核验检查全部
笛卡尔枚举、位移、排序和原生时间轴，并从原图复算全部 43,950 个支持组首选
的共同可见像素 NCC。编码控制全部字段/数组精确一致。A 按查询＋支持计路径，
B 按唯一支持计路径，**两者原始数量不能直接解释为覆盖提升或正确率**。

同一事后原片窗口暴露了限制，以下均为 B 的首选路径，不是人工轨迹真值：

| 源位置 | 首步位移 px | 次步位移 px | 三项 NCC 最小值 |
|---|---|---|---:|
| (109.10,70.87) | (+1,0) | (+19,+6) | 0.852691 |
| (119.51,66.87) | (−43,−13) | (+11,+4) | 0.863240 |
| (136.06,75.10) | (+24,+7) | (−59,−17) | 0.923823 |

之前已检查的连接处真实首步并未稳定保住；第二行虽保住首步，次步仍跳向相似
纹理。第三行得到较高相关性，仍不能证明跟踪了同一个窗部。由此排除“仅需
取消中间特征检测即可恢复真实运动”的解释；不据更大的路径长度宣称恢复。
三次相关性共享图像与模板，也不是三份独立物理证据。

下一步需检验多个局部结构/边界在空间与时间上的共同运动解释，并用未参与
拟合的图像证据比较身份假设。应允许一个区域内的多个部件、加速与周期路径，
不能硬锁恒速或再次只按单窗口最高 NCC 选择。完整五源评分、雪地证据、
自然周期/镜头/小运动验证、尺度校准及独立测试仍未完成；G1 仍最多 8/10，
所有新视频 score=null，默认 Repair 不变。

实现与证据：

- [路径算法](../../metrics/dynamic-degree/src/dynamic_degree/appearance_paths.py)、
  [运行入口](../../scripts/counterfactual/probe_appearance_paths.py)、
  [独立路径核验](../../scripts/counterfactual/audit_appearance_paths.py)。
- [A 四分片](../../output/dynamic-static-jitter/appearance-paths-train-allphase-v1/)，
  执行快照 `code-appearance-paths-train-v1/`。PID 1155980/81/82/83 均已结束，
  UTC 分别为 23:39:10–23:40:20、23:39:10–23:40:20、23:39:11–23:40:32、
  23:39:11–23:40:38；单调钟 74.550 / 74.897 / 86.768 / 95.029 秒。
  均为 2026-09-22，本地四 CPU 单线程并行；两类时间原始值分别保留。
- [A 核验](../../output/dynamic-static-jitter/appearance-paths-train-allphase-v1-analysis/diagnostic.json)
  SHA256 `4f6daf78ce08f860b48bd644f93dd7a4bf10b5d9ab05d522e70d9dad6b8bad36`。
- [B 四分片](../../output/dynamic-static-jitter/appearance-reference-train-allphase-v1/)，
  快照 `code-appearance-reference-train-v1/`。PID 1168714/15/16/17 均已结束；
  UTC 从 2026-09-22 23:52:01/01/02/02 到 2026-09-23
  00:01:39/00:01:36/00:02:01/00:02:34，单调钟
  627.677 / 624.734 / 649.168 / 684.970 秒。本地四 CPU 单线程并行，
  不将 UTC 差值改写成单调钟值；输入组/媒体时长同 A。
- [B 核验](../../output/dynamic-static-jitter/appearance-reference-train-allphase-v1-analysis/diagnostic.json)
  SHA256 `aeaceec011e63c41227eade80ec87fac28532c87f92eecfc9aec3622c64639a5`。

两轮实际代码为 HEAD `fdf4890` 加相应工作树快照；逐文件 SHA、输入/SAM/父级
证据身份和所有 NPZ 哈希均绑定在 provenance。B 运行后只补齐了空查询场景的
输出数组字段；本批每窗至少 439 个源查询，不涉及该分支。相应缺失合约新增
测试，不能把未重跑的当前工作树冒充 B 执行版本。

本轮最后 CPU 回归 **1115 passed, 3 skipped（15.91 秒）**；锁文件、八个既有
CLI 和两个路径研究入口 `--help` 均通过。构造 manifest、人工确认清单与冻结
研究目录不变；未下载权重、修改上游、替换默认 Repair、提交或推送。所有本轮
进程均已结束，goal 保持 active；实现正确与科学验收仍分开报告。

### 不相交空间证据检查：扩大平移支持仍会丢掉运动

三帧检查后进一步检验：一个局部位移是否被**未作为源匹配模板**的邻近结构
支持。仍只用同一待测视频，没有配对原片、构造场、seed、运动方向或 8px 幅度
输入。四个版本/全部 15 个相邻帧对，不筛选窗口；没有更换或减弱干预。

数据流为：

```text
同视频原生 RGB + 自己的 SAM 区域 + 已完成的全部局部匹配候选
  → 原源窗口 core，半径 r = 4 × SIFT_size
  → 同一源区域内两个不相交环带 (r,2r]、(2r,4r]
  → 每个原候选 + 零位移控制；不在环带上重新搜索或拟合
  → 三处分别计算 NCC / RGB MSE / 同可见像素上的零位移比较 / 可见比例
  → core、core+内圈、core+外圈、core+两圈四种排名，所有缺失与备选保留
```

原窗口和两圈在源图像上互不重叠，并裁剪在各自同一 SAM 区域内，whole-frame
control 另保留。候选是原窗口提出的全部 12 个离散峰，零位移若已存在不重复
添加；不以位移大小删候选。多参考恰好共享三个掩码时只去重计算，全部原位置/
区域引用保留。联合排名用相应 NCC 的最小值，仍不是经过校准的运动置信度。
每个候选与零位移 MSE/NCC 比较使用相同可见源像素，但不同候选可见集合可能
不同，比例原样保存；最低可见比例仍为 80%。平坦 NCC、少于三像素和空环带
保留缺失，不填零。没有对视频运动做删除或输出最终 Repair 分数。

“不参与源模板搜索”不等于独立数据或独立物理证据：SIFT/SAM 来自相同图像，
全域搜索也读取同一目标帧。也不能要求同一 SAM 区域必然只有一个刚性部件，
关节、遮挡和多个部件都可能使环带不服从中心平移。本轮正是检查此限制。

**实际产物与分母。** 四分片均完成、0 运行失败、编码控制全部科学字段和数组
与原片完全一致。每条仍为 256×256、16 帧、8 FPS、媒体时长 2 秒。

| 输入 | 帧对 | 唯一 core/两圈几何组 | 候选比较 | 内圈联合可排名组 | 外圈联合可排名组 | 两圈联合可排名组 | 原图复算候选 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 列车原片 | 15 | 12,271 | 149,041 | 11,985 | 11,604 | 11,604 | 15,965 |
| 编码控制 | 15 | 12,271 | 149,041 | 11,985 | 11,604 | 11,604 | 15,965 |
| 8px / 1701 | 15 | 12,624 | 163,142 | 12,367 | 12,002 | 12,002 | 35,362 |
| 8px / 2904 | 15 | 14,312 | 185,027 | 14,078 | 13,702 | 13,701 | 41,811 |
| 合计 | 60 | 51,478 | 646,251 | 50,415 | 48,912 | 48,911 | 109,103 |

所有组的 core 均有可排名候选；两圈联合有 2,567 组无排名。这些计数包含零
位移控制，“可排名”只表示至少一个相关性可算，不是真实对应覆盖、静止判断
或 Repair 有效评分。区域/位置/尺度仍有相互依赖，不能把组数当独立样本量。

独立核验用径向不等式重建所有掩码，不调用运行器的环带构造；检查所有引用、
几何哈希、候选成员、零位移去重、全部排名、时间轴及输出文件完整性。再从
原图直接中心化像素重算每组四种排名的不同首选与零控制，共 109,103 个候选，
每个都检查 core/两圈的 NCC、MSE、零控制与可见比例。不声称所有非首选像素
都重算过，也不把程序核验当物理运动真值。

**保留具体反例。** 下面仍是事后查看的原片 frame 2→3、源区域 0，不代表
总体正确率；前两行连接处的向左上候选来自之前原图检查。

| 源位置 | core 首选 px | 加内圈首选 px | 加外圈首选 px | 两圈首选 px |
|---|---|---|---|---|
| (109.10,70.87) | (−43,−13) | (+1,0) | (−43,−13) | (+1,0) |
| (119.51,66.87) | (−43,−13) | (0,0) | (0,0) | (0,0) |
| (136.06,75.10) | (+1,0) | (0,0) | (0,0) | (0,0) |

第一行正确方向候选的 core/内圈/外圈 NCC 为
`0.879407 / 0.856257 / 0.874318`；近零 `(1,0)` 为
`0.859388 / 0.877955 / 0.874162`。扩大支持并未稳定确认真实局部移动。
第二行 `(-43,-13)` 的最外圈可见比例只有 **0.715488**，按原 0.8 规则保持
缺失；其源最外圈共 2,376 像素。不能取消门槛让这个例子通过，也不能因剩下
零位移候选可算就宣布整个部件静止。MSE/零控制也全部保留，没有只展示有利
的 NCC 或空间范围。

结论：**不将扩大同一区域的平移一致性当作新的抖动删除门槛**。下一步应针对
可见的边界/部件提出独立运动和遮挡/形变解释，核验哪些像素确属同一运动单元，
而不是继续让更大区域的近零票数抹掉小部件。它还须与单视频干扰分解联合，
并回到五源/十 CF、雪地、真实周期/镜头/小运动与自然偏好完整验收。本轮所有
视频 `score=null`，旧 G1 最多 8/10 不变，五源完整 Repair 仍未通过。

实现：[核心算法](../../metrics/dynamic-degree/src/dynamic_degree/appearance_context.py)、
[运行器](../../scripts/counterfactual/probe_appearance_context.py)、
[独立核验](../../scripts/counterfactual/audit_appearance_context.py)。
[四分片产物](../../output/dynamic-static-jitter/appearance-context-train-allphase-v1/)，
执行快照 `code-appearance-context-train-v1/`，版本仍为 `fdf4890` 加任务工作树。
四 PID 1195612/1195614/1195615/1195617 均已结束；UTC 从
2026-09-23 00:15:57 到 00:18:27 / 00:18:28 / 00:18:39 / 00:18:56，
单调钟分别 162.239 / 163.652 / 176.690 / 193.566 秒。
本地四 CPU 单线程并行，UTC 和单调钟读数分别保留。父级输入/区域/源码与本轮
逐文件哈希均由 provenance 绑定，没有新增模型推理或下载权重。
[核验记录](../../output/dynamic-static-jitter/appearance-context-train-allphase-v1-analysis/diagnostic.json)
SHA256 `50c1c5f4844f46d71312207c2fc49ef0468176f7c30c2a44a227c3dabd02cec0`。

本轮 CPU 回归 **1125 passed, 3 skipped（16.81 秒）**；锁文件、八个既有入口
和两项 context 研究入口的 `--help` 通过。冻结输入/结果、默认 Repair 和已有
他人改动不变，没有 commit/push；goal 保持 active，独立验证仍 NOT RUN。

### 线段与交点对应：完整五源的稀疏几何仍不足

本轮不再要求整个 SAM 区域同速移动，尝试直接定位两条可见边界的交点。
交点可以沿固定长边界移动，也可以往返；它不等于线段端点或物体质心。
同五原片、五编码控制、十条干预均使用原生全部 16 帧、8 FPS、2 秒媒体，
不改 8px，不读取配对 clean、构造场、种子或人工指定的连接处坐标。
相邻及 2/3/4 帧间隔的所有起点均参与，共 **20 视频、320 帧、1080 帧对**。

实现为 [line_structure.py](../../metrics/dynamic-degree/src/dynamic_degree/line_structure.py)，
[运行入口](../../scripts/counterfactual/probe_line_structure.py)与
[独立核验入口](../../scripts/counterfactual/audit_line_structure.py)。数据流：

```text
当前视频原生 RGB / 时间戳
  → OpenCV LSD_REFINE_STD，保留全部线段及测量宽度
  → 每段 32×9 RGB 条带；沿线覆盖端点间长度的 1.5 倍，法向半宽为测量宽度的 3 倍
  → 全域 RGB NCC，显式共同可见像素 ≥80%，比较两种端点方向
  → 每源保留三个最佳不同目标；双向最近邻且两侧描述子距离比 <0.75
  → 两条已匹配线段的交点对应；有限段两端只延伸一个测量宽度
  → 所有可用第三帧的身份组合检查；原始 / 有支持 / 冲突 / 缺失分别保留
```

没有先删短线、设置速度门槛、限定运动方向或对往返位移相消。条带匹配是自建
开发候选，不是某个预训练模型已认证的物理对应；可见交点也可能是遮挡边缘的
投影相交，不自动等于物体关节。没有输出最终标量分数或替换默认 Repair。

**全相位覆盖。** 共检测 141,391 条线段、70,142 个交点；跨帧源查询分别为
479,033 和 238,282 个。接受 35,349 个线段对、3,319 个交点对；其中有第三帧
身份支持且无冲突的分别为 19,914、1,103 个。查询随帧/间隔重复，不是独立样本，
这些比例不是准确率，更不能代替 G1 的有效视频覆盖率。

以下另列全部 **相邻帧**，每格是“已匹配交点数 / 全部源交点查询数”，括号内
为通过第三帧身份检查的匹配数。它不是分数表，也不是不变性通过率。

| 官方源 | 原片 | 8px / 1701 | 8px / 2904 |
|---|---:|---:|---:|
| 自行车减速 | 25/1370（13） | 9/1652（0） | 2/1646（0） |
| 桥上列车 | 147/2547（71） | 12/2994（4） | 9/3093（0） |
| 雪地自行车 | 28/3313（8） | 4/3576（0） | 4/3691（0） |
| 马 | 270/6484（68） | 34/7280（6） | 27/7326（6） |
| 长颈鹿 | 17/1761（2） | 5/1926（0） | 4/1886（0） |

五组原片与编码控制的全部科学字段精确一致。CF 并没有缺少检测到的几何点，
但匹配大量丢失；**七条 CF 连一个有第三帧支持的相邻交点对都没有**。
不能把空集合均值记为零，或只用剩余点宣称“降分且保持真实运动”。
完整条件运动量保留在核验 JSON，空集合为 null；观测集合发生变化，不作有效
配对评分，不算正式 CI。全部二十记录均为 `diagnostic_only, score=null`。

**列车事后定位。** 原片 frame 2→3，显示用区域 `80<x<150, 40<y<90` 内有九个
源交点，但一个也未接受。连接处附近源线段 26 的第一候选是目标线段 40，
NCC=0.775520，但描述子距离比 0.959804，不满足预设 0.75；其他邻边的排序也
有歧义。检测出的线段长度、边界支持和相交方式会随帧变化，不能只放宽一个
比值就称为恢复。这些坐标仅用于结果后的诊断，不进入算法，也不是人工光流
真值；附近任意目标交点不被自动标为“正确对应”。

结论：**“两条边各自先唯一匹配，再取交点”的稀疏门槛不能用于当前 Repair**。
下一步须检验遮挡/边界分裂条件下的局部连接关系联合假设，先确认可见结构对应
是否存在，而非继续收紧/放松逐线门槛或用剩余背景点认证整个视频。仍需解决
真实原片运动恢复、全十条 CF 的有效评分、自然运动保护及独立验收。

产物位于 [line-structure-dev5-8px-v1](../../output/dynamic-static-jitter/line-structure-dev5-8px-v1/)，
四个 shard 均为 5/5 视频、270 帧对、0 运行失败。执行快照
`output/dynamic-static-jitter/code-line-structure-dev5-v1/`；基于 `fdf4890` 工作树，
实际源码逐文件 SHA 随 provenance 保存，不能只引用 HEAD 当作执行版本。

| CPU 分片 | PID | UTC 起止（2026-09-23） | monotonic 秒 |
|---|---:|---|---:|
| 原片 | 1227084 | 00:46:09–00:49:03 | 189.157 |
| 编码控制 | 1227085 | 00:46:09–00:49:05 | 190.189 |
| seed 1701 | 1227086 | 00:46:09–00:49:30 | 218.197 |
| seed 2904 | 1227087 | 00:46:10–00:49:32 | 219.998 |

四个本地 CPU 进程并行，每进程 OpenCV/OpenBLAS 单线程；UTC 与单调钟读数
按原值保留，不互相替换，不冒充四卡计时。所有进程已结束；没有新权重或 GPU 推理。

[独立核验 JSON](../../output/dynamic-static-jitter/line-structure-dev5-8px-v1-analysis/diagnostic.json)
SHA256 `7b3a9c37c11e6886639c1cd4848651f72891f3fca10aab594cec284174583302`：
核验了输入/代码身份、原生时间轴、全部帧/帧对、全部检测线段和交点几何、全部
匹配/缺失及第三帧组合、五组编码控制；**35,349 个接受线段对**全部从原始 RGB
独立复算 NCC。另按预定索引每帧对选一个源线段，对全部目标独立复核全域排序，
共 **1080 查询、477,916 个目标比较**，不是所有拒绝查询的穷尽排序复算。
一致输入的 NCC 浮点舍入经开平方会放大，因此独立核验比较距离比的平方方程，
不改匹配阈值或实验结果。

新增 15 项测试覆盖共同可见 NCC、端点换向、小位移、旋转、重复外观、空/出画、
交点沿固定边界往返、完整相位和篡改拒绝。完整 CPU 测试 **1140 passed, 3 skipped**；
`uv lock --check`、`git diff --check`、八个既有 CLI 及两个新入口的 `--help` 通过。
冻结目录状态和索引哈希未变；工作树算法/runner 与执行快照、auditor 与核验身份
逐项一致。数学与像素复算只证明实现/记录一致，
不证明物理身份、自然非劣或联合 Repair 成功。

### 下一独立跟踪器对照准备：未下载，NOT RUN

在上述完整线段实验后，额外做了列车原片 frame 2→3 的连接处描述子快速检查：
以全部检测交点为查询，KeyPoint 尺度为两条线宽几何均值的四倍，沿两条线各取
正反方向，用 OpenCV SIFT 描述子作 RootSIFT 归一化；同一个物理查询的四个方向
视为备选，不把方向当不同目标。这个单帧对检查中，已检查的连接处查询仍会
首选重复窗部或近零候选，未提供足够理由晋升。**这只是控制台预检，不是新
封装后的全五源候选或一轮可用评分**，不能与上一节 20/20 的完整诊断混用。

现有 CoTracker2、多尺度 RAFT、DINO 和局部几何路径都尚未恢复足够的真实结构
运动，因此下一步建议改变跟踪表示，比较 **CoTracker3 offline**。这是下一轮
假设，不是“换新模型就能修好”的结论；即使对应更准确，也可能更准确地跟随
抖动，仍须验证运动分解、完整十条覆盖和真实运动保护。
[官方实现](https://github.com/facebookresearch/co-tracker)给出离线模型及其权重，
[论文](https://arxiv.org/abs/2410.11831)讨论独立的预训练点跟踪，不为本任务的
8px 不变性提供实测保证。

H200 的以下目录已只读检查：`/data/chenjiayu/wenbiao_zhao/models/`、
`/data/chenjiayu/.cache/{torch/hub,huggingface/hub}/` 及两个 Dynamic 任务目录。
检查范围内找到现有 CoTracker2/RAFT/DINO，但没有找到官方 CoTracker3
`scaled_offline.pth`/`scaled_online.pth`；这不是对整台服务器所有文件的缺失证明。
已有 `facebookresearch_co-tracker_main` 源码确实支持
`CoTrackerPredictor(v2=False, offline=True, window_len=60)`，无需改写共享上游。

[准备配置](../../configs/dynamic-static-jitter/tracker.cotracker3-preparation-v1.json)
记录权重元数据与检查过的接口源码 SHA。官方模型仓库 revision
`bf55ea50d4390e1820a267f131cd6587240fb2c5` 的 `scaled_offline.pth` 为
101,890,938 bytes（约 102 MB），LFS SHA256
`2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`；
来源为[官方固定版本文件](https://huggingface.co/facebook/cotracker3/blob/bf55ea50d4390e1820a267f131cd6587240fb2c5/scaled_offline.pth)。
这里只读取了元数据，**没有下载或校验本地权重字节**，也没有启动新 GPU 推理。
两个接口文件的哈希不等于全部运行源码身份，实际运行前仍须完整绑定。

仓库 `AGENTS.md` 禁止自动下载权重，因此已向用户请求本任务独立目录内的下载
许可；当前 **许可待回复、权重下载 NOT RUN、CoTracker3 评分 NOT RUN**。
不修改默认 Repair，不打开预留 120 源，不把等待新候选当作完成联合目标。

### 纯本地接入与全起点请求准备：仍无模型实验

下载许可仍未收到；自动 goal 续跑不视为下载授权。本轮只实现本地接口并准备
已确认开发集的查询，不下载权重，不启动 GPU 推理，不新增 Repair 分数。

[CoTracker3OfflineModel](../../packages/audit-models/src/vbench_audit_models/cotracker3.py)
要求调用者显式提供本地权重路径、SHA256、大小和完整 `cotracker` Python 源码
清单。加载前验证这些身份，拒绝混入另一个或未绑定的 `cotracker` 模块；
以 `weights_only=True` 读取状态字典，严格加载到
`v2=False, offline=True, window_len=60` 的本地架构，再检查资产未变。
接口不调用 torch.hub 或下载助手，缺文件即报错，不退回 v2 或随机权重。

查询只复用已有 native `(t,x,y)` 张量约定，不复用 v2 网络或权重；保留全部输入
帧、任意合法查询时刻和预测出画坐标，拒绝非有限轨迹及非布尔可见性。
`backward_tracking` 仍是补齐查询之前的时间段，**不是独立反向端点核验**；
上游在查询帧强制坐标/可见性，因此该位置也不能作为准确性证据。
新接口只接到研究脚本 [probe_feature_tracks](../../scripts/counterfactual/probe_feature_tracks.py)
的显式 `prepare --tracker-kind cotracker3-offline`；旧请求缺该字段时继续使用
CoTracker2，公开评分 CLI 的默认值没有改变。

H200 在 **2026-09-23 01:15:38 UTC** 再次只读核验：官方缓存位置及计划独立
目录仍没有 `scaled_offline.pth`。完整的 **31 个 Python 文件**哈希保存于
[源码清单](../../configs/dynamic-static-jitter/tracker.cotracker3-source-v1.json)，
SHA256 `6884483a8f3639d20ce81b604f5e9f33bf323e069b52a85b4ce20e7ad31bceb0`。
这是 `cotracker` 命名空间的源码身份，不是所有第三方依赖的源码证明；该环境
元数据为 Python 3.10.20、PyTorch 2.6.0+cu124、NumPy 1.26.4。真实权重加载、
CUDA parity 和视频推理均 **NOT RUN**，CPU 替身测试不能替代它们。

随后实际准备了同二十视频的全部 **16 个查询起点**：每起点在当前视频当前帧
独立提取全部 SIFT 空间位置，精确重合的多个方向归为同一位置，原始 key 分组
保留。共 **320 个视频×查询帧记录、416,419 个查询观察**，不是独立样本数或
模型预测数。五原片、五编码控制、十条 8px 干预全部保留；没有人工指定连接处、
配对 clean 的查询迁移、偶数帧选择或视频重新构造。

请求在 [cotracker3-allphase-dev5-v1-requests](../../output/dynamic-static-jitter/cotracker3-allphase-dev5-v1-requests/)，
每个 `frame-00.json` 至 `frame-15.json` 都含完整二十视频。执行代码快照为
`output/dynamic-static-jitter/code-cotracker3-query-prep-v1/`。引用的模型信息明确标为
`planned_local_assets_not_previous_model_inference`，不能当作此前真实模型 provenance。
这些是**结构点查询臂**；固定几何网格臂仍待接入/对照。SIFT 查询密度随视频
变化，不能把所有这些点的条件均值当作固定空间分母的 Dynamic 分数。

[准备核对摘要](../../output/dynamic-static-jitter/cotracker3-allphase-dev5-v1-preparation-audit/summary.json)
SHA256 `2999c68a2f981ab41722350c7224f07cc9df4cf99575967d7bb4735fd3e22916`
记录了十六请求的 SHA、完整 cohort/phase、源码/helper、原生时间轴元数据、
点的唯一性/界内性、原始检测 key 分组完整性，以及 **80 组**原片/编码控制的
查询和解码像素哈希一致性。准备时已检查 MP4 字节 SHA 并实际解码、提取 SIFT；
这次摘要核对不重复解码或独立重检 SIFT，不冒充模型对应关系核验。

**显存边界。** 320 组中 128 组超过旧研究入口默认的 2048 查询保护上限；最大
为 `video_000003 / query frame 7` 的 **3353 点**。没有删点、隐式分批或放宽默认
上限。真实运行需在确认空闲卡后显式给出容量预算（如 `--max-queries 4096`），
再验证显存是否足够；4096 是容量参数，不是已验证的 GPU 内存结论。若不足，
须保留失败并重新设计有记录的调度，不能静默减少联合查询。

本轮新增 19 项模型替身/入口合约测试：本地严格加载、缺失/篡改资产拒绝、命名
空间防混用、原生时空几何、非有限输出拒绝、v2/v3 显式选择与旧请求兼容。
完整 CPU 测试 **1159 passed, 3 skipped**；锁检查、diff 检查、prepare/infer 帮助
通过，冻结目录状态及索引哈希未变。没有新的模型分数、G1 覆盖提升或 Repair
验收结论；下一实测步骤仍需要用户提供该权重或授权下载。

### CoTracker3 实测：双查询臂完成，但仍未修复

用户在上述阻塞后明确回复“允许下载并测试”。本轮只落实该授权：使用原有
**5 个官方完整原片、5 个编码控制、10 条用户认可的 8px 干预**，不改构造、
默认评分或正式留出。执行日期为 2026-09-22 当地时间 / 2026-09-23 UTC。

**资产与接入。** 固定官方 revision、101,890,938-byte checkpoint 及 SHA
与准备配置一致。H200 直连官方站点超时，改为本机从同一固定 URL 下载、核验，
再断点续传到任务独立目录；服务器端 SHA 再次一致。没有修改共享上游源码。
权重为 `/data/chenjiayu/dynamic-structural-motion-20260922/assets/cotracker3-scaled-offline-bf55ea50/scaled_offline.pth`，
不入 Git。[真实加载对照](../../output/dynamic-static-jitter/cotracker3-loading-parity-v1/summary.json)
验证安全加载器与上游加载器的 **188 个状态张量完全相同**；在一个官方原片的
16 帧 / 12×12 网格上，轨迹最大绝对差 **0 px**、可见性完全一致。这是该
输入上的实现 parity，不是物理跟踪准确率或全部场景的模型 benchmark。

**固定输入与数据流。** 每次只输入当前视频的全部 16 个原生帧（8 FPS、2 s），
不读取配对原片、seed、构造场或人工连接处坐标。两条查询臂分别为：

- SIFT：每个当前帧独立提取全部不同空间位置，保留原准备查询；320 组共
  **416,419** 个查询观察，最大 3353 点。显式容量 `--max-queries 4096`，不删点、
  不隐式分批；全部查询在各自一次联合模型调用中处理。
- 固定网格：每帧使用原生图像 **12×12 等面积网格的全部中心**，另跑相同
  320 组，共 **46,080** 个查询观察。位置不随图像内容选择。

两臂各有 20 视频 × 16 查询起点，合计 **640/640 组、0 失败**，不是 640 个
独立视频。全部模型预测（含出画点）保留；上游强制查询帧的坐标/可见性，
`backward_tracking` 不是独立的反向对应认证。统计分别使用全部起点的 1/2/3/4
帧间隔，不只取偶数帧。五组编码控制在每臂 16 起点上均逐数组完全相同，
共 **160/160**；图像/时间、输入/代码/权重/缓存哈希核验全部通过。

**预设的既有分解重放。** 固定网格的相邻位移除以原生时间差，形成完整
15×144 速度场；使用当前视频中间帧的既有 SAM 区域，重放既有
`regional_reversal_ablation`，不调整其门槛。删除部分仍须与每时刻区域仿射投影、
每位置时间加权平均速度正交，退化小区域和未知位置不删除。
这里的观测掩码仅为 **模型可见且坐标在画内**，未获独立物理对应认证。
因此这只是开发失败消融，不能把它的覆盖率等同于旧 G1 的有效评分率。
执行顺序上，列车对应仍未恢复就重放了已有分解，用于检查单换 tracker 是否
足以改变负结果；这不代表已通过“先恢复真实运动”的放行条件，也未设计新分解。

下表均为**条件诊断量，不是最终 Repair score**。单位为短边长度/秒；同一视频
两列使用同一观测掩码，但 base 与 CF 的可见子集仍会不同。单元格顺序为
`base → seed 1701 / seed 2904`；不能把缺失当零或据此声称不变性通过。

| 官方源 | CoTracker3 条件运动量 | 加既有保护分解 | 模型可见且在画内的相邻点比例 |
|---|---|---|---|
| 自行车 | 0.027018 → 0.039084 / 0.038593 | 0.027018 → 0.039084 / 0.038593 | 98.19% → 97.69% / 97.92% |
| 桥上列车 | 0.006824 → 0.097101 / 0.103196 | 0.006824 → 0.061354 / 0.103196 | 99.86% → 94.58% / 91.99% |
| 雪地自行车 | 0.123356 → 0.160279 / 0.165216 | 0.123356 → 0.160279 / 0.165216 | 73.38% → 63.80% / 61.90% |
| 马 | 0.009471 → 0.035344 / 0.033935 | 0.009471 → 0.023603 / 0.025612 | 98.24% → 95.32% / 96.20% |
| 长颈鹿 | 0.036840 → 0.089307 / 0.086608 | 0.036840 → 0.071413 / 0.066060 | 95.93% → 88.19% / 86.06% |

完整未筛选模型预测的统计也保存在摘要 `arms.*.videos[].motion`；固定网格和
SIFT 两臂的相邻运动量均为 **10/10 干预上升**，不是仅在可见子集上的现象。
保护分解只对 **5/10** 条干预发生删除；条件配对 MAE 从 **0.044164** 降到
**0.034739**（下降 21.34%），但仍为 **10/10 干预高于原片**。这些是已见五源
开发诊断，不是已校准的不变性分数、总体准确率或独立测试结论；本轮不新增 CI。

**真实运动负例仍在。** 沿用之前披露的事后显示点 `video_000004 / frame 2 /
SIFT key 272`（不用于挑选模型查询），CoTracker3 的下一帧位移仅
`(+0.040855, +0.020073) px`，可见性仍为真，但原图中的车厢连接处已经移动。
[逐帧显示与诊断](../../output/dynamic-static-jitter/cotracker3-train-anchor-v1/diagnostic.json)
及[显示图](../../output/dynamic-static-jitter/cotracker3-train-anchor-v1/posthoc_anchor.png)
表明这个已知失败未解决；不能把原片低运动量当作静止真值。一个显示点也不
代表所有列车点的真实误差，完整身份准确率仍未标注。

**运行与证据。** H200 物理卡 4–7，各进程只见对应 GPU UUID / 逻辑 cuda:0。
SIFT 四卡墙钟区间 **01:59:22–02:01:34 UTC（132 s）**，网格为
**02:02:27–02:04:06（99 s）**；包含逐相位启动、解码/查询检查与模型运行，
不含下载/传输/审核。单进程最大已分配显存分别约 **5.02 / 1.76 GiB**。
SIFT 的 4 个父进程 PID 为 925750–925753，网格为 933525–933528，均已结束。

- [完整逐起点产物](../../output/dynamic-static-jitter/cotracker3-allphase-dev5-v1/)：
  与 H200 任务根下同名目录一致，全部取回本机，不覆盖旧实验。
- [权威核验摘要](../../output/dynamic-static-jitter/cotracker3-allphase-dev5-v1-audit/summary.json)：
  SHA256 `ce249fe1c4c86ff62d53439bad4864329849b274af2aeadf2ea7d392058664bc`。
  所有查询、来源、形状、时间、有限性、布尔可见性、160 组编码控制及保护约束已核验。
- 执行/核验快照为 `code-cotracker3-comparison-v1` / `code-cotracker3-audit-v1`。
  执行入口 SHA `d3137dc1b28f15d21771a5919baa794c1a2e2a0bf34e6a7a7989f8de6698d0b2`；
  模型适配 SHA `b068be0b48ee68421ae5b6e438da9c38c83bd8337c000f2a0ee82b727b6e4bca`。
  HEAD 仍为 `fdf4890c` 加未提交工作树，不能仅凭 HEAD 复现本轮。
- [本地重放核验](../../output/dynamic-static-jitter/cotracker3-allphase-dev5-v1-audit-local-recheck/summary.json)
  的原始统计/来源/控制与 H200 摘要完全一致；条件标量最大差 0，分解数组在
  `atol=rtol=1e-9` 内一致，五个分解编码控制完全一致。它是验证副本，不是第二套结果表。

完整 CPU 检查 **1167 passed, 3 skipped**；模型加载 parity 为上述已执行范围，
冻结目录和索引身份未变。**本次授权的模型对照完成，但 Repair 仍未成功**。
默认评分、历史 Origin 40%→80% 的结果和旧 G1 有效评分口径不变；自然尺度校准、
自然周期运动/人类偏好非劣验证、正式留出和最终 CLI parity 仍 NOT RUN。

### 几何模型对照：保留旋转/缩放，不用质心代替部件路径

仅联合平移还无法表达轮子旋转、拍翼部件和接近镜头时的尺寸变化。本轮新增
[区域几何拟合](../../metrics/dynamic-degree/src/dynamic_degree/regional_geometry.py)，
比较平移、相似（旋转＋等比缩放＋平移）、仿射三个模型，各含普通最小二乘和
Huber IRLS 两种估计器。稳健残差尺度来自当前拟合，迭代 20 次、乘数 1.345，
不读取构造幅度、相位、seed 或配对；它是开发候选，不是已冻结可靠性阈值。
每个源空间位置总权重为 1，同一位置的多方向 SIFT 不重复投票，也不能分别进入
训练/检查两组造成泄漏。两组按独立源坐标交错划分，报告未参与拟合位置的误差；
它们仍不是独立物理真值。运动按变换后**各点位移**表示，中心不动的旋转或缩放
不被记为零，真实返回的两段路径也不相消。欠秩/无对应保持缺失，不选一个满秩
模型就自动声称可靠。

[重放入口](../../scripts/counterfactual/probe_regional_geometry.py)读取已核验的
native v1 全部稀疏对应，在独立快照 `code-regional-geometry-v1/` 上完成
**20/20 视频、1080 帧对、37,653 区域帧对、0 运行失败**；五组编码对照逐项
完全相同。输出为 [regional-geometry-dev5-8px-v1](../../output/dynamic-static-jitter/regional-geometry-dev5-8px-v1/provenance.json)，
JSONL SHA256 `62df1acdf77f4ce8566e42d7b7b6378ba731c0d6671f8c515001f75f2a68efac`，
20 条分数仍为 null。monotonic 耗时 211.71 秒；UTC 日志 19:34:20–19:37:37，
同样保留与 monotonic 的差异。

随后修正了一个诊断字段问题：IRLS 达到迭代上限时，`fit_weights` 曾记录准备
用于下一轮的权重，而非产生当前矩阵的权重。算法数值未改，新增回归测试后在
独立快照完整重放同二十条，保留原输出；修正版位于
[regional-geometry-dev5-8px-v1-weights-fix](../../output/dynamic-static-jitter/regional-geometry-dev5-8px-v1-weights-fix/provenance.json)，
20/20、1080 帧对、0 运行失败，SHA256 为
`c906dc75e95dacfaa87f50425b40e1380f22b215aec84084a6bc3833f2813a34`。
逐项核对全部二十条，除 35,196 个拟合记录的权重字段外，矩阵、误差、状态和
所有其他记录完全相同；下表数值不变。修正的是 provenance，不是 Repair 效果。

| 模型 | 有满秩拟合的区域帧对 | 两组均能进行留位检查 | 普通/稳健：留位平均误差的中位数（px） |
|---|---:|---:|---:|
| 平移 | 18,402 / 37,653 | 12,101 / 37,653 | 1.1997 / 1.1580 |
| 相似 | 12,101 / 37,653 | 7,745 / 37,653 | 1.4747 / 1.4053 |
| 仿射 | 9,092 / 37,653 | 6,234 / 37,653 | 1.7446 / 1.5962 |

分母含原片/编码控制/干预、全部时间间隔、嵌套区域和全画面对照，**不是独立
视频样本数，也不是有效评分覆盖率**。每行误差仅在该模型可检查的条件子集上
统计，不能横向据此宣称平移比仿射更好。普通/稳健的两组可计算数相同。
19,251 个区域帧对没有 SIFT 对应；稳健平移/相似/仿射的留位误差 95 分位仍为
26.56/29.16/31.45 px，远不能把“稳健拟合”当成运动正确性证明。

列车原片首帧 region 0 的稳健平移为 (−34.680, −9.881) px，但两组检查误差
仍大。第一条 8px 的对应列车 region 6 虽有两个 SIFT key，实际是**同一空间
位置的两个方向**：平移可拟合，独立空间核验不足，相似/仿射欠秩。此发现要求
后续先补独立的区域/跨帧对应证据，不继续调稳健拟合参数来制造有效覆盖。
几何分支尚未与原生 RGB 多假设、跨帧身份/遮挡共同融合，也未产生最终分数。

以上新数学测试覆盖亚像素镜头位移、旋转/接近、返回路径保留、仿射剪切、欠秩、
错误对应、重复特征泄漏和失败保留，不替代自然视频验证。完整 CPU 检查为
**1021 passed, 3 skipped**；锁和 diff 检查通过，冻结研究目录 Git blob identity 未变。

## 当前进展：模型/空间分辨率对照与稠密轨迹分解（仍未通过）

本轮仍是相同 **5 个官方原片＋5 个编码控制＋10 条已确认 8px**，不增加或替换
样本、不改构造，也不读取留出结果。两次模型对照分别尝试二十条，不能记为四十
条不同视频。Origin 使用原锁定上游和 `raft-things.pth`，二十条结果在两次运行中
均逐条复现，仍为干预前后动态比例 **40%→80%**，并非准确率。

### G1 第三/四对照：现有 torchvision 权重，内部 1× / 2× 推理

前两候选不足以覆盖列车干预，因此接入现成的 torchvision RAFT-large
`C_T_SKHT_V2`。其预训练/微调来源见 [torchvision 0.21 官方说明](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.optical_flow.raft_large.html)；
这只是对应关系模型对照，不能由光流基准性能推断本实验的抖动鲁棒性。
权重本地已存在，本轮未下载、训练或修改；SHA256 为
`ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`。

适配：[torchvision_flow.py](../../packages/audit-models/src/vbench_audit_models/torchvision_flow.py)。
显式本地权重、严格载入，RGB 归一化到 `[-1,1]`，20 次更新。1× 保持旧候选
256 边长的模型输入；2× 仅在模型内部双线性放大，预测光流返回原评分像素单位。
不生成/替换输入视频，不改时间采样、运动量分母、核验阈值或 60% 覆盖门槛。
Origin 和 Repair 的权重参数独立，绝不把新权重注入 Origin。

两者均为 **Origin 20/20、对应关系消融 16/20 有效、4 条不足、0 运行异常**；
CF 有效配对均为 **8/10**，从旧 RAFT 对照的 6/10 改善，但雪地原片及两个
干预仍不足。这里“有效”仅指通过现有证据门槛，不代表最终运动修复成立。
下表仍使用旧观测弧长累计，**不是完整 Repair**；单位为短边长度/秒。

| 官方源 | 1×：base → seed 1701 / 2904 | 2×：base → seed 1701 / 2904 | 2× 内部证据覆盖：base / 1701 / 2904 |
|---|---|---|---|
| 自行车减速 / LaVie | 0.013279 → 0.009965 / 0.009780 | 0.011423 → 0.010085 / 0.008371 | 87.73% / 80.74% / 76.25% |
| 桥上列车 / ModelScope | 0 → 0.039495 / 0.044354 | 0 → 0.085394 / 0.090829 | 96.94% / 81.99% / 83.38% |
| 雪地自行车 / VideoCrafter | null → null / null | null → null / null | 49.31% / 36.85% / 34.81% |
| 马 / LaVie | 0.003604 → 0.008803 / 0.006109 | 0.003796 → 0.041754 / 0.040325 | 94.72% / 90.51% / 88.24% |
| 长颈鹿 / ModelScope | 0.017709 → 0.023444 / 0.042256 | 0.017436 → 0.080184 / 0.082217 | 92.59% / 84.12% / 81.57% |

对应更准确之后，弧长反而更明显地追随构造抖动，这正说明“跟踪可靠”与“真实
运动量”必须分开处理。列车干预的覆盖由 1× 的 61.76%/61.06% 增至 2× 的
81.99%/83.38%。雪地 2× 干预的局部循环通过率仍只有 46.16%/35.69%，匹配
patch 通过率 53.38%/51.99%；并非取消其中一个核验就已经解决。

执行证据（同 H200 物理 4，UUID `GPU-490b4a76-6210-31b9-4e03-838a113cf5f4`）：

- 1×：PID 348861；**2026-09-22 16:00:26–16:01:12 UTC**，约 46 秒墙钟，
  视频循环 42.92 秒；隔离代码 `/data/chenjiayu/dynamic-structural-motion-20260922/code-dense-tv-v1/`。
  本地[评分/运行与模型身份](../../output/dynamic-static-jitter/dense-tv-dev5-8px-v1-scores/)、
  [逐条表和完整分析](../../output/dynamic-static-jitter/dense-tv-dev5-8px-v1-analysis/)。
  scores SHA256 `829b421d6953f3dfa95fd0230e9c2e6387095d743334d5bd8ffd0d1e6d393a64`。
- 2×：PID 352928；**16:05:27–16:06:19 UTC**，约 52 秒墙钟，视频循环 48.58 秒；
  隔离代码同根 `code-dense-tv-x2-v1/`。
  本地[评分/运行与模型身份](../../output/dynamic-static-jitter/dense-tv-x2-dev5-8px-v1-scores/)、
  [逐条表和完整分析](../../output/dynamic-static-jitter/dense-tv-x2-dev5-8px-v1-analysis/)。
  scores SHA256 `d6ed60e29838ef8fb04bbdbb83f540e58d0b29d95651d855b7192d84e883ac5b`。
- 两次均逐项核对各自隔离快照的 57 项源码哈希；旧 1× 适配源码与后续新增
  scale 参数的源码不同，不能仅用当前工作树或 HEAD 代替其执行版本。
  Python 3.10.20 / torch 2.6.0+cu124 / torchvision 0.21.0+cu124；同配置
  `trajectory.dense-dev-v1.json`。五组原片/控制的分数、状态和覆盖完全一致，
  两次首条 Origin 原生 infer parity 均通过。
- 各二十份 NPZ 保留在远端上述代码目录的 `output/<run-name>/evidence/`，已
  全部读取，验证数值有限和 16 个原生时间样本。**本地同步的是文本结果，不
  声称这四十份稠密场已经复制到本地**；2× 缓存逐条 SHA 另见下述 probe provenance。

### G2：将两种分解应用于新 2× 稠密对应（开发诊断，不是有效评分）

新增 [dense_paths.py](../../metrics/dynamic-degree/src/dynamic_degree/dense_paths.py)：
从每个固定短窗的网格点出发，双线性采样已存前向场并逐帧推进；在实际轨迹
位置重新做逐像素形变 patch 和局部正反循环核验。可见性仍只是几何代理，不
冒充 learned visibility。重叠时间仍按原固定所有权统计一次，不按结果选窗。
没有重新运行模型，没有输入建库位移场、seed 或配对原片。

固定位置的原缓存光流弧长先做精确重放核验；然后才构建新的随动轨迹。
后者不等于固定网格光流或 CoTracker 轨迹，因此不能把不同来源的 MAE 大小
直接当成同一方法提升/退化。以下各行则使用**同一批新随动轨迹**，可作配对
开发消融。所有二十条记录继续标注 `diagnostic_only / score=null`，十对全部
报告，来源只有 8/10 有效配对；未知轨迹并未被认证为真实运动。

| 消融 | base → CF（十对全轨迹诊断均值） | 平均绝对变化 [prompt-cluster 95% CI] |
|---|---|---|
| 原始随动轨迹 | 0.049059 → 0.113656 | 0.064597 [0.032005, 0.097828] |
| 仅时间趋势，失败对照 | 0.034638 → 0.042480 | 0.009061 [0.006639, 0.011983] |
| 仅局部结构支持 | 0.045168 → 0.071334 | 0.026165 [0.010437, 0.044061] |
| 时间＋局部结构支持 | 0.048957 → 0.079653 | 0.030696 [0.015497, 0.046172] |
| 共同模态＋空间保护，不要求反向 | 0.049088 → 0.079723 | 0.030635 [0.015405, 0.047691] |
| 共同模态＋空间保护＋重复反向 | 0.049059 → 0.079723 | 0.030664 [0.015457, 0.047699] |

主要发现：列车强往返模态的证据能量覆盖已达到约 **80.6–81.9%**，不像此前
CoTracker 的 31–42% 那样无法核验；共同模态候选确实抑制了非仿射部分。
但列车仍为 `0.001087 → 0.041136/0.040722`，长颈鹿仍为
`0.038551 → 0.069238/0.065090`。保留的仿射投影、其他残差以及轨迹累积误差
仍贡献变化；不能为了这几个数字取消真实镜头/周期运动保护。
列车“仅时间趋势”本身也为 `0.000703 → 0.015071/0.015311`，并不等于恢复
原片运动。雪地随动路径的内部证据覆盖只有 41.53%→29.07%/21.81%，仍未解决。

- CPU-only，同一隔离代码 `.../dynamic-structural-motion-20260922/code-dense-path-probe-v1/`。
  两次各完成二十条；局部支持 **16:10:38–16:10:53 UTC，15.43 秒**，共同模态
  **16:10:38–16:10:47 UTC，9.38 秒**。并行 CPU 重放，不是两次 GPU 实验。
- 本地[局部支持全部诊断/统计/身份](../../output/dynamic-static-jitter/dense-path-local-dev5-8px-v1-probe/)、
  [共同模态全部诊断/统计/身份](../../output/dynamic-static-jitter/dense-path-common-dev5-8px-v1-probe/)。
  两份 provenance 的 58 项源码与 probe 脚本均和执行时本地源码一致；输入与
  二十份源缓存 SHA 均保存。五组原片/编码控制四个量逐条完全一致。
  诊断 JSONL SHA256 分别为 `de0d59b71edb20f25fbcee3009ecd82aedb51f043ee61ef359947f8efee6c2ee`
  和 `6da31a18058296a6f5cff7321c68a3b43879ac5023fc07f8311a908d89d11e7b`。

当前结论：**主要运动成分的对应支持有所恢复，现有完整分解仍未通过**。
下一步需要改善结构层运动证据与多帧对应，而不是继续按抖动幅度调低置信门槛。
已有 DINO 本地资产可作为后续结构特征对照的候选，但本轮尚未接入或测试；
不能宣称它已经解决此问题。真实周期运动、自然偏好、尺度校准与独立测试仍未完成。
本轮完整 CPU 测试 **900 passed, 3 skipped**；新增测试覆盖独立权重参数约束、
无下载加载、输入单位/插值还原、随动位置匹配、越界与错误反向证据拒绝。
重新完成 uv locked sync、CPU torch overlay、锁检查、八公共/两研究入口 help、
源码编译与 diff 检查；冻结目录 blob 摘要仍为
`0648e4ebf351ebc7268f01830615f8e7d964d6eb8c627f84ed4e1ef9084115a2`。
公共默认和其余任务的已有改动未覆盖；未提交/推送未完成候选，goal 保持 active。

## 前两轮：G1 稠密对应关系与 CoTracker 上的 G2 分解（均未通过）

### G1 第二个候选：逐帧重定位与可变形 patch 核验

短窗 CoTracker2 对雪地自行车原片仍只有 39.12% 证据覆盖，因此追加一个局部
对应关系对照，复用已有本地 RAFT 权重，不下载模型、不改变视频：

`单视频 RGB/时间戳 → 相邻帧正反 RAFT → 逐像素形变补偿的 patch 匹配与局部循环 → 运动弧长诊断/覆盖率`。

这是[RAFT 逐像素光流模型](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470392.pdf)
在本地对应关系估计中的应用，不是该论文已经解决真实运动/抖动区分的证据。
新候选没有 CoTracker 的 learned visibility head；几何入界代理、正反循环和匹配
检查分别记录，不能把其代理可见性与旧模型可见性当作同一统计量。

另发现旧静止回退的反例：运动匹配更优，但循环失败时，旧规则仍可能因静止误差
低于阈值而记为零。新候选要求静止解释不存在更优运动匹配；矛盾证据不伪装为
静止。覆盖率门槛仍为 60%，不靠调低门槛通过。旧方法与旧结果不改写。

源码：[dense_correspondence.py](../../metrics/dynamic-degree/src/dynamic_degree/dense_correspondence.py)；
配置：[trajectory.dense-dev-v1.json](../../configs/dynamic-static-jitter/trajectory.dense-dev-v1.json)。
当前仅接研究入口 `score_static_jitter --repair-variant dense-correspondence`，
尚非公共 CLI 的默认或最终 Repair。与旧候选相比，同时改变了对应关系估计器、
形变匹配与冲突处理，且 RAFT 输入不额外做 1.2px 模糊，不能将差异归因于单一因素。

同二十条全部尝试：Origin **20/20 有效且与旧运行逐条一致**；候选 **14/20 有效、
6/20 证据不足、0 运行异常**。十条 CF 有效配对为 **6/10**，仍不满足五源回归门槛。

| 官方源 | 候选：base → 8px seed 1701 | 候选：base → 8px seed 2904 | 证据覆盖：base / 1701 / 2904 |
|---|---|---|---|
| 自行车减速 / LaVie | 0.012275 → 0.009518 | 0.012275 → 0.008255 | 88.29% / 81.85% / 76.20% |
| 桥上列车 / ModelScope | 0 → null | 0 → null | 96.94% / 53.89% / 48.84% |
| 雪地自行车 / VideoCrafter | null → null | null → null | 50.79% / 30.23% / 28.94% |
| 马 / LaVie | 0.003451 → 0.004536 | 0.003451 → 0.004413 | 94.49% / 66.16% / 68.84% |
| 长颈鹿 / ModelScope | 0.018116 → 0.016446 | 0.018116 → 0.011364 | 91.76% / 74.03% / 66.67% |

雪地原片的循环通过率由短窗的 33.2% 增至 65.0%，但 photometric/入界等联合
证据仍不足；seed 1701 CF 的循环通过率只有 40.0%。列车 seed 1701 的 moving
patch 通过率约 64.1%。因此“改成逐像素形变核验”也不足以使全部视频可靠评分。

- H200 物理 **7**，UUID `GPU-5e07f671-a7e1-b38d-4d14-30316dfd9ce8`；PID 326784，
  **2026-09-22 15:05:56–15:06:51 UTC**，单卡约 55 秒，逐条循环 50.76 秒。
  首次物理 4 的启动前检查未放行，确认无进程/输出后才在 7 上启动，没有重复推理。
- 隔离代码 `/data/chenjiayu/dynamic-structural-motion-20260922/code-dense-v1/`，运行
  `output/dense-correspondence-dev5-8px-v1-scores/`，20 份正反稠密场保存在其 `evidence/`。
  Python 3.10.20 / torch 2.6.0+cu124，同一 RAFT 权重和锁定上游；provenance 明确
  tracker 权重/code 为 null，避免假称仍用了 CoTracker。54 项执行代码哈希已核对。
- 本地[评分/provenance](../../output/dynamic-static-jitter/dense-correspondence-dev5-8px-v1-scores/)、
  [完整逐条表](../../output/dynamic-static-jitter/dense-correspondence-dev5-8px-v1-analysis/cases.csv)、
  [统计与区间](../../output/dynamic-static-jitter/dense-correspondence-dev5-8px-v1-analysis/summary.json)。
  scores SHA256：`23f57e76df1194eeff2eaea876cd43f2b6091b0c70bcd33a6fe174195ba79006`。
  五组原片/编码控制分数、状态与覆盖完全一致；Origin 首条原生 infer parity 通过。
  二十份模型缓存现已全部同步到本地，逐文件 SHA256 与远端一致；本地也逐一
  读取检查全部数值有限、每条 16 个原生时间样本。没有追加或重复模型推理。

### G2 首轮：时间趋势与区域支持残差

实现 [structural_decomposition.py](../../metrics/dynamic-degree/src/dynamic_degree/structural_decomposition.py)：
按真实时间的二阶导惩罚分解轨迹 `q=b+r`，全部帧参与；只有观察到重复反向且
留一点法局部仿射预测不支持残差时才衰减 `Δr`。邻域由空间距离及视频 RGB 外观
加权，不是语义分割。邻居不足/跟踪不可靠时保留残差，不自动认定为干扰。
输出 raw、仅时间、仅结构、完整组合四个消融，并记录 1/2/3/4 帧间隔诊断。

配置 [decomposition.dev-v1.json](../../configs/dynamic-static-jitter/decomposition.dev-v1.json)
在本次 probe 前写入。复用的是**前一轮 CoTracker2 短窗缓存**，不是上节 RAFT 场，
不混充同一方法或新的模型推理。[probe 脚本](../../scripts/counterfactual/probe_structural_decomposition.py)
重新读取同一原始视频获取外观特征，校验媒体/缓存、窗口所有权及原始弧长重放 parity。

**下表是十对全轨迹开发诊断，不是已验证的 Repair 分数。** 其中含不可靠轨迹，
源运行只支持 5/10 有效配对；新记录均为 `status=diagnostic_only, score=null`，
不能把“十对诊断可计算”冒充十对有效评分。单位为短边长度/秒。

| 消融 | base → CF（全轨迹诊断均值） | 平均绝对变化 [prompt-cluster 95% CI] |
|---|---|---|
| 原始轨迹 | 0.039616 → 0.056690 | 0.017074 [0.007451, 0.034036] |
| 仅时间平滑 | 0.026664 → 0.026694 | 0.001586 [0.000672, 0.002688] |
| 仅区域结构 | 0.036881 → 0.051966 | 0.015085 [0.007429, 0.027228] |
| 时间＋区域支持残差 | 0.039370 → 0.053059 | 0.013689 [0.006149, 0.026538] |

组合仅减轻部分误增：列车仍为 `0.002645 → 0.043956/0.038900`。其“未知时保留”
策略保护了缺乏区域支持的小运动，也留下大量疑似抖动。时间平滑的低 MAE 不能
直接采纳：数学周期运动对照会被它衰减，雪地原片的轨迹量也从 0.123512 降到
0.070816；这里不能据此断言损失全是真实运动，但更不能宣称已证明真实响应保留。

组合在数学对照中保留了 2/3/4/8 帧周期的协调平移、周期仿射运动及外观孤立的
小区域，并压低了不协调往返残差。这仅验证算法机制；**自然周期运动、人类偏好、
自然尺度校准和独立不变性验证仍未完成**，不是拿合成测试替代官方视频实验。

- H200 CPU-only，**2026-09-22 15:17:41–15:17:51 UTC**，10.12 秒，20/20 诊断完成，
  零次新模型推理；源运行的不足状态完整保留，五组原片/控制四项诊断完全一致。
- 隔离源码 `/data/chenjiayu/dynamic-structural-motion-20260922/code-decompose-v1/`；
  本地[逐条诊断](../../output/dynamic-static-jitter/decomposition-dev5-8px-v1-probe/diagnostics.jsonl)、
  [汇总](../../output/dynamic-static-jitter/decomposition-dev5-8px-v1-probe/summary.json)、
  [输入、模型缓存与代码证据](../../output/dynamic-static-jitter/decomposition-dev5-8px-v1-probe/provenance.json)。

首轮之后需同时处理跨区域共同扰动与局部真实运动证据，不能将局部未知简单认定为
真实残差或静止。任何新候选仍须保留真实周期/小区域运动、覆盖所有十条 8px，
然后再进入自然开发与冻结；本轮没有扩大样本、减弱构造、读取留出分数或晋升默认。

补充只读诊断：对三个旧短窗分别取上述 0.2 秒趋势的残差增量，展开为
`(T−1) × (2N)` 矩阵做 SVD。列车原片/seed 1701/seed 2904 的首模态能量占比
（三窗均值）约为 **44.0% / 97.9% / 96.5%**；将其二维空间载荷回归到
`[1,x,y]` 仿射场，未中心化的解释能量约为 **66.8% / 11.7% / 11.7%**。
这提示可研究共同时间成分与空间结构的联合判据；后续实际实现的负结果见下节。
该模式本身更不证明其他真实周期/非刚性运动不会误判。

首轮验证：**851 passed, 3 skipped**；uv lock/sync、CPU torch overlay、八公共
CLI 与研究入口 help、diff 检查通过，冻结目录 blob 摘要仍为
`0648e4ebf351ebc7268f01830615f8e7d964d6eb8c627f84ed4e1ef9084115a2`。
数学测试包括协调周期平移/仿射运动、小区域保护、重叠时间不重复计数、帧相位/
空间单位检查，以及“运动证据失败不能退回假静止”的反例；这些测试不等同于
真实自然运动、人类偏好或独立测试已通过。Goal 保持 active，未提交/推送未完成方法。

### G2 第二候选：共同时间成分＋空间保护（负结果，不晋升）

实现 [common_mode.py](../../metrics/dynamic-degree/src/dynamic_degree/common_mode.py)，
配置 [decomposition.common-mode-dev-v1.json](../../configs/dynamic-static-jitter/decomposition.common-mode-dev-v1.json)。
数据流为：单视频轨迹 → 物理时间趋势与残差 → 对残差速度做时间积分加权 SVD
→ 检验模态及原运动是否重复反向 → 检查局部/全局结构及证据支持 → 仅抑制
满足全部条件的非仿射分量。全部帧参与，不固定频率或相位，不读取 8px/seed/配对。

全局仿射、同向视差、局部协调的两个区域、小范围孤立运动分别有保护条件；
近退化奇异值对应的任意基底不用于抑制。即便一个模态被判为候选干扰，也保留
其仿射投影，防止把与干扰同频的真实镜头运动一起删除。无足够证据则不执行
模态删除。这些是待检验的算法先验，**不是 RGB 能保证识别物理成因的证明**。

同二十条 CoTracker 缓存全部重放，仍为 **diagnostic_only / score=null**，并没有
得到二十条有效 Repair 分数。原来源只支持 5/10 有效配对的事实不变。十对结果：

| 新候选消融 | base → CF（全轨迹诊断均值） | 平均绝对变化 [prompt-cluster 95% CI] |
|---|---|---|
| 共同模态＋空间保护，不要求重复反向 | 0.039549 → 0.055952 | 0.016404 [0.006885, 0.034044] |
| 共同模态＋空间保护＋重复反向 | 0.039574 → 0.055952 | 0.016379 [0.006896, 0.033970] |

原始轨迹与仅时间平滑诊断逐条复现上一节，其数值不重建为另一份主表。这里的
`structure_only` 是在共同模态方案内去掉重复反向条件，不等于首轮局部支持消融。
完整候选 MAE **高于**首轮局部支持的 0.013689，没有改进前一候选。

| 官方源 | 完整候选全轨迹诊断：base → seed 1701 / seed 2904 |
|---|---|
| 自行车减速 / LaVie | 0.020974 → 0.024210 / 0.029308 |
| 桥上列车 / ModelScope | 0.002993 → 0.057514 / 0.050829 |
| 雪地自行车 / VideoCrafter | 0.123512 → 0.127958 / 0.137008 |
| 马 / LaVie | 0.007139 → 0.013957 / 0.015927 |
| 长颈鹿 / ModelScope | 0.043250 → 0.050508 / 0.052305 |

关键失败原因：列车干预的第一模态虽占残差能量约 96–98%，但按该模态空间载荷
能量加权，满足正反可见性/循环的证据只有 **31–42%**（六个窗口），不足以安全
删除该成分。其全点数量通过率约 65%，却不能代表主要运动成分被可靠覆盖。
直接取消置信检查可能改善开发数字，但没有解决运动证据问题，本轮未这样做。

另做只读诊断：把两轨迹的坐标差检查换成相邻增量差检查，同取 2 个评分像素，
仍保留原正反可见性约束；列车两种子第一模态的能量加权通过率（三窗均值）
分别约为 35.9% / 37.4%，对比原 37.1% / 37.2% 无实质恢复。因此并非简单的
常数坐标偏移就能解释失败。该诊断未更改任何评分规则或冻结门槛。

- H200 **CPU-only**，2026-09-22 **15:47:53–15:47:59 UTC**，逐条循环 5.56 秒；
  20/20 诊断完成，无新增 GPU 推理。源媒体仍为同五条官方 MP4、五个编码控制、
  十条已确认干预；未构造新视频、未读取留出结果。
- 隔离执行源码 `/data/chenjiayu/dynamic-structural-motion-20260922/code-commonmode-v1/`；
  产物 `output/common-mode-dev5-8px-v1-probe/`。
- 本地[逐条诊断/逐窗口保留原因](../../output/dynamic-static-jitter/common-mode-dev5-8px-v1-probe/diagnostics.jsonl)、
  [统计与区间](../../output/dynamic-static-jitter/common-mode-dev5-8px-v1-probe/summary.json)、
  [代码/配置/媒体/源缓存身份](../../output/dynamic-static-jitter/common-mode-dev5-8px-v1-probe/provenance.json)。
  56 项执行源码哈希、probe 脚本及配置与本地一致；五组原片/编码控制四项诊断
  完全一致。诊断 JSONL SHA256：`d708d2f14067f41c9240e25fb78d2b66a6736c2279c4d173d52e6151a4876caf`。

追加后完整 CPU 测试 **879 passed, 3 skipped**；锁文件、八公共/研究入口 help、diff 与
冻结目录检查通过。新增数学控制包括 2/3/4/8 帧周期的平移、转动、视差、两个
相反运动区域及小区域，以及同频镜头/干扰混合、坐标旋转/缩放、时间反转和
等能量退化模态的保守保留。
这不代替自然周期运动与人类偏好验证。下一步须先获得能支持主要运动成分的
可靠对应或区域估计；不能靠去掉置信检查、改小 8px 或把全轨迹诊断改名为有效
分数来进入独立验证。公共默认未变，G1–G5 的联合目标仍未完成。

## 上一轮：G1 多起点短窗可靠性消融（未通过）

新 goal 已处于 active。首个受控实现只改变跟踪时间组织，不改变 8px 构造、模型
权重、图像预处理、对应关系阈值或旧运动累计公式：每 0.5 秒重启一个 1 秒局部
CoTracker2 窗，各窗独立正反跟踪。16 帧输入产生 `[0,8)`、`[4,12)`、`[8,16)`
三个窗；每个相邻帧对固定归属上下文最完整的窗，并列取较早窗，不按分数或
置信度择优，不重复累计重叠时间。所有帧/相位仍参与。

入口为 `--audit-variant local-trajectory`，不是默认 Repair。源码见
[local_trajectory.py](../../metrics/dynamic-degree/src/dynamic_degree/local_trajectory.py)，
配置见[trajectory.local-dev-v1.json](../../configs/dynamic-static-jitter/trajectory.local-dev-v1.json)。
这是**旧运动公式的可靠性消融**：尚无高频往返/真实运动分解，旧公式忽略无效点的
局部运动累计也不是无偏的全画面真实运动估计，不能直接晋升最终 Repair。

同一用户认可的五源、两种子，5 原片 + 5 编码控制 + 10 干预，**20 条全部完成尝试**：
Origin 20/20 有效，局部候选 13/20 有效、7/20 证据不足、0 运行异常；CF 有效配对
从旧 dev-v4 的 **4/10 增至 5/10**，五个原片仍仅 4 个有效，G1 放行条件未达到。

| 官方源 | 局部候选：base → 8px seed 1701 | 局部候选：base → 8px seed 2904 | 证据覆盖：base / 1701 / 2904 |
|---|---|---|---|
| 自行车减速 / LaVie | 0.006604 → 0.005329 | 0.006604 → 0.003645 | 82.45% / 76.85% / 71.20% |
| 桥上列车 / ModelScope | 0 → null | 0 → null | 96.85% / 50.28% / 44.95% |
| 雪地自行车 / VideoCrafter | null → null | null → null | 39.12% / 25.69% / 23.98% |
| 马 / LaVie | 0.002670 → 0.003737 | 0.002670 → 0.004480 | 91.57% / 62.27% / 67.31% |
| 长颈鹿 / ModelScope | 0.010388 → 0.006044 | 0.010388 → null | 76.44% / 61.76% / 59.91% |

Origin 二十条均与先前 8px 运行完全一致；CF 动态比例仍为 40%→80%，四次 0→1、
零次 1→0。候选有效五对的均值 0.005787→0.004647、MAE 0.002291，prompt-cluster
95% CI [0.001438, 0.004344]，只含三个源且一源仅一个种子；**不能将其代表全体，
也不能与旧四对的 MAE 直接当作同总体比较**。所有 null 留在十条分母中。

局部门槛诊断（均由所存轨迹缓存计算）：雪地自行车原片循环一致性通过率从
约 10.9% 提高到 33.2%，可见性从 38.5% 提高到 55.3%，但整体证据覆盖仍只有
39.12%。列车 seed 1701 的 moving-patch 通过率仅 64.0%，静止 patch 为 42.0%，
循环通过率 77.5%；不是单纯把窗口缩短就能解决。后续需继续检验局部对应关系、
多尺度/可变形结构证据，并将可靠性与真实运动分解分开做消融，不能靠降低 60%
覆盖阈值通过。

执行与证据：

- Scorer 新增 `--review`：校验复核清单、原 manifest/config/source 哈希、十条身份
  与无折叠/原生时序等技术检查，保留旧自动状态；不将 review、seed、配对传给
  metric。分析新增 `reviewed_tables`，旧严格 `tables` 仍保留，不回写旧实验。
- H200 物理 **6**，UUID `GPU-8cf0627c-5919-bace-1c42-e5d627c36cb5`，逻辑 cuda:0；
  PID 309903，**2026-09-22 14:45:15–14:45:50 UTC**，单卡约 35 秒，逐条循环
  27.45 秒。Python 3.10.20 / torch 2.6.0+cu124；未下载模型、未终止其他任务。
  退出 1 来自七条证据不足；`runtime.json` 为 finished/completed=20，不是漏跑。
- 隔离源码保留于 H200 `/data/chenjiayu/dynamic-structural-motion-20260922/code-g1-v1/`；
  运行目录为其中 `output/local-trajectory-dev5-8px-v1-scores/`。新源文件、未提交
  工作树与旧源码不可仅用 HEAD 区分，实际执行的 **53 项代码哈希**已与本地核对。
- 本地[评分与 provenance](../../output/dynamic-static-jitter/local-trajectory-dev5-8px-v1-scores/)、
  [完整逐条表](../../output/dynamic-static-jitter/local-trajectory-dev5-8px-v1-analysis/cases.csv)、
  [统计与区间](../../output/dynamic-static-jitter/local-trajectory-dev5-8px-v1-analysis/summary.json)。
  原有构造的 manifest SHA256 仍为 `57d462c3fed8dcb4a116ac6a059877e0b40dbd67668642ac8bdd53797f603928`；
  本轮 scores SHA256 为 `67d72b646fbaccfb2eb634320b63f621eaaf44806e6effca051abbd0c8d05eae`。
- 二十条无重复/缺失，20 份 NPZ 均已同步；五组原片/零编辑控制的分数、覆盖率与
  状态完全一致。旧 Origin 的 20/20 数值重现；首条同时通过官方原生 infer parity。
- CPU 全仓 **832 passed, 3 skipped**；uv lock/sync、CPU torch overlay、八 CLI 和
  两个研究入口 help、diff 检查通过；冻结目录 blob 摘要未变化。这里的真实模型
  验证仅覆盖本轮二十条及研究入口，最终方法单/多卡 CLI parity 仍待后续完成。

## 最新决定：用户视觉确认 8 px，固定构造，改进 Repair

在看过已展示的五条对比视频后，用户明确确认“8px我视觉检查过了，非常合适”，
并指定后续反事实沿用此方案。本批五源 × 两种子的十条干预均纳入新开发目标，
不减弱幅度，不再将旧 `min_warp_jacobian=0.5` 自动筛查当作语义有效性的否决。
身份、哈希、用户原话与确认范围见
[人工复核清单](../../configs/dynamic-static-jitter/review.local-texture-dev5-8px-v1.json)。

该确认是**结果展示后的开发集审核**，不是评分前盲审，也不自动涵盖未展示的新
视频。它不改变既有分数：同五源 Origin 为 **40%→80%**，两种子共四次 0→1；
Repair 仍仅 **4/10 有效配对**，不足以证明修复成功。全体 Δ=+0.40 的
prompt-cluster 95% CI 为 [0.00, 0.80]，仅五组，不外推为正式总体结论。

下节保留的是**人工确认前**的自动资格与压力测试记录：4 qualified / 6 rejected
和严格子集 50%→50% 均是历史事实；不能把后者继续当作本批用户认可样本的主分母。
其中“没有人工审核”“仅压力观察”等措辞仅描述当时状态，已由本节后续审核更新。
原始 config、candidates、scores、分析 JSON 与预览标签不回写、不伪装成事前通过。
人工确认当时 scorer/分析脚本尚不读取新复核清单；上述 G1 已新增 `--review`。
历史 `--stress-test` 仍是获取全部十条旧结果的选择入口，不是当前人工语义判定。

下一阶段见[新 Repair 目标与数据流](../plans/2026-09-22-dynamic-structural-motion-repair-goal.md)：
同时解决轨迹证据覆盖与往返抖动计入运动的问题；禁止以全零、弃权或压低真实运动
通过验收。最初确认轮只改文档和复核记录；后续 G1 的模型运行单列于上节。

## 历史运行记录：8 px 追加评分（人工确认前的自动口径）

用户要求“大幅度地 8 px 再看看”。本次只加大同一位移场的幅度：源视频、两个种子、
空间尺度、时间相位、原生 FPS、Origin、CoTracker2 Repair 参数和权重均不变。
10 份实际 8 px 位移场已逐一验证为上轮对应 4 px 场的**精确两倍**。
这里 8 px 是每帧最大位移，正负交替时相邻帧最大跳动为 16 px，不是加像素噪声。

**没有下调质量门槛**：256×256 三个源的最小 warp Jacobian 为 0.448519 / 0.442275，
低于原定 0.5，因此六个干预仍为 rejected；512×512 两个源为 0.543930 / 0.535653，
四个干预 qualified。全部无折叠、无越界，最低低频相关 0.925308，像素与时间轴核验
通过。中间帧复核可见桥面/桥柱局部弯曲，强度已不只是细小纹理变化；没有人工语义
审核结论。为响应用户观察强扰动的要求，显式 `--stress-test` 另评分这些 rejected
样本，**不更改台账资格，也不将其计作合法不变性验收样本**。

### 批量结果：严格区分全体压力测试与合格子集

同五源、两种子；Origin 各种子得分一致。下表均值是动态判定比例，不是准确率。

| 最大位移 / 统计范围 | Origin：base → CF | 0→1 / 1→0 | 构造合格 / 干预数 | Repair 有效配对 / 干预数 |
|---|---|---|---|---|
| 1 px，全体 | 40% → 40% | 0 / 0 | 10/10 | 8/10 |
| 2 px，全体 | 40% → 40% | 0 / 0 | 10/10 | 8/10 |
| 4 px，全体 | 40% → 60% | 2 / 0 | 10/10 | 8/10 |
| **8 px，全体压力测试（含拒收）** | **40% → 80%** | **4 / 0** | **4/10** | **4/10** |
| 8 px，仅合格子集 | 50% → 50% | 0 / 0 | 4/4 | 4/4 |

合格子集仅含 LaVie 自行车和马，因此其配对原片比例是 50%，不能与全体原片的
40% 混用。8 px 全体四次 0→1 都来自超质量门槛的列车和长颈鹿，不能据此单独
证明“语义不变的合格反事实”上 Origin 失效。全体变化 +0.40 的 prompt-cluster
95% CI 为 [0.00, 0.80]（5 个 prompt，2,000 次 bootstrap），仅作开发描述。

| 官方源 | Origin：base → 8 px（两种子一致） | Repair：base → seed 1701 | Repair：base → seed 2904 | 8 px 构造资格 |
|---|---|---|---|---|
| 自行车减速 / LaVie | 1 → 1 | 0.002819 → 0.001535 | 0.002819 → 0.003232 | 两条合格 |
| 桥上列车 / ModelScope | 0 → 1 | 0 → null | 0 → null | 两条拒收，仅压力观察 |
| 雪地自行车 / VideoCrafter | 1 → 1 | null → null | null → null | 两条拒收，仅压力观察 |
| 马 / LaVie | 0 → 0 | 0.003105 → 0.003695 | 0.003105 → 0.003598 | 两条合格 |
| 长颈鹿 / ModelScope | 0 → 1 | 0.006212 → null | 0.006212 → null | 两条拒收，仅压力观察 |

Repair 的 null 均是证据不足，不是零或成功保持不变。列车覆盖率降到
49.54% / 43.19%，长颈鹿为 58.61% / 57.41%，都低于未改动的 60% 门槛；雪地
自行车为 22.78% / 23.70%。全体 10 个干预仅 4 个 Repair 有效配对，不能以只剩
这四对的均值 `0.002962 → 0.003015` 宣称总体稳定。有效四对绝对变化均值为
0.000695，95% CI [0.000542, 0.000848]，也仅含两个源。

### 分母、产物和验证

本次新建 5 原片 + 5 编码控制 + 10 干预 = **20 候选**；14 合格、6 质量拒收，
0 构造运行失败。作为显式压力测试，20 条全部尝试评分：Origin **20/20 成功**；
Repair **12/20 有效、8/20 证据不足、0 运行失败**。终态 finished / completed=20；
非零退出码来自证据不足，没有遗漏、替换样本或放宽 Repair 门槛。

- 配置：[construction.local-texture-dev5-8px-v1.json](../../configs/dynamic-static-jitter/construction.local-texture-dev5-8px-v1.json)。
  本地 `output/dynamic-static-jitter/local-texture-dev5-8px-v1{,-scores,-analysis,-previews}/`；
  H200 同隔离 code 根下 `output/local-texture-dev5-8px-v1{,-scores}/`。
- [完整逐条 CSV](../../output/dynamic-static-jitter/local-texture-dev5-8px-v1-analysis/cases.csv)、
  [统计 JSON](../../output/dynamic-static-jitter/local-texture-dev5-8px-v1-analysis/summary.json)、
  [结果表](../../output/dynamic-static-jitter/local-texture-dev5-8px-v1-analysis/SUMMARY.md)。
  JSON 的 `tables` 保留严格合格子集，`stress_tables` 才包含质量拒收；CSV 同列记录
  原始 construction status / reason。两套统计不互相替代。
- 五条预览为原片 / 8 px seed 1701 / 8 px seed 2904；带星号的是质量拒收的压力
  样本。预览只是展示，不用于评分。复制到桌面 `dynamic_local_texture_dev5_8px_20260922/`，
  未覆盖上一轮桌面目录。
- 与上轮逐项核对：源 SHA、所有 metric/model/core 代码、权重、Repair config 不变；
  评分入口仅新增显式开发压力样本选择及状态记录，不改公式。十个原片/编码控制的
  两后端数值及状态均与上轮完全一致；首次原生官方 infer parity 仍通过。
  旧 runner / 分析 / 预览源码另存于 `local-texture-dev5-v1-scored-source/`，旧产物不重写。
- H200 物理 7 / 逻辑 cuda:0，2026-09-22 **14:08:33–14:09:05 UTC**，单卡约
  32 秒（逐视频循环 26.49 秒），环境同下文首轮。首次 SSH 握手超时未启动评分；
  确认无进程及输出后才重试，没有重复模型推理。
- 本地回归 **808 passed, 3 skipped**，锁与 diff 检查通过；五个原片未修改，
  20 条输入身份、10 份位移场及评分代码哈希已核验。未运行独立测试，未完成总体 goal。

## 当前实验：五条官方原视频的局部纹理空间抖动

用户最新指定的是“画面内部纹理局部抖动”，**不是加颜色噪声或亮度频闪**。
本轮实际实现并测量 5 条 VBench 1.0 官方完整原视频；没有抽帧静态化、生成新内容、
替换原片、调整 Repair 参数或扩跑独立测试。旧外观噪声结果移至下文历史段落。

### 构造及评分数据流

官方原 MP4 的每帧 `I_t` → 固定种子的平滑局部二维位移场 `u(x)` →
正负快速交替相位 `a_t` → **只对当前原帧重采样一次**
`I'_t(x) = I_t(x + a_t u(x))` → RGB 无损编码 → 解码像素/时间轴核验 →
将原始 MP4 和干预 MP4 分别送入不读取配对信息的 Origin / Repair。

- 位移场的空间均值为零，画面边缘衰减到零；相位时间均值为零。不整帧平移、不
  累积变形、不加 RGB 噪声。空间相关尺度、边界衰减宽度均为 32 原生像素。
- **1/2/4 px 是每帧最大位移，不是所有像素的位移，也不是相邻两帧的最大差**；
  两个相位之间最大跳动可达 2/4/8 px。种子固定为 1701、2904。
- 官方源按既有 dev metadata 顺序取五个不同 prompt，生成器预定轮换；不是按
  Origin/Repair 分数选择。清单固定于
  [sources.local-texture-dev5-v1.jsonl](../../configs/dynamic-static-jitter/sources.local-texture-dev5-v1.jsonl)。
- 5 条均为 16 帧、8 FPS、2 秒；LaVie 两条为 512×512，其余三条为 256×256。
  全部源帧及时间戳原样保留。30 条干预最小 warp Jacobian 为 **0.704081**，
  最低低频结构相关为 **0.979719**，所有边界位移为零，无越界采样或折叠。
- 两个评分器均采到全部 16 帧，全部 30 条干预都有 **15 次相位反转**；本批
  不存在因隔帧抽样而漏掉抖动的混叠。Repair 内部仍沿用 max-side 256，因此
  512 与 256 原生分辨率下的同档原生像素扰动，在模型输入上并非同一像素幅度。
- 构造代码：[local_texture_jitter.py](../../scripts/counterfactual/local_texture_jitter.py)；
  [协议配置](../../configs/dynamic-static-jitter/construction.local-texture-dev5-v1.json)。
  评分端沿用 CoTracker2 **dev-v4-tracker-denoise** 候选，未对这五条调参。

线性插值会改变高频纹理对比度；局部位移也可能轻微弯曲物体边缘，不能声称逐像素
几何完全不变或已经通过人工语义审核。这里检验的是对小幅、快速局部形变的鲁棒性。
两端不同量纲：Origin 是官方动态布尔值；Repair 是 **短边长度/秒**，不能直接
比较两列数值大小，也不能因 Repair 数值小就称其稳定。

### 完整分母与预定主展示

5 原片 + 5 零编辑编码 control + 30 干预 = **40 候选；40 构造合格，0 拒收/失败**。
Origin **40/40 成功**；Repair **32/40 有效、8/40 证据不足、0 运行失败**。
8 条证据不足均为雪地自行车这一源的全部变体，保留 `null`，不当作零分或漏跑。
脚本因此按契约退出 1，但 `runtime.json` 的终态是 finished、completed=40。

下面是评分前固定的 **2 px / seed 1701**，不是事后挑选效果最好的一档：

| 官方源 prompt（文件后缀） | 生成器 | Origin：base → CF | Repair：base → CF | Repair 证据覆盖率：base → CF |
|---|---|---|---|---|
| a bicycle slowing down to stop (-3) | LaVie | 1 → 1 | 0.002819 → 0.002644 | 79.77% → 79.77% |
| a train crossing over a tall bridge (-4) | ModelScope | 0 → 0 | 0.000000 → 0.005699 | 96.90% → 94.35% |
| a bicycle gliding through a snowy field (-2) | VideoCrafter | 1 → 1 | null → null | 33.29% → 30.88% |
| a horse taking a peaceful walk (-1) | LaVie | 0 → 0 | 0.003105 → 0.002793 | 91.90% → 91.16% |
| a giraffe taking a peaceful walk (-3) | ModelScope | 0 → 0 | 0.006212 → 0.005689 | 73.29% → 71.81% |

### 三档、两个种子的全部配对结果

Origin 列是 5 个源 × 2 种子的动态判定比例，**不是准确率**。Repair 均值只覆盖
两端有效的 4 个源 × 2 种子，覆盖率恒为 8/10；雪地自行车不在均值中但保留在分母。
95% CI 按 prompt 分组 bootstrap 2,000 次，不能把种子当独立视频。仅 4–5 个组，
这些区间是开发描述，不支持总体效果的强结论。

| 最大位移 | Origin 均值：base → CF | 0→1 / 1→0 | Repair 均值：base → CF | Repair 平均绝对变化 [95% CI] | 有效配对 O / R |
|---|---|---|---|---|---|
| 零编辑 control | 0.400 → 0.400 | 0 / 0 | 0.003034 → 0.003034 | 0 [0, 0] | 5 / 4 |
| 1 px | 0.400 → 0.400 | 0 / 0 | 0.003034 → 0.003456 | 0.000836 [0.000453, 0.001199] | 10 / 8 |
| 2 px | 0.400 → 0.400 | 0 / 0 | 0.003034 → 0.004491 | 0.001822 [0.000226, 0.004448] | 10 / 8 |
| 4 px | 0.400 → 0.600 | 2 / 0 | 0.003034 → 0.005671 | 0.003134 [0.000311, 0.007789] | 10 / 8 |

可支持与不可支持的结论：

1. **确实出现了 Origin 被局部抖动改变的例子**：长颈鹿的两个 4 px 种子都为
   0→1。超过官方 6 px 阈值的帧对数量由 2 增至 5/6，达到其 4 对判定门槛。
   Repair 分别为 0.006212→0.004256（约 −31.5%）和 0.006212→0.007702（约 +24.0%）。
   一条降分不代表另一条也稳定；不能只展示 seed 1701 然后宣称修复已成功。
2. **当前 Repair 对局部往复抖动仍会误增运动量**：列车原片为 0，2 px 时为
   0.005699/0.005995，4 px 时为 0.009956/0.010426。原算法验证空间对应后累计
   路径长度，没有额外分离高频往返纹理位移；这解释了仍将部分抖动计入运动的风险。
3. **有效覆盖不足**：雪地自行车原片覆盖仅 33.29%，低于固定 60% 门槛，全部
   八个版本均不足。未降低门槛、删除此源或用零值掩盖。现阶段不能证明原生运动
   响应充分、局部抖动不变性达标，更不能晋升默认 Repair。
4. 1/2 px 下 Origin 的 20 个布尔结果均未翻转。即使部分连续光流诊断变化，也
   不能冒充官方最终分数变化。五视频 pilot 已完成，数值容差冻结、更多源验证、
   人类偏好/独立测试仍 **NOT RUN**。

### 同一有效子集的消融诊断

均复用本批实际 CoTracker2 推理，不是另造模型结果。以下三列为配对平均绝对变化，
都限定在 full Repair 两端有效的同一 8 对上；不是各消融的全体覆盖率评估。

| 位移 | 原始轨迹量 | 对应关系验证后 | 完整 Repair（再加 0.35 px 噪声容限） |
|---|---:|---:|---:|
| 1 px | 0.001270 | 0.000858 | 0.000836 |
| 2 px | 0.004258 | 0.001860 | 0.001822 |
| 4 px | 0.009825 | 0.003175 | 0.003134 |

核验降低了部分扰动响应，但不足以解决列车误增和自然视频覆盖问题；也不能由较低
分差推断真实运动保留良好。此处不沿用旧合成协议的 ε 或验收门槛。

### 产物、版本与核验

H200 隔离代码根：`/data/chenjiayu/dynamic-static-jitter-20260922/code/`。
`output/local-texture-dev5-v1/` 是 40 条完整构造 ledger、30 份位移场 NPZ、35 个
派生 MP4（5 个原片直接引用只读官方目录）；`output/local-texture-dev5-v1-scores/`
保存 40 条评分、40 份轨迹证据、源码/权重哈希及运行记录。本地对应目录位于
`output/dynamic-static-jitter/`。

- [逐条结果 CSV](../../output/dynamic-static-jitter/local-texture-dev5-v1-analysis/cases.csv)、
  [完整统计/区间 JSON](../../output/dynamic-static-jitter/local-texture-dev5-v1-analysis/summary.json)、
  [生成的结果表](../../output/dynamic-static-jitter/local-texture-dev5-v1-analysis/SUMMARY.md)。
- [五条对比视频及预览清单](../../output/dynamic-static-jitter/local-texture-dev5-v1-previews/)：
  每条横排为原片 / 2 px / 4 px，均固定 seed 1701。预览原片栏使用已验证逐像素
  等同原片解码的零编辑 control；带文字、四次循环、有损展示版**从未送入评分**。
- 评分基于本地 HEAD `fdf4890c67bee5881e53fc43926d555a63f9cef8` 加未提交源码，
  不能将 HEAD 单独当作完整方法身份。完整逐文件 SHA 在 scoring `provenance.json`；
  构造源码 SHA 在 `construction.json`。远端额外纳入哈希的旧
  `analyze_static_jitter.py` 与本地版本不同，但本次评分不调用它，相关差异仅为
  旧合成分析；远端快照保留于 `local-texture-dev5-v1-scored-source/`。实际 metric、
  模型适配、评分入口及构造代码与本地一致；本次统计用原生视频专用分析入口。
- 上游固定 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`；CoTracker2 权重 SHA
  `362f5274376d610dc987b6daf2c2fefe63e06e1835f4ec1a10d0a15c5a4eef4f`；RAFT SHA
  `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`。
- H200 NVL 物理 7 / 逻辑 cuda:0，Python 3.10.20、torch 2.6.0+cu124，沿用现有
  环境与权重，未下载。2026-09-22 **13:35:10–13:39:44 UTC**，总墙钟约 274 秒，
  其中逐视频循环 256.10 秒；不是四卡计时，也未终止其他任务。
- 零编辑控制 Origin 5/5、有效 Repair 4/4 与原片完全相同；另外 1/1 Repair
  两端同为证据不足。首条 Origin 与原生 `DynamicDegree.infer` 判定一致，
  不将此称为全体 40 条原生入口 parity。所有分数均来自真实本地模型推理。
- 源文件/构造媒体/位移场哈希、评分输入身份与完整性已核对；冻结
  `data/ results/ splits/ runs/` 未改。复现命令与配置见
  [配置说明](../../configs/dynamic-static-jitter/README.md)。本轮无全量扩跑或重新调参。
- 本地全仓回归 **800 passed, 3 skipped**；新增局部坐标位移、边界、零均值、
  无加性噪声、原始逐帧重采样、实际无损编码、证据不足与采样混叠检测均有测试。
  `uv lock --check`、`git diff --check` 通过。五条预览及完整分数另复制到 Windows
  桌面 `dynamic_local_texture_dev5_20260922/`，不覆盖其他任务文件。

## 历史：官方原视频的 RGB 噪声/频闪预览（已被局部空间抖动口径取代）

**base 必须是 VBench 1.0 官方原视频，反事实只对原视频逐帧加入像素震颤。**
先前虽然从官方媒体取材，却将其抽帧静态化并另造运动阶梯，偏离了用户明确的
主实验要求。下文已有合成结果仅保留为历史开发诊断，不能作为修订后的主结果。

旧全量合成评分已按用户纠正中止：H200 的本任务 PID 91135、91138、91141
均经核验后 SIGINT，随后确认进程不存在；三个 shard 分别保留 744、788、771 条，
合计 **2,303 / 5,368** 条部分记录。没有删除产物、终止他人进程或将其称为完成。

新入口为 [official_video_jitter.py](../../scripts/counterfactual/official_video_jitter.py)，
配置为 [construction.official-dev-v1.json](../../configs/dynamic-static-jitter/construction.official-dev-v1.json)。
直接读取 H200 `/data/chenjiayu/wenbiao_zhao/vbench-official-v1/`：保留完整源帧、
原分辨率、原 FPS/时间轴，禁止静态化、裁剪、缩放和亮度归一化。
以 RGB lossless 编码逐帧验证 intended→decoded 像素完全一致，另列零编辑编码控制。
原 MP4 文件直接评分；CogVideo 官方 GIF 需要格式适配，因为上游原生 `infer`
不接受 GIF 路径。不能无证据改写其时间轴：首个 GIF 的 33 帧均无时延元数据，
OpenCV 的 10 FPS 不能当作文件声明的时间轴；初次新构造明确拒收，保留待处理，
不伪装为已完成适配。目前先运行原生 MP4 的配对实验。

首轮新协议预览已完成：4 个源 / 1 个 prompt 共 152 条候选，98 条合格；
54 条拒收中，38 条属于上述 GIF 时间元数据问题，16 条为原生 MP4 干预质量拒收。
98 条 Origin 均评分成功；Repair 有效 **63/98**，另 **35 条证据不足**，不是零分。
三个原 MP4 的 Origin 均为 1；VideoCrafter 原片加入强独立像素噪声后，两个种子
均 **1 → 0**，但其 Repair 原片和干预均证据不足。因此当前不能声称已实现
“Origin 改变而 Repair 保持稳定”。LaVie 的 Repair 原片为 0.00281854，部分频闪
会升至 0.00637702；即使原始量较小，相对变化仍大，不能用小数位制造稳定印象。

零编辑编码控制：Origin 3/3 与直接原片完全一致；Repair 有效的 2/2 数值完全
一致，第三个原片与 control 均证据不足。这一检查支持变化来自干预而非转码，
但不支持当前候选已经修复。三个 shard 的首个原片均与上游原生 `infer` 布尔一致。
新产物位于 H200 目标 code 下 `output/official-dev-preview-v1-s{0,1,2,3}/` 和
`output/official-dev-preview-v1-scores{1,2,3}/`；本地证据为
`output/dynamic-static-jitter/official-dev-preview-v1-analysis-v2/summary.json`，
包含完整逐族配对值、有效分母、失败状态与输入/权重/代码身份。禁止用旧合成
预览的 674/674 成功替代此处的 63/98。

新协议完整开发、标定和独立验证仍待完成。旧合成运动锚点 R/ε 不能沿用于原生视频
不变性验收；当前要求是绝对变化小，而非把原本有运动的视频压低分就算成功。
有效目标见 [修订协议](../plans/2026-09-22-dynamic-static-jitter-goal-prompt.md)。

执行目标见[goal](../plans/2026-09-22-dynamic-static-jitter-goal-prompt.md)，
构造口径和命令见[配置说明](../../configs/dynamic-static-jitter/README.md)。

## 方法数据流

Origin：当前视频 → 上游原生约 8 FPS 采样 → 相邻帧 RAFT 光流 → 每对帧最大的
5% 像素光流幅度均值 → 尺寸相关阈值与运动帧对计数 → **官方布尔判定**。
锁定上游 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。连续光流均值仅作机制
诊断，不是官方最终分数。

候选 Repair：当前视频及时间戳 → 固定物理时间采样与空间网格 → 本地预训练
CoTracker2 的正向轨迹及独立反向端点查询 → 局部归一化结构匹配、正反循环
误差、运动解释与同位置解释比较 → 有证据的位移 / 短边 / 秒 → 固定空间和
时间分母汇总。可靠静止贡献零，证据不足另报 `insufficient_evidence`、分数
`null`，不得当作成功零分。累计路径长度保留方向反转，不以净位移代替运动量。

消融输出为原始轨迹量、对应关系验证后、再加入定位噪声容限后的完整候选。
当前阈值仍为开发候选，须经开发标定后冻结。原有 CLI 默认方法尚未改变；
新方法必须显式 `--audit-variant trajectory`。

评分函数仅输入当前视频；构造标签、配对原片、种子与合成轨迹不传给评分器。
文件名仅用于 I/O。入口：[trajectory.py](../../metrics/dynamic-degree/src/dynamic_degree/trajectory.py)、
[模型封装](../../packages/audit-models/src/vbench_audit_models/point_tracker.py)。

## 旧合成协议的数据与划分（已撤销为主实验）

源清单共 152 条：开发 32 条 / 8 个 prompt，预留测试 120 条 / 30 个 prompt，
每个 prompt 覆盖 4 个生成器；同 prompt 的生成器与所有变体不跨集合。
来源为冻结 E0 manifest 的 `dynamics_degree`，视频复用 `subject_consistency/`。
这些自然视频曾用于历史评估，明确属于**新干预协议的分组留出验证**，不是全新
自然视频外部测试。输入清单不读取任何评分。

静态主实验从固定规则抽取的同一真实帧重复构成；六族外观干预不改坐标：独立
RGB 噪声、空间相关 RGB 噪声、全局交替/随机频闪、固定局部交替/随机频闪。
各含 8/20/40 三档和两个种子；clean 与所有变体共用相同编码。

真实运动控制采用固定面积的真实图像纹理卡片平移，保持背景和纹理不变，移动
路径不越界；另外保留往复运动。这是受控几何敏感性检查，不能代替自然对象
运动验证。自然视频外部有效性目前 **NOT RUN**。

## 历史合成开发证据（不属于修订后的主实验）

以下均是旧合成 pilot，不是官方原视频的配对结果，也不是正式测试结论。

| 运行 | 基底 / prompt | 候选 | 构造合格 | 拒收 | 实际评分 | 结果及限制 |
|---|---:|---:|---:|---:|---:|---|
| construction dev-v1 | 1 / 1 | 191 | 174 | 17 | Origin 36；Repair NOT RUN | 17 条高光裁切拒收；Origin 36/36 为 0；未观察到静态误判，亦未检出该低速卡片控制 |
| construction dev-v2 + trajectory dev-v1 | 4 / 1 | 764 | 674 | 90 | Origin 674；Repair 尝试 674、有效 629 | 45 条 Repair 证据不足；整体未达标 |
| dev-v2-highpass 描述子重放 | 同上 | 同上 | 同上 | 同上 | 重用 674 条真实模型轨迹，有效 632 | 42 条证据不足；未运行新的模型推理 |
| dev-v3-bandpass 描述子重放 | 同上 | 同上 | 同上 | 同上 | 重用 674 条真实模型轨迹，有效 673 | 仍有 1 条证据不足，强像素噪声运动排序未达标 |
| dev-v4 固定空间去噪 + 带通对应验证 | 同上 | 同上 | 同上 | 同上 | 674 条新模型推理，全部有效 | 强像素噪声阶梯 23/24 正确，响应保留 92.80%；仍仅一个 prompt，非原视频实验 |

v1 的两个交替种子同为奇数，产生相同相位；v2 改成奇偶种子。v1 保留，不将
重复相位计为独立干预证据。v2 将最高速度从 4 提升到 8 px/frame；剂量仍在
开发阶段，未冻结测试。v1 第一条的 adapter 布尔结果与上游原生 `infer` 均为
false；这只是一个案例的 parity，不是全量官方基线复现。

v2 按生成器的构造合格数：CogVideo 191/191、LaVie 161/191、ModelScope
131/191、VideoCrafter 191/191。构造拒收与原片、失败信息全部保留；不能只报告
合格项而隐去分母。自动结构检查还需要可视化复核。

当前只有一个 prompt 的预览，聚类置信区间没有可识别的独立组，故不报告 CI。
候选已有真实实测，但不能据此推断全面达标。初版的无效项主要集中于强噪声；
开发对照依次比较局部均值中心化、去除缓慢空间亮度变化、以及同时抑制高频
噪声的带通结构描述。v1–v3 模型仍读取原始解码帧，仅对应关系核验的描述子变化。
重放脚本只允许 `dev` 清单，测试数据会被拒绝；轨迹文件和原视频均校验哈希。

单个 CogVideo 开发例子（同一静态图像 + 强空间相关噪声）：Origin **0 → 1**；
dev-v3 Repair **0 → 0**，干预后的有效证据覆盖率 90.97%。该例的干净平移阶梯
为 **0 → 0.010383 → 0.022767 → 0.046865**（0/2/4/8 px/frame）。
这说明当前候选并非恒零，同时提供了目标失配的实际案例，不能代替总体通过。

dev-v3 在已有效评分的静态预览中最大分数为 0.00003335；四个中等运动基底
给出临时开发锚点 R=0.02347439、ε=0.00117372。**这不是最终冻结标定**。
强独立像素噪声的运动阶梯全候选相邻排序率仅 75%，仍须诊断；其他干预的
全候选排序还混有构造拒收，必须分别列出输入资格与评分缺失，不能混称模型
失效或直接丢弃。正式多 prompt 标定与独立测试均未完成。

## 原始产物与版本

H100 独立工作目录（未改写共享 checkout）：
`/root/wenbiao_zhao/tmp/dynamic-static-jitter-20260922/code/`。
其 `output/` 下保存：

- `dev-build-v1/candidates.jsonl`：191 条构造台账、原片身份、编码前重放哈希、
  解码时间轴、强度及资格；视频和预览在同目录。
- `dev-origin-smoke-v1/{scores.jsonl,provenance.json,runtime.json}`：36 条官方
  布尔结果与逐帧对光流诊断、精确源码文件哈希、RAFT 权重哈希、GPU 与耗时。
- `dev-origin-smoke-v1-analysis/{summary.json,SUMMARY.md}`：明确未完成的
  全候选表，缺失 Repair 为 NOT RUN，联合门槛 false。
- `dev-build-v2/candidates.jsonl`：764 条候选与 90 条结构检查拒收。
- `dev-build-v2-full-s{0,1,2,3}/`：四个互不重叠的开发构造 shard，合并为
  `dev-build-v2-full/candidates.jsonl`，32 个源 / 8 个 prompt / 6,112 条候选，
  已完成构造与逐合格文件哈希核验；旧合成评分部分完成后已按用户纠正中止。

H200 独立运行目录：`/data/chenjiayu/dynamic-static-jitter-20260922/code/`。
`output/dev-both-preview-v1-s{0,1}/` 保存 674 条真实 Origin/Repair 记录、轨迹
NPZ 与 provenance；使用当时空闲的物理 6、7 卡。原共享 VBench checkout
存在他人的未跟踪文件，没有删除；正式评分使用同目标命名空间内单独建立、
相同锁定 SHA 的干净 checkout。

H200 `dev-repair-preview-v4-s{0,1,2,3}/` 保存 674 条完整旧合成预览结果。
`dev-prepared-build-v2-s{0,1,2,3}/` 保存一次缺少 ffmpeg 的 6,112 条构造失败台账；
找到已有 imageio_ffmpeg 7.0.2 编码器后，在新目录
`dev-prepared-build-v2-ffmpeg7-s{0,1,2,3}/` 重建，5,368 合格 / 744 结构拒收。
编码器版本不同导致 MP4 字节变化，虽编码前像素一致，仍重跑 Origin，没有套用
旧编码器分数。`dev-full-v4-s{0,1,2}/` 是上述 2,303 条中止评分，不能算全量完成。

真实 CLI 单视频分数 0.0471759606566694 与批量 runner 对同一输入完全一致；
目前仅证明一个快速运动例子的 CLI parity，其他接口路径尚需完成对应验证。
本地 `output/dynamic-static-jitter/dev-preview-v{1-analysis,2-highpass-analysis,3-bandpass-analysis}/`
保存逐族表、开发临时标定和未通过门槛。旧结果没有覆盖。

本地复制产物位于被忽略的 `output/dynamic-static-jitter/`。大视频、轨迹缓存、
模型权重不提交 Git。源清单 SHA256：
`ed46df1c5e7ca1a56dcf542b4d171fb8f6a4d4756fab0c7340ed3fafc434bd56`。
实现基于仓库 `fdf4890` 加本任务工作树改动；开发评分的实际源码文件哈希由
`provenance.json` 记录，不将未提交实现声称为 `fdf4890` 原生内容。

已有 CoTracker2 权重源为 H200 本地 torch hub cache，源 SHA256：
`362f5274376d610dc987b6daf2c2fefe63e06e1835f4ec1a10d0a15c5a4eef4f`。
跨机副本在校验完整哈希之前不得用于评分。没有下载新权重或改写上游代码。

## 验证与剩余工作

已执行：`uv lock --check`、锁定依赖同步与 CPU torch overlay、全仓 CPU 测试
（阶段全量 785 passed / 3 skipped）、12 个 CLI 的 `--help`、
`git diff --check`；随后官方原视频构造与分析定向测试 8/8 通过，其中实际调用
本地 ffmpeg 验证无损像素和原生时间轴。最终代码还需重跑全量。
真实 CoTracker 推理与单例 CLI parity 已验证；其他验证范围不得由 CPU 测试外推。

按修订协议尚未完成：官方原视频的多 prompt 开发测量、模型与可靠性参数比较、噪声容限标定、
自然运动验证、真实候选 CLI parity、独立测试协议冻结及评分、聚类 CI 与
联合验收、论文可用案例、最终 scoped commit/push。ε 仍未标定，不沿用
Subject 的 0.01。仅当静态低分/不抬分、真实运动响应、覆盖和独立验证同时
达标，才能宣布 goal 完成。
