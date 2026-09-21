# Color：后端与验证进展

独立 uv 包，使用 audit-core 的统一 CLI、元数据、分片与结果格式。
Official 直调锁定上游 `compute_color`；Repair 核对两路 box/order，必要时一对一 IoU 关联，
歧义不绑定；颜色采用有词边界、否定及对象归属检查的确定性谓词，全帧分母。
`binding` / `binding_lexical` / `repair` 分开消融，原始 tuple 消费仅作为明确标记的 trace 诊断。

2026-09-20：构建、help、实例置换、red/colored/hundred、零可见帧保留等纯测试通过。
25 个预选基底中 7 个通过独立定位和施工检查，构成 42 条请求；18 个拒收保留且不补选。
H100 Official 返回 33 条、上游丢弃 9 条；统一结果仍为 42 条。Repair 42/42 完成。
5 个合格 test bases 的 100%→0% 中位降分为 1.0，95% CI [0.625,1.0]，五档均严格单调；
两个 dev bases 的响应门槛未通过，零效应保留。Official 同义控制无有效配对，不能宣称双方都平。
binding-only / binding+lexical 由真实 trace 重放，原规则重放与 true Official 的有效值误差为 0。
独立 Color LoRA 已训练 300 steps，四级原始 query 视频消融已实测；确定性/base/LoRA 修复分相同，
未显示该小队列的视频增量。真实四卡 CLI 8/8 结果对齐，最大误差 0。
最新全量 676 passed / 3 skipped；单独 wheel 在仓库外安装、help 与失败输出通过。不能凭均值或覆盖率
宣称修复更好，silver 参考一致性不能当作人工标注精度。

配置、命令及限制见 [两维执行记录](../../docs/object-color-repair.md)。
