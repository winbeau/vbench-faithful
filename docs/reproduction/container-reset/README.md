# 容器重置交接记录

项目恢复依赖已经保存到公开 HF 数据库；选定模型保存在公开 HF 模型库。
完整工程和执行入口保存在私有 GitHub。恢复命令见 [运行手册](../CONTAINER_RESET.md)。

- [发布归档](hf-backup-receipt.json)：8 份归档、15,500,728,039 字节，前 7 份另有 120 个可续传分片。
- [H100 九维代表性评分](h100-repair-verification.json)：19 条输入全部通过，最大分数误差为 0。
- [H100 官方 16 维](h100-official-verification.json)与 [H200 官方 16 维](h200-official-verification.json)：各维一条真实视频，均通过；[跨机器最大差](official-cross-machine.json)约 9.3e-9。
- [代码与工作区检查](local-validation.json)：1289 passed、3 skipped，16 个 CLI help 通过；完整冻结主表复算核对 24,408 个分数单元。
- [验收状态](status.json)：按用户要求停止后续端到端测试。H200 的 Qwen 基座下载和九维修复安装验收未完成，已有下载保留；不宣称全量 GPU 重跑或从头训练通过。
- [执行代码身份](implementation.json)：508 个源码／配置文件与 H200 验证目录完全相同。核心实现提交为 `1b5808e9a2b5d6db2161cf3986d92b05e990d8f9`，也已同步到 H100 持久 checkout。

H200 的额外系统 Python 3.12 主表复算触发严格相等校验；未放宽阈值、未改冻结数据。
文档已明确用项目锁定的 Python 3.11.14 执行精确重放。此前在锁定环境中的完整复算通过。
