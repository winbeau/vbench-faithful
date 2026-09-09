from vbench_audit_core.errors import DependencyError


def official_backend(video: str):
    raise DependencyError(
        "dynamic-degree 官方 backend 尚未迁移：需要先固定 VBench1.0 RAFT/torch/CUDA 依赖和权重，"
        "当前不会用替代算法冒充官方结果"
    )
