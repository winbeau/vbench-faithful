"""Replay a frozen image encoder's original CUDA kernels at a fixed shape."""
from __future__ import annotations


class CudaImageGraph:
    """Inference-only, single-input/single-output adapter for the SAM encoder.

    No batching, autocast, compiler fusion or precision changes. A changed input
    layout or model state uses the original forward. Returned tensors are cloned
    because callers must not observe later replays overwriting earlier outputs.
    Install only on an audited stateless encoder, after loading weights/eval().
    """

    def __init__(self, module):
        self.module, self.original = module, module.forward
        self.graph = None
        self.signature = None
        self.disabled = False
        self.replays = 0

    def __call__(self, value):
        import torch

        if (self.disabled or value.device.type != "cuda" or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or any(layer.training for layer in self.module.modules())):
            return self.original(value)
        tensors = (*self.module.parameters(), *self.module.buffers())
        state = tuple((id(p), None if p.is_inference() else p._version, p.device, p.dtype) for p in tensors)
        precision = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                     torch.are_deterministic_algorithms_enabled())
        signature = (tuple(value.shape), tuple(value.stride()), value.device, value.dtype, state, precision)
        if self.graph is not None and signature != self.signature:
            return self.original(value)
        if self.graph is None:
            try:
                self.input = torch.empty_like(value)
                self.input.copy_(value)
                stream = torch.cuda.Stream(device=value.device)
                current = torch.cuda.current_stream(value.device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.original(self.input)
                current.wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.output = self.original(self.input)
                if not isinstance(self.output, torch.Tensor):
                    raise TypeError("CUDA image graph requires a tensor output")
                self.graph, self.signature = graph, signature
                print("CUDA image encoder graph captured; original dtype and batch shape retained", flush=True)
            except RuntimeError as exc:
                self.disabled = True
                self.graph = None
                print(f"CUDA image graph unavailable; using original encoder: {exc}", flush=True)
                return self.original(value)
        self.input.copy_(value)
        self.graph.replay()
        self.replays += 1
        return self.output.clone()


def capture_image_encoder(module):
    adapter = CudaImageGraph(module)
    module.forward = adapter
    return adapter
