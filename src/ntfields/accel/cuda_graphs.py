from collections.abc import Callable

import torch
from torch import Tensor

Outputs = Tensor | tuple[Tensor, ...]


class CudaGraphs:
    """Replays fn from a CUDA graph captured per input shape. Inputs are copied into the graph's static buffers
    (so one graph serves every goal embedding) and outputs are cloned out of them (so they outlive the next call).
    fn must be a plain forward computation: no autograd and no host synchronization."""

    def __init__(self, fn: Callable[..., Outputs]) -> None:
        self.fn = fn
        self._graphs: dict[tuple[object, ...], tuple[torch.cuda.CUDAGraph, tuple[Tensor, ...], Outputs]] = {}

    def __call__(self, *args: Tensor) -> Outputs:
        key = tuple((a.shape, a.dtype, a.device) for a in args)
        if key not in self._graphs:
            self._graphs[key] = self._capture(args)
        graph, inputs, out = self._graphs[key]
        for buffer, arg in zip(inputs, args, strict=True):
            buffer.copy_(arg)
        graph.replay()
        return out.clone() if isinstance(out, Tensor) else tuple(o.clone() for o in out)

    def _capture(self, args: tuple[Tensor, ...]) -> tuple[torch.cuda.CUDAGraph, tuple[Tensor, ...], Outputs]:
        inputs = tuple(a.clone() for a in args)
        with torch.no_grad():
            # warm up on a side stream (lazy initialization, autotuning), as torch.cuda.graph requires
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.fn(*inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self.fn(*inputs)
        return graph, inputs, out
