"""CUDA-graph capture for the CVPN forward (MX550 specialization).

At the small batch sizes the search / single-observation inference use, the
forward is dominated by kernel-launch and Python overhead (see
``reports/explore/mx550_kernels.md``). Capturing the whole forward into a CUDA
graph replays it with a single launch, removing that overhead.

Graphs are keyed by input shape and captured lazily on first use (inputs are
copied into static buffers each call). Inference only: no gradients, static
shapes, and the returned tensors are *clones* of the graph-owned outputs so the
caller may keep them across replays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GraphForward:
    """Wrap ``forward(*tensors)`` with a per-shape CUDA-graph replay cache."""

    forward: Callable[..., Any]
    max_entries: int = 32
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __call__(self, *inputs):
        import torch

        if not inputs or not inputs[0].is_cuda:
            return self.forward(*inputs)
        key = tuple((tuple(t.shape), t.dtype) for t in inputs)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._capture(*inputs)
            if len(self._cache) >= self.max_entries:
                self._cache.clear()
            self._cache[key] = entry
        static, graph, out = entry
        for buffer, tensor in zip(static, inputs):
            buffer.copy_(tensor)
        graph.replay()
        return tuple(o.clone() for o in out)

    def _capture(self, *inputs):
        import torch

        static = [t.clone() for t in inputs]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.forward(*static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.forward(*static)
        return static, graph, out

    def clear(self) -> None:
        self._cache.clear()
