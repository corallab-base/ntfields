"""Latency of TD-NTFields' accelerated inference form against model size and batch size, to pick student sizes.

Weights are random (latency does not depend on them). Each function is compiled and replayed from a CUDA graph, as
`field.accelerate("compile")` does.

    uv run python benchmarks/distill/profile_sizes.py --dim 6
"""

import itertools
from dataclasses import dataclass

import torch
import tyro

from ntfields.accel import COMPILE_MODE, CudaGraphs
from ntfields.methods.td_ntfields.inference import fold
from ntfields.methods.td_ntfields.model import TDMetricModel


@dataclass
class Args:
    dim: int = 6
    hidden: tuple[int, ...] = (64, 128, 256, 512)
    n_blocks: tuple[int, ...] = (1, 2)
    batch: tuple[int, ...] = (100, 1024, 8192, 65536)
    iters: int = 200


def time_ms(fn: object, *args: torch.Tensor, iters: int) -> float:
    assert callable(fn)
    for _ in range(10):
        fn(*args)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main(args: Args) -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda")
    print(f"# {torch.cuda.get_device_name()}, dim {args.dim}; ms per call (travel_time | gradient)")
    print("hidden blocks params " + " ".join(f"{f'B={b}':>17}" for b in args.batch))
    for hidden, n_blocks in itertools.product(args.hidden, args.n_blocks):
        torch._dynamo.reset()
        model = TDMetricModel(args.dim, hidden, n_blocks).to(dev)
        params = sum(p.numel() for p in model.parameters())
        inf = fold(model).map(lambda fn: CudaGraphs(torch.compile(fn, mode=COMPILE_MODE, dynamic=False)))
        assert inf.gradient is not None
        e_g = inf.encode_goal(torch.rand(1, args.dim, device=dev))
        cells = []
        for b in args.batch:
            q = torch.rand(b, args.dim, device=dev)
            t = time_ms(inf.travel_time, q, e_g, iters=args.iters)
            g = time_ms(inf.gradient, q, e_g, iters=args.iters)
            cells.append(f"{t:7.3f} | {g:7.3f}")
        print(f"{hidden:6d} {n_blocks:6d} {params:6d} " + " ".join(cells), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
