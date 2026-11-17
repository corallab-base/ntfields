import math
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ntfields.plan.gradient_bidirectional import PlanResult

if TYPE_CHECKING:
    from ntfields.field import Field, GoalField

# MPPI iterations per CUDA graph replay; the goal is checked (one host sync) after each replay once it is reachable
CHUNK = 8


def mppi(
    field: "Field",
    start: Tensor,
    goal: Tensor,
    samples: int = 50,
    horizon: int = 5,
    step: float = 0.015,
    prior_gain: float = 2.0,
    first_weight: float = 10.0,
    temperature: float = 50.0,
    goal_tol: float = 0.01,
    max_steps: int = 200,
    generator: torch.Generator | None = None,
) -> PlanResult:
    """TD-NTFields' MPPI (ntrl-demo `arm_plan_stat.py::MPPI`): only the start moves.

    Each iteration samples `samples` rollouts of `horizon` steps (noise of std `step` per rollout plus per step,
    around `prior_gain`·previous step, each step clipped to length `step`), scores them by
    `first_weight`·T(first state, goal) + T(last state, goal), and takes the softmax(-temperature·cost)-weighted
    first step.

    On a field accelerated with CUDA graphs (and the default generator), the iterations run from a CUDA graph of
    `CHUNK` iterations, with no host sync until the goal can be within reach; the path and the generator's state
    afterwards are the eager loop's.
    """
    target = field.towards(goal)
    args = (samples, horizon, step, prior_gain, first_weight, temperature, goal_tol)
    captured = _captured_fn(target) if generator is None and start.is_cuda else None
    if captured is not None:
        return _CapturedMPPI.get(captured, start, *args).plan(target.e_g, start, goal, max_steps)
    dim = start.shape[-1]
    q, prior = start.clone(), torch.zeros_like(start)
    path, reached = [start], False
    for _ in range(max_steps):
        noise = step * _randn(samples, 1, dim, like=start, generator=generator)
        noise = noise + step * _randn(samples, horizon, dim, like=start, generator=generator)
        steps = noise + prior_gain * prior
        steps = steps / (steps.norm(dim=2, keepdim=True).clamp(min=step) / step)
        ends = (q + steps.cumsum(dim=1))[:, [0, -1]].reshape(-1, dim)
        cost = target.travel_time(ends).reshape(samples, 2)
        weight = torch.softmax(-temperature * (first_weight * cost[:, 0] + cost[:, 1]), dim=0)
        prior = weight @ steps[:, 0]
        q = q + prior
        path.append(q)
        if (goal - q).norm() < goal_tol:
            reached = True
            break
    path.append(goal)
    return PlanResult(torch.stack(path), reached)


def _randn(*shape: int, like: Tensor, generator: torch.Generator | None) -> Tensor:
    return torch.randn(shape, generator=generator, dtype=like.dtype, device=like.device)


def _captured_fn(target: "GoalField") -> Callable[[Tensor, Tensor], Tensor] | None:
    """The function under travel_time's CudaGraphs wrapper, if it has one (graphs cannot be captured inside graphs)."""
    from ntfields.accel.cuda_graphs import CudaGraphs

    fn = target.inference.travel_time
    return fn.fn if isinstance(fn, CudaGraphs) else None  # pyright: ignore[reportReturnType]


_CACHE: "weakref.WeakKeyDictionary[Callable[..., Tensor], dict[tuple[object, ...], _CapturedMPPI]]" = (
    weakref.WeakKeyDictionary()
)  # per travel_time function, per planner arguments


class _CapturedMPPI:
    """`CHUNK` MPPI iterations captured in one CUDA graph, on static buffers (state, goal embedding, visited states
    and whether each reached the goal). The noise is drawn inside the graph from the default generator in the eager
    loop's order, so a replay draws what CHUNK eager iterations would."""

    @classmethod
    def get(cls, fn: Callable[[Tensor, Tensor], Tensor], start: Tensor, *args: float) -> "_CapturedMPPI":
        key = (start.shape, start.dtype, start.device, *args)
        graphs = _CACHE.setdefault(fn, {})
        if key not in graphs:
            graphs[key] = cls(fn, start, *args)  # pyright: ignore[reportArgumentType]
        return graphs[key]

    def __init__(
        self,
        fn: Callable[[Tensor, Tensor], Tensor],
        start: Tensor,
        samples: int,
        horizon: int,
        step: float,
        prior_gain: float,
        first_weight: float,
        temperature: float,
        goal_tol: float,
    ) -> None:
        self.fn, self.samples, self.horizon, self.step = fn, samples, horizon, step
        self.prior_gain, self.first_weight, self.temperature, self.goal_tol = (
            prior_gain,
            first_weight,
            temperature,
            goal_tol,
        )
        self.generator = torch.cuda.default_generators[start.device.index or 0]
        self.q, self.prior, self.goal = start.clone(), torch.zeros_like(start), start.clone()
        self.e_g: Tensor | None = None
        self.visited = start.new_empty(CHUNK, start.shape[-1])
        self.reached = torch.zeros(CHUNK, dtype=torch.bool, device=start.device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.offset_per_iteration = 0

    def _iterations(self) -> None:
        assert self.e_g is not None
        dim, step = self.q.shape[-1], self.step
        for i in range(CHUNK):
            noise = step * torch.randn(self.samples, 1, dim, dtype=self.q.dtype, device=self.q.device)
            noise = noise + step * torch.randn(
                self.samples, self.horizon, dim, dtype=self.q.dtype, device=self.q.device
            )
            steps = noise + self.prior_gain * self.prior
            steps = steps / (steps.norm(dim=2, keepdim=True).clamp(min=step) / step)
            c = self.q + steps.cumsum(dim=1)
            ends = torch.stack((c[:, 0], c[:, -1]), dim=1).reshape(-1, dim)
            cost = self.fn(ends, self.e_g).reshape(self.samples, 2)
            weight = torch.softmax(-self.temperature * (self.first_weight * cost[:, 0] + cost[:, 1]), dim=0)
            self.prior.copy_(weight @ steps[:, 0])
            self.q.add_(self.prior)
            self.visited[i].copy_(self.q)
            self.reached[i] = (self.goal - self.q).norm() < self.goal_tol

    def _capture(self) -> None:
        # warm up on a side stream (lazy initialization, autotuning), as torch.cuda.graph requires; the warm-up's
        # draws are rolled back so the first plan sees the generator as the eager loop would
        offset = self.generator.get_offset()
        with torch.no_grad():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._iterations()
            torch.cuda.current_stream().wait_stream(stream)
            self.generator.set_offset(offset)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._iterations()
            self.generator.set_offset(offset)
            self.graph.replay()
            self.offset_per_iteration, rem = divmod(self.generator.get_offset() - offset, CHUNK)
            assert rem == 0, "a CUDA graph replay advanced the generator by a non-multiple of its iterations"
            self.generator.set_offset(offset)

    def plan(self, e_g: Tensor, start: Tensor, goal: Tensor, max_steps: int) -> PlanResult:
        if self.e_g is None:
            self.e_g = e_g.clone()
        self.e_g.copy_(e_g)
        self.goal.copy_(goal)
        if self.graph is None:
            self.q.copy_(start)
            self.prior.zero_()
            self._capture()
        assert self.graph is not None
        self.q.copy_(start)
        self.prior.zero_()
        offset = self.generator.get_offset()
        # each step is at most `step` long (a convex combination of steps clipped to it), so no iteration before
        # this one can reach the goal; checking is only skipped, never trusted, so rounding cannot change the path
        reachable = max(0, math.floor((float((goal - start).norm()) - self.goal_tol) / self.step) - 1)
        visited, reached, n, hit = [], [], 0, None
        while n < max_steps:
            self.graph.replay()
            visited.append(self.visited.clone())
            reached.append(self.reached.clone())
            n += CHUNK
            if n >= reachable:
                hits = torch.cat(reached)[:max_steps].nonzero()
                if len(hits):
                    hit = int(hits[0])
                    break
        steps = hit + 1 if hit is not None else max_steps
        self.generator.set_offset(offset + steps * self.offset_per_iteration)
        path = torch.cat((start[None], torch.cat(visited)[:steps], goal[None]))
        return PlanResult(path, hit is not None)
