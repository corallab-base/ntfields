"""Teacher-student distillation: train a (smaller) student model to reproduce a trained field.

    student = ntfields.distill(teacher, "td_ntfields_arm", Epochs(200), cfg=replace(TDNTFieldsConfig.arm(), hidden=128))

The student is fit to the teacher's travel times and their gradients (the planners follow ∇T, so matching T alone
is not enough), optionally plus the student method's own physics loss. Any method can teach any other: the teacher
is only queried as a Field (an AcceleratedField labels fastest), and the student is built, optimized and checkpointed
by its own method, so the result loads and accelerates like any field of that method.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from ntfields.api import TrainConfig, resolve_method, train_with
from ntfields.base import Batch, LossOut, Method, MethodConfig, SceneCondition, TrainState, TravelTimeModel
from ntfields.data import PairDataset, PairSource, Sampler
from ntfields.field import Field, TrainedField
from ntfields.problem import Problem
from ntfields.train import Budget, Callback

T_KEY, GRAD_S_KEY, GRAD_G_KEY = "teacher_t", "teacher_grad_s", "teacher_grad_g"


@dataclass
class DistillConfig:
    w_time: float = 1.0  # (T - T_teacher)²
    w_grad: float = 0.1  # ‖∇_{q_s} T - teacher's‖² + ‖∇_{q_g} T - teacher's‖²
    w_physics: float = 0.0  # weight of the student method's own loss
    # pairs weighted 1 + near_weight·(1 - min(S*_s, S*_g)), normalized to mean 1 per batch: more weight near obstacles,
    # where the field's errors become collisions
    near_weight: float = 0.0
    # the student method's callbacks (e.g. TD-NTFields' rollback and loss scaling), which are tuned for its own loss
    method_callbacks: bool = False


@dataclass
class TeacherLabels:
    """A PairSource whose batches carry the teacher's T, ∇_{q_s} T and ∇_{q_g} T as extras."""

    source: PairSource
    teacher: Field

    def batches(
        self, batch_size: int, scene: SceneCondition, generator: torch.Generator | None = None
    ) -> Iterator[Batch]:
        for batch in self.source.batches(batch_size, scene, generator):
            t = self.teacher.travel_time(batch.q_s, batch.q_g)
            g_s, g_g = self.teacher.gradient(batch.q_s, batch.q_g)
            labels = {T_KEY: t.detach(), GRAD_S_KEY: g_s.detach(), GRAD_G_KEY: g_g.detach()}
            yield replace(batch, extras={**batch.extras, **labels})


def distill_loss(model: TravelTimeModel, batch: Batch, cfg: DistillConfig) -> LossOut:
    """w_time·(T - T_teacher)² + w_grad·(‖Δ∇_{q_s} T‖² + ‖Δ∇_{q_g} T‖²), a weighted batch mean (see near_weight);
    labels from TeacherLabels."""
    q_s = batch.q_s.detach().requires_grad_(cfg.w_grad > 0)
    q_g = batch.q_g.detach().requires_grad_(cfg.w_grad > 0)
    w = 1 + cfg.near_weight * (1 - torch.minimum(batch.speed_s, batch.speed_g).clamp(0, 1))
    w = w / w.mean()
    t = model(q_s, q_g, batch.scene)
    time = (w * (t - batch.extras[T_KEY]).pow(2)).mean()
    total = cfg.w_time * time
    terms = {"time": float(time.detach())}
    if cfg.w_grad > 0:
        g_s, g_g = torch.autograd.grad(t.sum(), (q_s, q_g), create_graph=True)
        grad = (g_s - batch.extras[GRAD_S_KEY]).pow(2).sum(-1) + (g_g - batch.extras[GRAD_G_KEY]).pow(2).sum(-1)
        grad = (w * grad).mean()
        total = total + cfg.w_grad * grad
        terms["grad"] = float(grad.detach())
    return LossOut(total, terms)


class Distillation(Method[Any]):
    """The student method with the distillation loss: model, optimizer and config (lr, gradient clipping) are the
    student's. Only what `fit` runs; the student method itself is what gets checkpointed."""

    name = "distillation"
    Config = MethodConfig

    def __init__(self, student: Method[Any], cfg: DistillConfig, callbacks: Sequence[Callback] = ()) -> None:
        super().__init__(student.cfg)
        self.student = student
        self.distill_cfg = cfg
        self.extra_callbacks = list(callbacks)

    def build_model(self, problem: Problem) -> TravelTimeModel:
        return self.student.build_model(problem)

    def optimizer(self, model: TravelTimeModel) -> torch.optim.Optimizer:
        return self.student.optimizer(model)

    def loss(self, model: TravelTimeModel, batch: Batch, state: TrainState) -> LossOut:
        out = distill_loss(model, batch, self.distill_cfg)
        if self.distill_cfg.w_physics == 0:
            return out
        physics = self.student.loss(model, batch, state)
        terms = {**out.terms, **{f"physics/{k}": v for k, v in physics.terms.items()}}
        return LossOut(out.total + self.distill_cfg.w_physics * physics.total, terms)

    def callbacks(self) -> list[Callback]:
        own = self.student.callbacks() if self.distill_cfg.method_callbacks else []
        return [*own, *self.extra_callbacks]


def distill(
    teacher: Field,
    student: str | Method[Any],
    budget: Budget,
    cfg: MethodConfig | None = None,
    distill_cfg: DistillConfig | None = None,
    train_cfg: TrainConfig | None = None,
    sampler: Sampler | None = None,
    data: PairDataset | None = None,
    callbacks: Sequence[Callback] = (),
) -> TrainedField:
    """Train `student` (a registered name or preset, or an instance) on `teacher.problem` to reproduce `teacher`.
    Pairs come from `sampler` (default: the student method's) or `data`, as in `ntfields.train`, and are labelled by
    the teacher batch by batch. `callbacks` (e.g. an LRScheduler) run besides the logger and checkpointing.
    `train_cfg.device` must be the teacher's."""
    method = resolve_method(student, cfg)
    tc = train_cfg or TrainConfig()
    if torch.device(tc.device).type != teacher.device.type:
        raise ValueError(f"train_cfg.device is {tc.device}, the teacher is on {teacher.device}")
    return train_with(
        method,
        Distillation(method, distill_cfg or DistillConfig(), callbacks),
        teacher.problem,
        budget,
        tc,
        sampler,
        data,
        source=lambda pairs: TeacherLabels(pairs, teacher),
    )
