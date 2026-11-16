from dataclasses import replace
from pathlib import Path

import torch

import ntfields
from ntfields import DistillConfig, Epochs, Problem, TrainConfig, TrainedField
from ntfields.data import sample_pairs
from ntfields.distill import GRAD_G_KEY, GRAD_S_KEY, T_KEY, TeacherLabels, distill_loss
from ntfields.methods.td_ntfields.config import TDNTFieldsConfig
from ntfields.registry import create


def _teacher(problem: Problem) -> TrainedField:
    torch.manual_seed(1)
    method = create("td_ntfields", replace(TDNTFieldsConfig.planar(), hidden=64))
    return TrainedField(method, problem, method.build_model(problem).eval())


def test_teacher_labels_match_the_teacher(problem: Problem):
    teacher = _teacher(problem)
    pairs = sample_pairs(problem, 32)
    batch = next(TeacherLabels(pairs, teacher).batches(32, problem.env.condition()))
    torch.testing.assert_close(batch.extras[T_KEY], teacher.travel_time(batch.q_s, batch.q_g))
    g_s, g_g = teacher.gradient(batch.q_s, batch.q_g)
    torch.testing.assert_close(batch.extras[GRAD_S_KEY], g_s)
    torch.testing.assert_close(batch.extras[GRAD_G_KEY], g_g)


def test_distilled_student_approaches_teacher_and_round_trips(problem: Problem, tmp_path: Path):
    teacher = _teacher(problem)
    student_cfg = replace(TDNTFieldsConfig.planar(), hidden=32, n_blocks=1, lr=2e-3, weight_decay=0.0)
    tc = TrainConfig(n_pairs=512, batch_size=128, device="cpu", out_dir=tmp_path)
    q_s, q_g = sample_pairs(problem, 256, seed=7).q_s, sample_pairs(problem, 256, seed=8).q_g

    def error(field: TrainedField) -> float:
        return float((field.travel_time(q_s, q_g) - teacher.travel_time(q_s, q_g)).abs().mean())

    untrained = ntfields.distill(
        teacher, "td_ntfields", Epochs(0), cfg=student_cfg, train_cfg=replace(tc, out_dir=None)
    )
    student = ntfields.distill(teacher, "td_ntfields", Epochs(40), cfg=student_cfg, train_cfg=tc)
    assert error(student) < 0.5 * error(untrained)

    assert student.ckpt_path is not None
    loaded = ntfields.load(student.ckpt_path)
    assert loaded.method.name == "td_ntfields" and loaded.method.cfg == student_cfg
    torch.testing.assert_close(loaded.travel_time(q_s, q_g), student.travel_time(q_s, q_g))


def test_distill_with_physics_term_runs(problem: Problem):
    teacher = _teacher(problem)
    tc = TrainConfig(n_pairs=64, batch_size=64, device="cpu", out_dir=None)
    cfg = replace(TDNTFieldsConfig.planar(), hidden=32, n_blocks=1)
    field = ntfields.distill(
        teacher, "td_ntfields", Epochs(1), cfg=cfg, distill_cfg=DistillConfig(w_physics=1.0), train_cfg=tc
    )
    assert field.method.name == "td_ntfields"


def test_near_weight_emphasizes_slow_pairs(problem: Problem):
    """With near_weight, the loss is a mean weighted toward low-speed pairs (weights normalized to mean 1)."""
    teacher = _teacher(problem)
    batch = next(TeacherLabels(sample_pairs(problem, 64), teacher).batches(64, problem.env.condition()))
    model = _teacher(problem).model
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    cfg = DistillConfig(w_grad=0.0, near_weight=4.0)
    t = model(batch.q_s, batch.q_g, batch.scene).detach()
    err = (t - batch.extras[T_KEY]).pow(2)
    w = 1 + 4.0 * (1 - torch.minimum(batch.speed_s, batch.speed_g))
    expected = (w / w.mean() * err).mean()
    torch.testing.assert_close(distill_loss(model, batch, cfg).total, expected)
