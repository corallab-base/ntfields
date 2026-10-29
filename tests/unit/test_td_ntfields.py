from dataclasses import asdict
from pathlib import Path

import torch

import ntfields
from ntfields import Batch, Problem, SceneCondition, TrainConfig
from ntfields.checkpoint import config_from_dict
from ntfields.envs import load_env
from ntfields.methods.td_ntfields import TDNTFields, TDNTFieldsConfig
from ntfields.methods.td_ntfields.loss import td_ntfields_loss
from ntfields.methods.td_ntfields.model import TDMetricModel
from ntfields.registry import default_config
from ntfields.speed import ClippedEased
from ntfields.testing import check_method


def test_check_method():
    check_method(TDNTFields)


def test_presets_are_registered():
    assert default_config("td_ntfields_arm") == TDNTFieldsConfig.arm()
    assert default_config("td_ntfields_stable") == TDNTFieldsConfig(detach_causality=True)


def test_td_term_is_masked_near_the_goal():
    torch.manual_seed(0)
    model = TDMetricModel(2, hidden=32)
    q = torch.rand(4, 2)
    ones = torch.ones(4)
    batch = Batch(q, q + 1e-4, ones, ones, torch.zeros(4, 2), torch.zeros(4, 2), SceneCondition())
    # T(q, q + ε) is far below Δt / S*, so the TD residual is masked out
    assert td_ntfields_loss(model, batch, TDNTFieldsConfig(hidden=32, dt=0.03)).terms["td"] == 0.0


def test_train_and_reload(tmp_path: Path):
    problem = Problem(load_env("box2d/one_sphere"), ClippedEased())
    tc = TrainConfig(n_pairs=256, batch_size=64, batches_per_epoch=2, device="cpu", out_dir=tmp_path)
    field = ntfields.train("td_ntfields", problem, ntfields.Epochs(3), cfg=TDNTFieldsConfig(hidden=32), train_cfg=tc)
    assert field.ckpt_path is not None
    loaded = ntfields.load(field.ckpt_path)
    assert loaded.method.cfg == field.method.cfg
    q = torch.rand(8, 2) - 0.5
    torch.testing.assert_close(loaded.travel_time(q, -q), field.travel_time(q, -q))


def test_surface_mix_preset_round_trips():
    cfg = default_config("td_ntfields_2d")
    assert config_from_dict(TDNTFieldsConfig, asdict(cfg)) == cfg
