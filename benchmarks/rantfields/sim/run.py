"""Closed-loop runs of `GeodesicTauController` in MuJoCo, and the metrics the paper reports."""

# mujoco ships no type stubs
# pyright: reportAttributeAccessIssue=false
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from ntfields.control import GeodesicTauController
from ntfields.envs import ArmEnv


@dataclass
class SimLog:
    time_s: np.ndarray  # (T,)
    qpos: np.ndarray  # (T, n) joint angles
    qvel: np.ndarray
    torque: np.ndarray  # commanded, after clipping
    time_to_goal: float | None  # None: not reached within the time limit
    query_times_s: list[float]


def simulate(
    model: mujoco.MjModel, active: tuple[str, ...], controller: GeodesicTauController, q0: np.ndarray, dt: float,
    max_time_s: float,
) -> SimLog:  # fmt: skip
    """Step from rest at `q0` until the controller reports convergence (checked before and after every step) or
    `max_time_s` passes."""
    from sim.ur5e import addresses

    model.opt.timestep = dt
    data = mujoco.MjData(model)
    qpos_adr, dof_adr, act, tau_max = addresses(model, active)
    data.qpos[qpos_adr] = q0
    mujoco.mj_forward(model, data)
    t_log, q_log, v_log, tau_log = [], [], [], []
    reached = None
    for _ in range(int(np.ceil(max_time_s / dt))):
        q, qdot = data.qpos[qpos_adr].copy(), data.qvel[dof_adr].copy()
        if controller.is_converged(q, qdot):
            reached = data.time
            break
        tau = np.clip(controller.compute_torque(q, qdot), -tau_max, tau_max)
        data.ctrl[:] = 0.0
        data.ctrl[act] = tau
        mujoco.mj_step(model, data)
        t_log.append(data.time)
        q_log.append(q)
        v_log.append(qdot)
        tau_log.append(tau)
        if controller.is_converged(data.qpos[qpos_adr], data.qvel[dof_adr]):
            reached = data.time
            break
    return SimLog(np.array(t_log), np.array(q_log), np.array(v_log), np.array(tau_log), reached,
                  list(controller.query_times_s))  # fmt: skip


def summarize(log: SimLog, env: ArmEnv, check_dt: float = 0.02) -> dict[str, float | bool | None]:
    """Time to goal, traced joint-space length, peak |τ|, mean control rate (1 / mean query time), and whether the
    traced motion stays clear of the environment (self-collision is not checked: the paper's arm data has none),
    sampled every `check_dt` seconds."""
    stride = max(1, round(check_dt / float(log.time_s[1] - log.time_s[0]))) if len(log.time_s) > 1 else 1
    q = torch.as_tensor(log.qpos[::stride] / np.asarray(env.joint_scale), dtype=torch.float64)
    clear = bool((env.distance(q) > 0).all()) if len(q) else True
    return {
        "converged": log.time_to_goal is not None,
        "collision_free": clear,
        "time_to_goal": log.time_to_goal,
        "traced_length": float(np.linalg.norm(np.diff(log.qpos, axis=0), axis=1).sum()) if len(log.qpos) > 1 else 0.0,
        "peak_torque": float(np.abs(log.torque).max()) if len(log.torque) else 0.0,
        "control_rate_hz": float(1 / np.mean(log.query_times_s)) if log.query_times_s else None,
    }
