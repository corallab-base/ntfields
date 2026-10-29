"""Closed-form UR5e inverse kinematics (Hawkins' analytic UR solution, 8 branches per pose).

Port of ntrl-demo's `torch_IK_UR5`, with the DH base frame corrected to the URDF `base_link` (Rz(π)).
"""

import math

import torch
from torch import Tensor

_D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
_A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
_ALPHA = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]
_URDF_TO_DH = torch.tensor([[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def rpy_pose(position: Tensor, roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
    """(N, 4, 4) poses from positions (N, 3) and Rz(yaw)·Ry(pitch)·Rx(roll) angles (N,), as ntrl-demo's `rot_pose`."""
    ca, sa = torch.cos(roll), torch.sin(roll)
    cb, sb = torch.cos(pitch), torch.sin(pitch)
    cc, sc = torch.cos(yaw), torch.sin(yaw)
    t = torch.zeros((position.shape[0], 4, 4), device=position.device, dtype=position.dtype)
    t[:, 0, 0] = cb * cc
    t[:, 0, 1] = sa * sb * cc - ca * sc
    t[:, 0, 2] = ca * sb * cc + sa * sc
    t[:, 1, 0] = cb * sc
    t[:, 1, 1] = sa * sb * sc + ca * cc
    t[:, 1, 2] = ca * sb * sc - sa * cc
    t[:, 2, 0] = -sb
    t[:, 2, 1] = sa * cb
    t[:, 2, 2] = ca * cb
    t[:, :3, 3] = position
    t[:, 3, 3] = 1
    return t


def solve_ur5e_ik(poses: Tensor, joint_limit: float = math.pi) -> Tensor:
    """(N, 8, 6) joint angles (rad, wrapped into ±joint_limit) for (N, 4, 4) wrist_3_link poses in the URDF base
    frame. Unreachable branches come from clamped acos/asin arguments; validate them downstream."""
    device, dtype = poses.device, poses.dtype
    n = poses.shape[0]
    d = [torch.tensor(v, device=device, dtype=dtype) for v in _D]
    a = _A

    def wrap(value: Tensor) -> Tensor:
        v = value.clone()
        while torch.any(v > joint_limit):
            v = torch.where(v > joint_limit, v - 2 * math.pi, v)
        while torch.any(v < -joint_limit):
            v = torch.where(v < -joint_limit, v + 2 * math.pi, v)
        return v

    zero = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
    d5_vec = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
    d5_vec[2] = -d[5]
    d3_vec = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)
    d3_vec[1] = -d[3]
    gd = _URDF_TO_DH.to(device=device, dtype=dtype) @ poses

    # theta1: 2 branches
    p05 = ((gd @ d5_vec.unsqueeze(1)).squeeze(-1) - zero)[:, :3]
    psi = torch.atan2(p05[:, 1], p05[:, 0])
    length = torch.sqrt(p05[:, 0] ** 2 + p05[:, 1] ** 2)
    length = torch.where(torch.abs(d[3]) > length, torch.abs(d[3]), length)
    phi = torch.acos(torch.clamp(d[3] / length, -1.0, 1.0))
    theta1 = torch.stack([wrap(psi + phi + math.pi / 2), wrap(psi - phi + math.pi / 2)], dim=1)

    # theta5: 2 x 2 branches
    p06 = gd[:, :3, 3]
    theta5 = torch.zeros((n, 2, 2), device=device, dtype=dtype)
    for i in range(2):
        p16z = p06[:, 0] * torch.sin(theta1[:, i]) - p06[:, 1] * torch.cos(theta1[:, i])
        denom = torch.abs(p16z - d[3])
        l5 = torch.where(denom > d[5], denom, d[5] * torch.ones_like(denom))
        theta5i = torch.acos(torch.clamp((p16z - d[3]) / l5, -1.0, 1.0))
        theta5[:, i, 0] = theta5i
        theta5[:, i, 1] = -theta5i

    # theta6: 2 x 2 branches
    theta6 = torch.zeros((n, 2, 2), device=device, dtype=dtype)
    for i in range(2):
        t61 = _inv(_inv(_dh(a[0], d[0], _ALPHA[0], theta1[:, i])) @ gd)
        for j in range(2):
            sin5 = torch.sin(theta5[:, i, j])
            safe = torch.where(sin5 == 0, torch.ones_like(sin5), sin5)
            solved = torch.atan2(-t61[:, 1, 2] / safe, t61[:, 0, 2] / safe)
            theta6[:, i, j] = torch.where(sin5 == 0, torch.zeros_like(solved), solved)

    # theta2, theta3: 2 x 2 x 2 branches
    theta2 = torch.zeros((n, 2, 2, 2), device=device, dtype=dtype)
    theta3 = torch.zeros((n, 2, 2, 2), device=device, dtype=dtype)
    for i in range(2):
        t16 = _inv(_dh(a[0], d[0], _ALPHA[0], theta1[:, i])) @ gd
        for j in range(2):
            t45 = _dh(a[4], d[4], _ALPHA[4], theta5[:, i, j])
            t56 = _dh(a[5], d[5], _ALPHA[5], theta6[:, i, j])
            p13 = ((t16 @ _inv(t45 @ t56) @ d3_vec.unsqueeze(1)).squeeze(-1) - zero)[:, :3]
            ratio = (torch.norm(p13, dim=1) ** 2 - a[1] ** 2 - a[2] ** 2) / (2 * a[1] * a[2])
            ratio = torch.where(torch.abs(ratio) > 1, torch.sign(ratio), ratio)
            theta3_0 = torch.acos(torch.clamp(ratio, -1.0, 1.0))
            base = -torch.atan2(p13[:, 1], -p13[:, 0])
            p13_norm = torch.norm(p13, dim=1)
            theta3[:, i, j, 0] = theta3_0
            theta2[:, i, j, 0] = base + torch.asin(torch.clamp(a[2] * torch.sin(theta3_0) / p13_norm, -1.0, 1.0))
            theta3[:, i, j, 1] = -theta3_0
            theta2[:, i, j, 1] = base + torch.asin(torch.clamp(a[2] * torch.sin(-theta3_0) / p13_norm, -1.0, 1.0))

    # theta4: 2 x 2 x 2 branches
    theta4 = torch.zeros((n, 2, 2, 2), device=device, dtype=dtype)
    for i in range(2):
        t16 = _inv(_dh(a[0], d[0], _ALPHA[0], theta1[:, i])) @ gd
        for j in range(2):
            t45 = _dh(a[4], d[4], _ALPHA[4], theta5[:, i, j])
            t56 = _dh(a[5], d[5], _ALPHA[5], theta6[:, i, j])
            t14 = t16 @ _inv(t45 @ t56)
            for k in range(2):
                t13 = _dh(a[1], d[1], _ALPHA[1], theta2[:, i, j, k]) @ _dh(a[2], d[2], _ALPHA[2], theta3[:, i, j, k])
                t34 = _inv(t13) @ t14
                theta4[:, i, j, k] = torch.atan2(t34[:, 1, 0], t34[:, 0, 0])

    q = torch.zeros((n, 8, 6), device=device, dtype=dtype)
    index = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                q[:, index, 0] = wrap(theta1[:, i])
                q[:, index, 1] = wrap(theta2[:, i, j, k])
                q[:, index, 2] = wrap(theta3[:, i, j, k])
                q[:, index, 3] = wrap(theta4[:, i, j, k])
                q[:, index, 4] = wrap(theta5[:, i, j])
                q[:, index, 5] = wrap(theta6[:, i, j])
                index += 1
    return q


def _dh(a: float, d: float | Tensor, alpha: float, theta: Tensor) -> Tensor:
    """(N, 4, 4) DH transforms for joint angles theta (N,)."""
    t = torch.zeros((theta.shape[0], 4, 4), device=theta.device, dtype=theta.dtype)
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    t[:, 0, 0] = ct
    t[:, 0, 1] = -st * ca
    t[:, 0, 2] = st * sa
    t[:, 0, 3] = ct * a
    t[:, 1, 0] = st
    t[:, 1, 1] = ct * ca
    t[:, 1, 2] = -ct * sa
    t[:, 1, 3] = st * a
    t[:, 2, 1] = sa
    t[:, 2, 2] = ca
    t[:, 2, 3] = d
    t[:, 3, 3] = 1
    return t


def _inv(t: Tensor) -> Tensor:
    """Inverse of rigid transforms (N, 4, 4)."""
    rt = torch.transpose(t[:, :3, :3], 1, 2)
    out = torch.zeros_like(t)
    out[:, :3, :3] = rt
    out[:, :3, 3] = -(rt @ t[:, :3, 3].unsqueeze(2)).squeeze(2)
    out[:, 3, 3] = 1
    return out
