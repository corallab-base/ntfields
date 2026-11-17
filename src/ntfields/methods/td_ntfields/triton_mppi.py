# Triton reads kernel parameter annotations (tl.constexpr), so the jit functions' other parameters stay unannotated
# ruff: noqa: ANN001, ANN202
# pyright doesn't model Triton's language (constexpr parameters, tensor indexing, launch options)
# pyright: reportArgumentType=false, reportCallIssue=false, reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false, reportReturnType=false
"""TD-NTFields' MPPI iterations as one persistent Triton kernel (the "triton" accel backend's planner).

At MPPI's batch (2·samples queries per iteration) the model is latency bound: the folded model through cuBLAS is about
45 kernels per iteration, each a few µs however small. Here one launch runs `chunk` whole iterations:

    per iteration: rollout + Fourier features → gates → (lin1 → lin2 → lin3) per block → final → head → update

Every CTA owns a tile (rows by columns) of each layer's output, keeps its own tiles of u, v and the block input in
registers, and meets the others at a grid barrier (an atomic counter, ~0.5 µs) where a layer needs whole rows. All
CTAs must be resident at once, so the grid is kept within the SM count. The softmax and the update (100 values) are
computed redundantly by every CTA, so the state needs no barrier.

The noise is drawn by torch before the launch, in the eager loop's order, so the random numbers are the eager loop's.
Matmuls are IEEE float32 (no TF32): values match the folded model's to ~1e-7, not bit for bit, so paths match the
eager loop's for a while and then drift apart, as any reordering of the sums makes them. (3xTF32 and 3xFP16 tensor-
core matmuls were no faster: at 16 x 32 tiles a layer is bound by L2 latency, ~4.5 µs, not arithmetic.)
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from ntfields.methods.td_ntfields.inference import INSTANCE_NORM_EPS, FoldedEmbedding

BM, BK = 16, 32
NUM_WARPS = 4


@triton.jit
def _barrier(cnt_ptr, target) -> None:
    tl.debug_barrier()
    tl.atomic_add(cnt_ptr, 1, sem="release")
    c = tl.atomic_add(cnt_ptr, 0, sem="acquire")
    while c < target:
        c = tl.atomic_add(cnt_ptr, 0, sem="acquire")
    tl.debug_barrier()


@triton.jit
def _rows_dot(x_ptr, rm, row, w_ptr, rn, H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """x[rm, :] @ Wᵀ[:, rn], x (rows, H) written by other CTAs (so read through L2), Wᵀ (H, H) stored transposed."""
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, H, BK, num_stages=3):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + rm[:, None] * H + rk[None, :], mask=row[:, None], other=0.0, cache_modifier=".cg")
        w = tl.load(w_ptr + rk[:, None] * H + rn[None, :])
        acc += tl.dot(a, w, input_precision="ieee")
    return acc


@triton.jit
def _clipped_steps(n1, n2, prior, step, gain, AXIS: tl.constexpr):
    """The eager loop's steps from its two noise draws: (step·n1 + step·n2) + gain·prior, clipped to length step."""
    st = (step * n1) + step * n2 + gain * prior
    norm = tl.sqrt(tl.sum(st * st, axis=AXIS, keep_dims=True))
    return st / (tl.maximum(norm, step) / step)


@triton.jit
def _mppi_kernel(
    # state (in: start; out: the state after the chunk), goal, goal embedding, noise (K, S, dim) and (K, S, H, dim)
    q_ptr, prior_ptr, goal_ptr, e_g_ptr, n1_ptr, n2_ptr,
    # weights: Fourier b (F, dim), gates (H, H) twice, blocks (NBLK, 3, H, H), mix (NBLK,), final (H, H)
    b_ptr, wu_ptr, bu_ptr, wv_ptr, bv_ptr, wb_ptr, bb_ptr, mix_ptr, wf_ptr, bf_ptr,
    # scratch: activations (ROWS, H) three times, final layer (ROWS, H), travel times (ROWS,), barrier counter
    xa_ptr, xb_ptr, xc_ptr, y_ptr, t_ptr, cnt_ptr,
    # outputs: the state after each iteration (K, dim), whether it reached the goal (K,)
    visited_ptr, reached_ptr,
    S, step, gain, first_weight, mppi_temperature, goal_tol, head_temperature, out_scale, eps,
    K: tl.constexpr, DIM: tl.constexpr, DP: tl.constexpr, HZ: tl.constexpr, HP: tl.constexpr, SP: tl.constexpr,
    F: tl.constexpr, H: tl.constexpr, NBLK: tl.constexpr, GROUP: tl.constexpr, NCOL: tl.constexpr,
    NPROG: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
) -> None:  # fmt: skip
    pid = tl.program_id(0)
    rm = (pid // NCOL) * BM + tl.arange(0, BM)
    rn = (pid % NCOL) * BN + tl.arange(0, BN)
    row = rm < 2 * S
    d = tl.arange(0, DP)
    dm = d < DIM
    hz = tl.arange(0, HP)
    hm = hz < HZ
    sv = tl.arange(0, SP)
    sm = sv < S
    q = tl.load(q_ptr + d, mask=dm, other=0.0)
    prior = tl.load(prior_ptr + d, mask=dm, other=0.0)
    goal = tl.load(goal_ptr + d, mask=dm, other=0.0)
    n_bar = 0
    for it in range(K):
        n1_it = n1_ptr + it * S * DIM
        n2_it = n2_ptr + it * S * HZ * DIM
        # rollout: row 2s is sample s's first state, row 2s + 1 its last
        s = rm // 2
        m3 = row[:, None, None] & hm[None, :, None] & dm[None, None, :]
        n1 = tl.load(n1_it + s[:, None] * DIM + d[None, :], mask=row[:, None] & dm[None, :], other=0.0)
        n2 = tl.load(
            n2_it + s[:, None, None] * (HZ * DIM) + hz[None, :, None] * DIM + d[None, None, :], mask=m3, other=0.0
        )
        st = tl.where(m3, _clipped_steps(n1[:, None, :], n2, prior[None, None, :], step, gain, 2), 0.0)
        c = q[None, None, :] + tl.cumsum(st, axis=1)
        pick = tl.where(rm % 2 == 1, HZ - 1, 0)
        ends = tl.sum(tl.where(hz[None, :, None] == pick[:, None, None], c, 0.0), axis=1)  # (BM, DP)
        # Fourier features (the gates' and the first block's input)
        bt = tl.load(b_ptr + (rn % F)[:, None] * DIM + d[None, :], mask=dm[None, :], other=0.0)
        proj = tl.sum(ends[:, None, :] * bt[None, :, :], axis=2)
        x_in = tl.where(rn[None, :] < F, tl.sin(proj), tl.cos(proj))
        tl.store(xa_ptr + rm[:, None] * H + rn[None, :], x_in, mask=row[:, None])
        n_bar += 1
        _barrier(cnt_ptr, n_bar * NPROG)
        u = tl.sin(_rows_dot(xa_ptr, rm, row, wu_ptr, rn, H, BM, BN, BK) + tl.load(bu_ptr + rn)[None, :])
        v = tl.sin(_rows_dot(xa_ptr, rm, row, wv_ptr, rn, H, BM, BN, BK) + tl.load(bv_ptr + rn)[None, :])
        uv = u - v
        for blk in tl.static_range(NBLK):
            w1 = wb_ptr + (blk * 3 + 0) * H * H
            w2 = wb_ptr + (blk * 3 + 1) * H * H
            w3 = wb_ptr + (blk * 3 + 2) * H * H
            z = _rows_dot(xa_ptr, rm, row, w1, rn, H, BM, BN, BK) + tl.load(bb_ptr + (blk * 3 + 0) * H + rn)[None, :]
            tl.store(xb_ptr + rm[:, None] * H + rn[None, :], v + uv * tl.sin(z), mask=row[:, None])
            n_bar += 1
            _barrier(cnt_ptr, n_bar * NPROG)
            z = _rows_dot(xb_ptr, rm, row, w2, rn, H, BM, BN, BK) + tl.load(bb_ptr + (blk * 3 + 1) * H + rn)[None, :]
            tl.store(xc_ptr + rm[:, None] * H + rn[None, :], v + uv * tl.sin(z), mask=row[:, None])
            n_bar += 1
            _barrier(cnt_ptr, n_bar * NPROG)
            z = _rows_dot(xc_ptr, rm, row, w3, rn, H, BM, BN, BK) + tl.load(bb_ptr + (blk * 3 + 2) * H + rn)[None, :]
            mix = tl.load(mix_ptr + blk)
            x_in = (1.0 - mix) * x_in + mix * tl.sin(z)
            tl.store(xa_ptr + rm[:, None] * H + rn[None, :], x_in, mask=row[:, None])
            n_bar += 1
            _barrier(cnt_ptr, n_bar * NPROG)
        y = _rows_dot(xa_ptr, rm, row, wf_ptr, rn, H, BM, BN, BK) + tl.load(bf_ptr + rn)[None, :]
        tl.store(y_ptr + rm[:, None] * H + rn[None, :], y, mask=row[:, None])
        n_bar += 1
        _barrier(cnt_ptr, n_bar * NPROG)
        # head, for whole rows in chunks of two groups: instance norm, then out_scale · Σ_groups smoothmax |e - e_g|
        if pid % NCOL == 0:
            total = tl.zeros((BM,), dtype=tl.float32)
            for c0 in range(0, H, 2 * GROUP):
                rc = c0 + tl.arange(0, 2 * GROUP)
                total += tl.sum(
                    tl.load(y_ptr + rm[:, None] * H + rc[None, :], mask=row[:, None], other=0.0, cache_modifier=".cg"),
                    axis=1,
                )
            mean = total / H
            sq = tl.zeros((BM,), dtype=tl.float32)
            for c0 in range(0, H, 2 * GROUP):
                rc = c0 + tl.arange(0, 2 * GROUP)
                cen = (
                    tl.load(y_ptr + rm[:, None] * H + rc[None, :], mask=row[:, None], other=0.0, cache_modifier=".cg")
                    - mean[:, None]
                )
                sq += tl.sum(cen * cen, axis=1)
            r = tl.rsqrt(sq / H + eps)
            t = tl.zeros((BM,), dtype=tl.float32)
            for c0 in range(0, H, 2 * GROUP):
                rc = c0 + tl.arange(0, 2 * GROUP)
                yc = tl.load(y_ptr + rm[:, None] * H + rc[None, :], mask=row[:, None], other=0.0, cache_modifier=".cg")
                diff = (yc - mean[:, None]) * r[:, None] - tl.load(e_g_ptr + rc)[None, :]
                dg = tl.reshape(tl.sqrt(diff * diff + 1e-6) * head_temperature, (BM, 2, GROUP))
                peak = tl.max(dg, axis=2)
                lse = tl.log(tl.sum(tl.exp(dg - peak[:, :, None]), axis=2)) + peak
                t += tl.sum((lse - math.log(GROUP)) / head_temperature, axis=1)
            tl.store(t_ptr + rm, out_scale * t, mask=row)
        n_bar += 1
        _barrier(cnt_ptr, n_bar * NPROG)
        # update, in every CTA: softmax(-temperature·(first_weight·T(first) + T(last)))-weighted first steps
        t0 = tl.load(t_ptr + 2 * sv, mask=sm, other=0.0, cache_modifier=".cg")
        t1 = tl.load(t_ptr + 2 * sv + 1, mask=sm, other=0.0, cache_modifier=".cg")
        logits = tl.where(sm, -mppi_temperature * (first_weight * t0 + t1), -float("inf"))
        w = tl.exp(logits - tl.max(logits, axis=0))
        w = w / tl.sum(w, axis=0)
        m2 = sm[:, None] & dm[None, :]
        a1 = tl.load(n1_it + sv[:, None] * DIM + d[None, :], mask=m2, other=0.0)
        a2 = tl.load(n2_it + sv[:, None] * (HZ * DIM) + d[None, :], mask=m2, other=0.0)
        first = tl.where(m2, _clipped_steps(a1, a2, prior[None, :], step, gain, 1), 0.0)
        prior = tl.sum(w[:, None] * first, axis=0)
        q = q + prior
        if pid == 0:
            tl.store(visited_ptr + it * DIM + d, q, mask=dm)
            gap = tl.where(dm, goal - q, 0.0)
            tl.store(reached_ptr + it, (tl.sqrt(tl.sum(gap * gap, axis=0)) < goal_tol).to(tl.int8))
    if pid == 0:
        tl.store(q_ptr + d, q, mask=dm)
        tl.store(prior_ptr + d, prior, mask=dm)


class FusedMPPIIterations:
    """`chunk` MPPI iterations of a TD-NTFields model as torch's two noise draws per iteration plus one launch of
    `_mppi_kernel`, on static buffers (as mppi's CUDA graph capture needs)."""

    def __init__(
        self,
        embed: FoldedEmbedding,
        group: int,
        temperature: float,
        out_scale: float,
        start: Tensor,
        e_g: Tensor,
        chunk: int,
        samples: int,
        horizon: int,
        step: float,
        prior_gain: float,
        first_weight: float,
        mppi_temperature: float,
        goal_tol: float,
    ) -> None:
        f, dim = embed.b.shape
        h = embed.final.weight.shape[0]
        if start.dtype != torch.float32 or embed.b.dtype != torch.float32 or not start.is_cuda:
            raise ValueError("the fused MPPI runs float32 on CUDA")
        if 2 * f != h or h != triton.next_power_of_2(h) or h % (2 * group) or dim > 16:
            raise ValueError(f"the fused MPPI needs 2·fourier = hidden, a power of 2 (got {2 * f}, {h}) and dim <= 16")
        dev = start.device
        n_rows = triton.cdiv(2 * samples, BM)
        self.chunk, self.samples, self.horizon, self.dim, self.h, self.f = chunk, samples, horizon, dim, h, f
        self.group, self.temperature, self.out_scale = group, temperature, out_scale
        self.step, self.prior_gain, self.first_weight = step, prior_gain, first_weight
        self.mppi_temperature, self.goal_tol = mppi_temperature, goal_tol
        blocks = embed.blocks()
        self.w = {
            "b": embed.b,
            "wu": embed.gate_u.weight.T, "bu": embed.gate_u.bias, "wv": embed.gate_v.weight.T, "bv": embed.gate_v.bias,
            "wb": torch.stack([torch.stack([lin.weight.T for lin in layers]) for layers in blocks]),
            "bb": torch.stack([torch.stack([lin.bias for lin in layers]) for layers in blocks]),
            "mix": embed.mix.float(), "wf": embed.final.weight.T, "bf": embed.final.bias,
        }  # fmt: skip
        self.w = {k: v.detach().contiguous() for k, v in self.w.items()}
        self.n_blocks = len(blocks)
        rows = n_rows * BM
        self.q, self.prior, self.goal = start.clone(), torch.zeros_like(start), start.clone()
        self.e_g = e_g.detach().clone().contiguous()
        self.n1 = torch.empty(chunk, samples, dim, device=dev)
        self.n2 = torch.empty(chunk, samples, horizon, dim, device=dev)
        self.xa, self.xb, self.xc, self.y = (torch.zeros(rows, h, device=dev) for _ in range(4))
        self.t = torch.zeros(rows, device=dev)
        self.cnt = torch.zeros(1, dtype=torch.int32, device=dev)
        self.visited = torch.zeros(chunk, dim, device=dev)
        self.reached = torch.zeros(chunk, dtype=torch.int8, device=dev)  # 1: reached the goal
        # the narrowest column tiles (most CTAs) for which the whole grid is resident at once, as the grid barriers
        # need (else they deadlock); bn = 32 on an RTX 4070 (46 SMs, 2 CTAs each)
        props = torch.cuda.get_device_properties(dev)
        for bn in (32, 64, 128, 256):
            if bn > h:
                break
            self.bn, self.ncol, self.nprog = bn, h // bn, n_rows * (h // bn)
            kernel = _mppi_kernel.warmup(*self._args(), grid=(self.nprog,), num_warps=NUM_WARPS)
            kernel._init_handles()  # loads the binary, which sets n_regs
            if self.nprog <= props.multi_processor_count * _ctas_per_sm(kernel.n_regs, kernel.metadata.shared, props):
                return
        raise ValueError(f"the fused MPPI's {2 * samples} queries cannot be resident on {props.name} at once")

    def _args(self) -> tuple[object, ...]:
        w = self.w
        return (
            self.q, self.prior, self.goal, self.e_g, self.n1, self.n2,
            w["b"], w["wu"], w["bu"], w["wv"], w["bv"], w["wb"], w["bb"], w["mix"], w["wf"], w["bf"],
            self.xa, self.xb, self.xc, self.y, self.t, self.cnt, self.visited, self.reached,
            self.samples, self.step, self.prior_gain, self.first_weight, self.mppi_temperature, self.goal_tol,
            self.temperature, self.out_scale, INSTANCE_NORM_EPS,
            self.chunk, self.dim, triton.next_power_of_2(self.dim), self.horizon, triton.next_power_of_2(self.horizon),
            triton.next_power_of_2(self.samples), self.f, self.h, self.n_blocks, self.group, self.ncol, self.nprog,
            BM, self.bn, BK,
        )  # fmt: skip

    def __call__(self) -> None:
        for i in range(self.chunk):  # the eager loop's draws: (S, 1, dim) then (S, H, dim), per iteration
            self.n1[i].normal_()
            self.n2[i].normal_()
        self.cnt.zero_()
        _mppi_kernel[(self.nprog,)](*self._args(), num_warps=NUM_WARPS)


def _ctas_per_sm(n_regs: int, shared: int, props: "torch._C._CudaDeviceProperties") -> int:
    """How many CTAs of a kernel fit on one SM at once (registers are allocated per warp, in units of 256)."""
    regs_per_warp = -(-n_regs * 32 // 256) * 256
    return min(
        props.regs_per_multiprocessor // (regs_per_warp * NUM_WARPS),  # pyright: ignore[reportAttributeAccessIssue]
        props.shared_memory_per_multiprocessor // max(shared + 1024, 1),  # 1 KB per CTA reserved by the driver
        props.max_threads_per_multi_processor // (32 * NUM_WARPS),
        16,
    )
