"""Flash-style fused causal attention in Triton (forward + backward)."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    num_heads,
    seq_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    batch_id = pid_bh // num_heads
    head_id = pid_bh % num_heads

    q_offset = batch_id * stride_qb + head_id * stride_qh
    k_offset = batch_id * stride_kb + head_id * stride_kh
    v_offset = batch_id * stride_vb + head_id * stride_vh
    o_offset = batch_id * stride_ob + head_id * stride_oh
    l_offset = batch_id * stride_lb + head_id * stride_lh

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(
        q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Loop over full sequence; causal masking zeros future keys.
    for start_n in range(0, seq_len, BLOCK_N):
        kn = start_n + offs_n
        k = tl.load(
            k_ptr + k_offset + kn[None, :] * stride_kn + offs_d[:, None],
            mask=(kn[None, :] < seq_len) & (offs_d[:, None] < BLOCK_D),
            other=0.0,
        )
        v = tl.load(
            v_ptr + v_offset + kn[:, None] * stride_vn + offs_d[None, :],
            mask=(kn[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        )

        qk = tl.dot(q, k) * scale
        qk = tl.where(kn[None, :] < seq_len, qk, -float("inf"))
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        # Masked columns contribute 0 after exp(-inf).
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        # Guard rows that are still entirely masked (m stays -inf).
        alpha = tl.where(m_ij > -float("inf"), alpha, 0.0)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # Avoid 0/0 for fully-padded query rows.
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]
    lse = m_i + tl.log(l_safe)

    tl.store(
        o_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
    )
    tl.store(lse_ptr + l_offset + offs_m, lse, mask=offs_m < seq_len)


@triton.jit
def _flash_attn_bwd_preprocess(
    o_ptr,
    do_ptr,
    delta_ptr,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_db,
    stride_dh,
    num_heads,
    seq_len,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    batch_id = pid_bh // num_heads
    head_id = pid_bh % num_heads
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    o = tl.load(
        o_ptr
        + batch_id * stride_ob
        + head_id * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :],
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr
        + batch_id * stride_dob
        + head_id * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(
        delta_ptr + batch_id * stride_db + head_id * stride_dh + offs_m,
        delta,
        mask=offs_m < seq_len,
    )


@triton.jit
def _flash_attn_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_lb,
    stride_lh,
    stride_db,
    stride_dh,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    num_heads,
    seq_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # One program per (batch*head, key-block). Accumulate dK/dV; atomic-add into dQ.
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch_id = pid_bh // num_heads
    head_id = pid_bh % num_heads

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    k = tl.load(
        k_ptr
        + batch_id * stride_kb
        + head_id * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :],
        mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + batch_id * stride_vb
        + head_id * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :],
        mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
        other=0.0,
    )

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    start_m = 0
    if CAUSAL:
        start_m = (start_n // BLOCK_M) * BLOCK_M

    for begin_m in range(start_m, seq_len, BLOCK_M):
        m_idx = begin_m + offs_m
        q = tl.load(
            q_ptr
            + batch_id * stride_qb
            + head_id * stride_qh
            + m_idx[:, None] * stride_qm
            + offs_d[None, :],
            mask=(m_idx[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        )
        do = tl.load(
            do_ptr
            + batch_id * stride_dob
            + head_id * stride_doh
            + m_idx[:, None] * stride_dom
            + offs_d[None, :],
            mask=(m_idx[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + batch_id * stride_lb + head_id * stride_lh + m_idx,
            mask=m_idx < seq_len,
            other=0.0,
        )
        delta = tl.load(
            delta_ptr + batch_id * stride_db + head_id * stride_dh + m_idx,
            mask=m_idx < seq_len,
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * scale
        if CAUSAL:
            qk = tl.where(m_idx[:, None] >= offs_n[None, :], qk, -float("inf"))
        qk = tl.where((m_idx[:, None] < seq_len) & (offs_n[None, :] < seq_len), qk, -float("inf"))
        p = tl.exp(qk - lse[:, None])

        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * scale
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
        dq = tl.dot(ds.to(k.dtype), k)
        # Atomic add into dq (multiple key-blocks write the same query rows).
        tl.atomic_add(
            dq_ptr
            + batch_id * stride_dqb
            + head_id * stride_dqh
            + m_idx[:, None] * stride_dqm
            + offs_d[None, :],
            dq,
            mask=(m_idx[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
        )

    tl.store(
        dk_ptr
        + batch_id * stride_dkb
        + head_id * stride_dkh
        + offs_n[:, None] * stride_dkn
        + offs_d[None, :],
        dk,
        mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
    )
    tl.store(
        dv_ptr
        + batch_id * stride_dvb
        + head_id * stride_dvh
        + offs_n[:, None] * stride_dvn
        + offs_d[None, :],
        dv,
        mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
    )


def _flash_attn_forward(q, k, v, causal: bool = True):
    b, h, t, d = q.shape
    assert d <= 128, "This educational kernel supports head_dim <= 128"
    o = torch.empty_like(q)
    lse = torch.empty(b, h, t, device=q.device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(d)
    grid = (b * h, triton.cdiv(t, BLOCK_M))
    _flash_attn_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        lse.stride(0),
        lse.stride(1),
        h,
        t,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        CAUSAL=causal,
    )
    return o, lse


def _flash_attn_backward(do, q, k, v, o, lse, causal: bool = True):
    b, h, t, d = q.shape
    scale = 1.0 / math.sqrt(d)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v, dtype=torch.float32)
    delta = torch.empty(b, h, t, device=q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(d)
    grid_pre = (b * h, triton.cdiv(t, BLOCK_M))
    _flash_attn_bwd_preprocess[grid_pre](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        delta.stride(0),
        delta.stride(1),
        h,
        t,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )

    # Use float32 q/k/v/do for stable bwd math inside the kernel.
    qf, kf, vf, dof = q.float(), k.float(), v.float(), do.float()
    grid = (b * h, triton.cdiv(t, BLOCK_N))
    _flash_attn_bwd_kernel[grid](
        qf,
        kf,
        vf,
        dof,
        lse,
        delta,
        dq,
        dk,
        dv,
        qf.stride(0),
        qf.stride(1),
        qf.stride(2),
        kf.stride(0),
        kf.stride(1),
        kf.stride(2),
        vf.stride(0),
        vf.stride(1),
        vf.stride(2),
        dof.stride(0),
        dof.stride(1),
        dof.stride(2),
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        h,
        t,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        CAUSAL=causal,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool):
        q_c, k_c, v_c = q.contiguous(), k.contiguous(), v.contiguous()
        o, lse = _flash_attn_forward(q_c, k_c, v_c, causal=causal)
        ctx.save_for_backward(q_c, k_c, v_c, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()
        dq, dk, dv = _flash_attn_backward(do, q, k, v, o, lse, causal=ctx.causal)
        return dq, dk, dv, None


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Fused causal attention. q/k/v: [B, H, T, D], CUDA tensors."""
    if q.device.type != "cuda":
        raise RuntimeError("flash_attention requires CUDA tensors")
    return FlashAttentionFunction.apply(q, k, v, causal)


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Reference attention that materializes the full score matrix."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        t = scores.size(-1)
        mask = torch.ones(t, t, device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
