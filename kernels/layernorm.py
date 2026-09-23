"""Triton LayerNorm with custom autograd (forward + backward)."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    stride_x,
    stride_y,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = x_centered * rstd

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        x_hat = x_hat * w
    if HAS_BIAS:
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x_hat + b

    tl.store(y_ptr + row * stride_y + cols, x_hat, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layernorm_bwd_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    dw_ptr,
    db_ptr,
    stride_dy,
    stride_x,
    stride_dx,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    dy = tl.load(dy_ptr + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    x_hat = (x - mean) * rstd

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        # Accumulate weight/bias grads with atomics across rows.
        tl.atomic_add(dw_ptr + cols, dy * x_hat, mask=mask)
        tl.atomic_add(db_ptr + cols, dy, mask=mask)
        dy = dy * w

    # dx = (1/N) * rstd * (N*dy - sum(dy) - x_hat * sum(dy * x_hat))
    sum_dy = tl.sum(tl.where(mask, dy, 0.0), axis=0)
    sum_dy_xhat = tl.sum(tl.where(mask, dy * x_hat, 0.0), axis=0)
    dx = (dy - sum_dy / n_cols - x_hat * sum_dy_xhat / n_cols) * rstd
    tl.store(dx_ptr + row * stride_dx + cols, dx, mask=mask)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1]).contiguous()
        n_rows, n_cols = x_2d.shape
        y = torch.empty_like(x_2d)
        mean = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        _layernorm_fwd_kernel[(n_rows,)](
            x_2d,
            y,
            weight if weight is not None else x_2d,
            bias if bias is not None else x_2d,
            mean,
            rstd,
            x_2d.stride(0),
            y.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
        )
        ctx.save_for_backward(x_2d, weight, bias, mean, rstd)
        ctx.eps = eps
        ctx.x_shape = x_shape
        return y.view(x_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, bias, mean, rstd = ctx.saved_tensors
        dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()
        n_rows, n_cols = x_2d.shape
        dx = torch.empty_like(x_2d)
        dw = torch.zeros(n_cols, device=x_2d.device, dtype=torch.float32) if weight is not None else None
        db = torch.zeros(n_cols, device=x_2d.device, dtype=torch.float32) if bias is not None else None
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        _layernorm_bwd_kernel[(n_rows,)](
            dy_2d,
            x_2d,
            weight if weight is not None else x_2d,
            mean,
            rstd,
            dx,
            dw if dw is not None else dx,
            db if db is not None else dx,
            dy_2d.stride(0),
            x_2d.stride(0),
            dx.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=weight is not None,
        )

        dw_out = dw.to(weight.dtype) if weight is not None else None
        db_out = db.to(bias.dtype) if bias is not None else None
        return dx.view(ctx.x_shape), dw_out, db_out, None


def triton_layer_norm(x, weight=None, bias=None, eps: float = 1e-5):
    return LayerNormFunction.apply(x, weight, bias, eps)


class TritonLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cuda":
            return torch.nn.functional.layer_norm(
                x, (self.normalized_shape,), self.weight, self.bias, self.eps
            )
        return triton_layer_norm(x, self.weight, self.bias, self.eps)
