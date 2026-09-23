"""Triton numerically-stable softmax (last dimension)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    row_input = input_ptr + row * input_row_stride
    row_output = output_ptr + row * output_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    values = tl.load(row_input + col_offsets, mask=mask, other=-float("inf"))
    row_max = tl.max(values, axis=0)
    values = values - row_max
    numer = tl.exp(values)
    denom = tl.sum(numer, axis=0)
    out = numer / denom
    tl.store(row_output + col_offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax along `dim` using a Triton kernel. Contiguous last-dim preferred."""
    if dim < 0:
        dim += x.ndim
    if dim != x.ndim - 1:
        # Move target dim to last, run kernel, restore.
        x_perm = x.transpose(dim, -1).contiguous()
        y_perm = triton_softmax(x_perm, dim=-1)
        return y_perm.transpose(dim, -1).contiguous()

    x_c = x.contiguous()
    n_cols = x_c.shape[-1]
    n_rows = x_c.numel() // n_cols
    y = torch.empty_like(x_c)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](
        x_c,
        y,
        n_rows,
        n_cols,
        x_c.stride(-2) if x_c.ndim > 1 else 0,
        y.stride(-2) if y.ndim > 1 else 0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
