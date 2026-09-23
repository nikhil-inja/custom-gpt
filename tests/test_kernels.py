"""Correctness tests for Triton kernels vs PyTorch reference."""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@requires_cuda
def test_triton_softmax():
    from kernels.softmax import triton_softmax

    torch.manual_seed(0)
    x = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    y_ref = F.softmax(x, dim=-1)
    y = triton_softmax(x, dim=-1)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-4)


@requires_cuda
def test_triton_layernorm_forward_backward():
    from kernels.layernorm import TritonLayerNorm

    torch.manual_seed(0)
    x = torch.randn(4, 32, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    ln = TritonLayerNorm(64).cuda()
    ln_ref = torch.nn.LayerNorm(64).cuda()
    ln_ref.load_state_dict(ln.state_dict())

    y = ln(x)
    y_ref = ln_ref(x_ref)
    assert torch.allclose(y, y_ref, atol=1e-4, rtol=1e-4)

    grad = torch.randn_like(y)
    y.backward(grad)
    y_ref.backward(grad)
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ln.weight.grad, ln_ref.weight.grad, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ln.bias.grad, ln_ref.bias.grad, atol=1e-3, rtol=1e-3)


@requires_cuda
@pytest.mark.parametrize("t", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention_forward(t, causal):
    from kernels.attention import flash_attention, naive_attention

    torch.manual_seed(0)
    b, h, d = 2, 4, 32
    q = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32)
    k = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32)
    v = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32)
    out = flash_attention(q, k, v, causal=causal)
    ref = naive_attention(q, k, v, causal=causal)
    assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2)


@requires_cuda
def test_flash_attention_backward():
    from kernels.attention import flash_attention, naive_attention

    torch.manual_seed(0)
    b, h, t, d = 2, 4, 64, 32
    q = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(b, h, t, d, device="cuda", dtype=torch.float32, requires_grad=True)
    q2, k2, v2 = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)

    out = flash_attention(q, k, v, causal=True)
    ref = naive_attention(q2, k2, v2, causal=True)
    go = torch.randn_like(out)
    out.backward(go)
    ref.backward(go)

    assert torch.allclose(q.grad, q2.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(k.grad, k2.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(v.grad, v2.grad, atol=2e-2, rtol=2e-2)


@requires_cuda
def test_gpt_triton_matches_torch_loss_step():
    """One training step: triton and torch backends should stay close with tied weights."""
    from model.gpt_train import GPT

    torch.manual_seed(0)
    cfg = dict(
        vocab_size=65,
        context_length=64,
        model_dim=64,
        num_blocks=2,
        num_heads=4,
        dropout=0.0,
    )
    m_torch = GPT(**cfg, attn_backend="torch", norm_backend="torch").cuda()
    m_triton = GPT(**cfg, attn_backend="triton", norm_backend="triton").cuda()
    m_triton.load_state_dict(m_torch.state_dict())

    x = torch.randint(0, 65, (4, 64), device="cuda")
    y = torch.randint(0, 65, (4, 64), device="cuda")
    _, loss_t = m_torch(x, y)
    _, loss_k = m_triton(x, y)
    assert torch.allclose(loss_t, loss_k, atol=5e-2, rtol=5e-2)
