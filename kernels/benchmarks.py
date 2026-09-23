"""Benchmark Triton kernels vs naive PyTorch attention / LayerNorm / Softmax.

Kernel latency uses ``triton.testing.do_bench`` (CUDA-event timing).
Peak memory and train A/B remain custom.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List

import torch
import triton.testing

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_ms(fn: Callable[[], None], warmup: int = 25, rep: int = 100) -> float:
    """Median GPU kernel/op time in milliseconds via Triton's CUDA-event bench."""
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))


def peak_mem_mb(fn: Callable[[], None]) -> float:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    _sync()
    fn()
    _sync()
    return torch.cuda.max_memory_allocated() / (1024**2)


def benchmark_attention(seq_lens: List[int], batch: int = 4, heads: int = 4, dim: int = 64) -> List[Dict]:
    from kernels.attention import flash_attention, naive_attention

    rows = []
    for t in seq_lens:
        q = torch.randn(batch, heads, t, dim, device="cuda")
        k = torch.randn(batch, heads, t, dim, device="cuda")
        v = torch.randn(batch, heads, t, dim, device="cuda")

        def run_naive():
            naive_attention(q, k, v, causal=True)

        def run_flash():
            flash_attention(q, k, v, causal=True)

        with torch.no_grad():
            err = (flash_attention(q, k, v, True) - naive_attention(q, k, v, True)).abs().max().item()

        ms_naive = bench_ms(run_naive)
        ms_flash = bench_ms(run_flash)
        mem_naive = peak_mem_mb(run_naive)
        mem_flash = peak_mem_mb(run_flash)
        speedup = ms_naive / ms_flash if ms_flash > 0 else float("inf")
        mem_reduction = 100.0 * (1.0 - mem_flash / mem_naive) if mem_naive > 0 else 0.0
        rows.append(
            {
                "T": t,
                "naive_ms": ms_naive,
                "flash_ms": ms_flash,
                "latency_reduction_pct": 100.0 * (1.0 - ms_flash / ms_naive),
                "speedup": speedup,
                "naive_mem_mb": mem_naive,
                "flash_mem_mb": mem_flash,
                "mem_reduction_pct": mem_reduction,
                "max_abs_err": err,
            }
        )
    return rows


def benchmark_layernorm(rows: int = 4096, cols: int = 768) -> Dict:
    from kernels.layernorm import triton_layer_norm

    x = torch.randn(rows, cols, device="cuda")
    w = torch.ones(cols, device="cuda")
    b = torch.zeros(cols, device="cuda")

    def run_torch():
        torch.nn.functional.layer_norm(x, (cols,), w, b, 1e-5)

    def run_triton():
        triton_layer_norm(x, w, b, 1e-5)

    ms_t = bench_ms(run_torch)
    ms_k = bench_ms(run_triton)
    return {
        "torch_ms": ms_t,
        "triton_ms": ms_k,
        "speedup_pct": 100.0 * (ms_t / ms_k - 1.0),
        "latency_reduction_pct": 100.0 * (1.0 - ms_k / ms_t),
    }


def benchmark_softmax(rows: int = 4096, cols: int = 1024) -> Dict:
    from kernels.softmax import triton_softmax

    x = torch.randn(rows, cols, device="cuda")

    def run_torch():
        torch.softmax(x, dim=-1)

    def run_triton():
        triton_softmax(x, dim=-1)

    ms_t = bench_ms(run_torch)
    ms_k = bench_ms(run_triton)
    return {
        "torch_ms": ms_t,
        "triton_ms": ms_k,
        "latency_reduction_pct": 100.0 * (1.0 - ms_k / ms_t),
    }


def ab_train_loss(steps: int = 50) -> Dict:
    """Compare torch vs triton backend train loss over a few steps (same init/data)."""
    from data.shakespeare import get_batch, load_shakespeare
    from model.gpt_train import GPT

    train_data, _, vocab = load_shakespeare(os.path.join(REPO_ROOT, "data", "datasets"))
    cfg = dict(
        vocab_size=vocab.size,
        context_length=128,
        model_dim=128,
        num_blocks=4,
        num_heads=4,
        dropout=0.0,
    )
    torch.manual_seed(0)
    m_torch = GPT(**cfg, attn_backend="torch", norm_backend="torch").cuda()
    torch.manual_seed(0)
    m_triton = GPT(**cfg, attn_backend="triton", norm_backend="triton").cuda()
    m_triton.load_state_dict(m_torch.state_dict())

    opt_t = torch.optim.AdamW(m_torch.parameters(), lr=3e-4)
    opt_k = torch.optim.AdamW(m_triton.parameters(), lr=3e-4)

    losses_t, losses_k = [], []
    for step in range(steps):
        torch.manual_seed(1000 + step)
        x, y = get_batch(train_data, 128, 32, torch.device("cuda"))
        _, lt = m_torch(x, y)
        opt_t.zero_grad(set_to_none=True)
        lt.backward()
        opt_t.step()
        losses_t.append(lt.item())

        torch.manual_seed(1000 + step)
        x, y = get_batch(train_data, 128, 32, torch.device("cuda"))
        _, lk = m_triton(x, y)
        opt_k.zero_grad(set_to_none=True)
        lk.backward()
        opt_k.step()
        losses_k.append(lk.item())

    x, y = get_batch(train_data, 128, 32, torch.device("cuda"))

    def step_torch():
        opt_t.zero_grad(set_to_none=True)
        _, loss = m_torch(x, y)
        loss.backward()
        opt_t.step()

    def step_triton():
        opt_k.zero_grad(set_to_none=True)
        _, loss = m_triton(x, y)
        loss.backward()
        opt_k.step()

    # Train-step timing: still CUDA-event timed via do_bench (fwd+bwd+opt).
    ms_t = bench_ms(step_torch, warmup=10, rep=50)
    ms_k = bench_ms(step_triton, warmup=10, rep=50)
    return {
        "final_loss_torch": losses_t[-1],
        "final_loss_triton": losses_k[-1],
        "max_abs_loss_diff": max(abs(a - b) for a, b in zip(losses_t, losses_k)),
        "step_ms_torch": ms_t,
        "step_ms_triton": ms_k,
        "throughput_improvement_pct": 100.0 * (ms_t / ms_k - 1.0),
    }


def format_report(attn_rows, ln, sm, ab) -> str:
    lines = [
        "# Kernel Benchmark Results",
        "",
        "Latency via `triton.testing.do_bench` (CUDA events, median). "
        "Peak memory via `torch.cuda.max_memory_allocated`. "
        "Educational Triton kernels vs naive PyTorch baselines.",
        "",
        "## Causal attention (Flash-style vs matmul+softmax)",
        "",
        "| T | naive ms | flash ms | latency ↓ % | mem naive MB | mem flash MB | mem ↓ % | max|err| |",
        "|---|----------|----------|-------------|--------------|--------------|---------|---------|",
    ]
    for r in attn_rows:
        lines.append(
            f"| {r['T']} | {r['naive_ms']:.3f} | {r['flash_ms']:.3f} | "
            f"{r['latency_reduction_pct']:.1f} | {r['naive_mem_mb']:.1f} | "
            f"{r['flash_mem_mb']:.1f} | {r['mem_reduction_pct']:.1f} | {r['max_abs_err']:.2e} |"
        )
    lines += [
        "",
        "## LayerNorm",
        "",
        f"- torch: {ln['torch_ms']:.3f} ms",
        f"- triton: {ln['triton_ms']:.3f} ms",
        f"- latency reduction: {ln['latency_reduction_pct']:.1f}%",
        "",
        "## Softmax",
        "",
        f"- torch: {sm['torch_ms']:.3f} ms",
        f"- triton: {sm['triton_ms']:.3f} ms",
        f"- latency reduction: {sm['latency_reduction_pct']:.1f}%",
        "",
        "## A/B train step (torch vs triton backends)",
        "",
        f"- final loss torch: {ab['final_loss_torch']:.4f}",
        f"- final loss triton: {ab['final_loss_triton']:.4f}",
        f"- max |Δloss| over steps: {ab['max_abs_loss_diff']:.4f}",
        f"- step time torch: {ab['step_ms_torch']:.3f} ms",
        f"- step time triton: {ab['step_ms_triton']:.3f} ms",
        f"- throughput improvement: {ab['throughput_improvement_pct']:.1f}%",
        "",
        "Re-run:",
        "",
        "```bash",
        "python kernels/benchmarks.py --seq 128 256 512",
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "BENCHMARKS.md"))
    parser.add_argument("--skip-ab", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for benchmarks")

    print("Benchmarking attention...")
    attn_rows = benchmark_attention(args.seq)
    print("Benchmarking LayerNorm...")
    ln = benchmark_layernorm()
    print("Benchmarking Softmax...")
    sm = benchmark_softmax()
    if args.skip_ab:
        ab = {
            "final_loss_torch": float("nan"),
            "final_loss_triton": float("nan"),
            "max_abs_loss_diff": float("nan"),
            "step_ms_torch": float("nan"),
            "step_ms_triton": float("nan"),
            "throughput_improvement_pct": float("nan"),
        }
    else:
        print("A/B train loss...")
        ab = ab_train_loss()

    report = format_report(attn_rows, ln, sm, ab)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
