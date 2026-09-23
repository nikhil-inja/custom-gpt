# Kernel Benchmark Results

Latency via `triton.testing.do_bench` (CUDA events, median). Peak memory via `torch.cuda.max_memory_allocated`. Educational Triton kernels vs naive PyTorch baselines.

## Causal attention (Flash-style vs matmul+softmax)

| T | naive ms | flash ms | latency ↓ % | mem naive MB | mem flash MB | mem ↓ % | max|err| |
|---|----------|----------|-------------|--------------|--------------|---------|---------|
| 128 | 0.091 | 0.027 | 70.8 | 12.1 | 10.1 | 16.5 | 2.68e-03 |
| 256 | 0.164 | 0.050 | 69.4 | 20.2 | 12.1 | 39.9 | 2.57e-03 |
| 512 | 0.570 | 0.146 | 74.3 | 48.4 | 16.2 | 66.6 | 2.70e-03 |

## LayerNorm

- torch: 0.121 ms
- triton: 0.119 ms
- latency reduction: 1.7%

## Softmax

- torch: 0.152 ms
- triton: 0.151 ms
- latency reduction: 1.0%

## A/B train step (torch vs triton backends)

- final loss torch: 2.8484
- final loss triton: 2.8483
- max |Δloss| over steps: 0.0002
- step time torch: 25.784 ms
- step time triton: 25.655 ms
- throughput improvement: 0.5%

Re-run:

```bash
python kernels/benchmarks.py --seq 128 256 512
```
