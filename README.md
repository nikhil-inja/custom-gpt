# My GPT — Built from Scratch

> Assembled from the NeetCode ML course on [NeetCode.io](https://neetcode.io), then extended with a CUDA training pipeline and custom Triton kernels.

Course `Solution` files are preserved. Training + kernels live in new modules beside them.

## Project Structure

```
model/
  gpt.py                   Course GPT (grader artifacts)
  gpt_train.py             Production GPT (fused QKV, device-safe, triton|torch backends)
data/
  shakespeare.py           Tiny Shakespeare download, char vocab, batching
  ...                      Course data exercises
kernels/
  softmax.py               Triton softmax
  layernorm.py             Triton LayerNorm (fwd + bwd)
  attention.py             Flash-style fused causal attention (fwd + bwd)
  benchmarks.py            Timing / memory / A/B loss
scripts/
  train_shakespeare.py     Train on Shakespeare
  generate.py              Sample from a checkpoint
tests/
  test_kernels.py          Correctness vs PyTorch
foundations/               Course NN primitives
BENCHMARKS.md              Measured kernel results
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA GPU for Triton backends. PyTorch attention training also runs on CPU.

## Train (Shakespeare)

```bash
# Baseline (PyTorch attention + LayerNorm)
python scripts/train_shakespeare.py --max-steps 2000

# Fused Triton attention (recommended Triton path)
python scripts/train_shakespeare.py --attn triton --norm torch --max-steps 2000

# Full Triton attention + LayerNorm
python scripts/train_shakespeare.py --attn triton --norm triton --max-steps 2000
```

Defaults: `n_layer=4`, `n_head=4`, `n_embd=128`, `block_size=128` (~0.82M params). Checkpoints land in `checkpoints/shakespeare_best.pt`.

## Generate

```bash
python scripts/generate.py --ckpt checkpoints/shakespeare_best.pt --prompt "ROMEO:"
```

## Kernels & tests

```bash
pytest tests/test_kernels.py -v
python kernels/benchmarks.py --seq 128 256 512
```

See [BENCHMARKS.md](BENCHMARKS.md) for measured latency/memory vs naive attention.

## Course archive

Original NeetCode entrypoints (`train.py`, `generate.py`, `model/gpt.py`, …) remain as submitted course solutions and are not used by the training scripts above.
