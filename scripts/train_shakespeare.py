#!/usr/bin/env python3
"""Train the production GPT on tiny Shakespeare."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# Allow running as `python scripts/train_shakespeare.py` from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.shakespeare import get_batch, load_shakespeare
from model.gpt_train import GPT


def parse_args():
    p = argparse.ArgumentParser(description="Train GPT on tiny Shakespeare")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--attn", choices=["torch", "triton"], default="torch")
    p.add_argument("--norm", choices=["torch", "triton"], default="torch")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data", "datasets"))
    p.add_argument("--ckpt-dir", default=os.path.join(REPO_ROOT, "checkpoints"))
    p.add_argument("--sample-tokens", type=int, default=200)
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(args.eval_batches, device=device)
        for i in range(args.eval_batches):
            x, y = get_batch(data, args.block_size, args.batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    if args.attn == "triton" or args.norm == "triton":
        if not torch.cuda.is_available():
            raise SystemExit("Triton backends require CUDA")
        args.device = "cuda"

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_data, val_data, vocab = load_shakespeare(args.data_dir)
    print(f"vocab_size={vocab.size} train_tokens={len(train_data)} val_tokens={len(val_data)}")

    model = GPT(
        vocab_size=vocab.size,
        context_length=args.block_size,
        model_dim=args.n_embd,
        num_blocks=args.n_layer,
        num_heads=args.n_head,
        dropout=args.dropout,
        attn_backend=args.attn,
        norm_backend=args.norm,
    ).to(device)
    print(f"parameters={model.num_parameters():,} attn={args.attn} norm={args.norm} device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    model.train()
    for step in range(1, args.max_steps + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            metrics = estimate_loss(model, train_data, val_data, args, device)
            dt = time.time() - t0
            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"train {metrics['train']:.4f} | val {metrics['val']:.4f} | {dt:.1f}s"
            )
            if metrics["val"] < best_val:
                best_val = metrics["val"]
                ckpt_path = os.path.join(args.ckpt_dir, "shakespeare_best.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": {
                            "vocab_size": vocab.size,
                            "context_length": args.block_size,
                            "model_dim": args.n_embd,
                            "num_blocks": args.n_layer,
                            "num_heads": args.n_head,
                            "dropout": args.dropout,
                            "attn_backend": args.attn,
                            "norm_backend": args.norm,
                        },
                        "itos": vocab.itos,
                        "stoi": vocab.stoi,
                        "val_loss": best_val,
                        "step": step,
                    },
                    ckpt_path,
                )
                print(f"  saved {ckpt_path} (val={best_val:.4f})")

            # Quick sample
            ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
            sample = model.generate(ctx, args.sample_tokens, temperature=0.8, top_k=20)[0].tolist()
            print("--- sample ---")
            print(vocab.decode(sample))
            print("--------------")

    print(f"done. best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
