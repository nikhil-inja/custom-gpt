#!/usr/bin/env python3
"""Generate text from a Shakespeare GPT checkpoint."""

from __future__ import annotations

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.shakespeare import CharVocab
from model.gpt_train import GPT


def parse_args():
    p = argparse.ArgumentParser(description="Generate from a trained GPT checkpoint")
    p.add_argument(
        "--ckpt",
        default=os.path.join(REPO_ROOT, "checkpoints", "shakespeare_best.pt"),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--prompt", default="")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vocab = CharVocab(stoi=ckpt["stoi"], itos={int(k): v for k, v in ckpt["itos"].items()})

    model = GPT(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        model_dim=cfg["model_dim"],
        num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"],
        dropout=0.0,
        attn_backend=cfg.get("attn_backend", "torch"),
        norm_backend=cfg.get("norm_backend", "torch"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.prompt:
        start = vocab.encode(args.prompt).unsqueeze(0).to(device)
    else:
        start = torch.zeros((1, 1), dtype=torch.long, device=device)

    out = model.generate(
        start,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(vocab.decode(out[0]))


if __name__ == "__main__":
    main()
