"""Production GPT for training — device-safe, fused QKV, no grader artifacts."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


AttnBackend = Literal["torch", "triton"]
NormBackend = Literal["torch", "triton"]


def _causal_attn_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Naive causal attention. q/k/v: [B, H, T, D]."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    t = scores.shape[-1]
    mask = torch.ones(t, t, device=scores.device, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attn_backend: AttnBackend = "torch",
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.attn_backend = attn_backend
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        if self.attn_backend == "triton":
            from kernels.attention import flash_attention

            y = flash_attention(q, k, v, causal=True)
        else:
            y = _causal_attn_torch(q, k, v)

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.attn_dropout(y)
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.ReLU(),
            nn.Linear(4 * model_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_norm(model_dim: int, norm_backend: NormBackend) -> nn.Module:
    if norm_backend == "triton":
        from kernels.layernorm import TritonLayerNorm

        return TritonLayerNorm(model_dim)
    return nn.LayerNorm(model_dim)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.2,
        attn_backend: AttnBackend = "torch",
        norm_backend: NormBackend = "torch",
    ):
        super().__init__()
        self.ln1 = _make_norm(model_dim, norm_backend)
        self.attn = CausalSelfAttention(model_dim, num_heads, dropout=dropout, attn_backend=attn_backend)
        self.ln2 = _make_norm(model_dim, norm_backend)
        self.ffn = FeedForward(model_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        model_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.2,
        attn_backend: AttnBackend = "torch",
        norm_backend: NormBackend = "torch",
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embed = nn.Embedding(vocab_size, model_dim)
        self.pos_embed = nn.Embedding(context_length, model_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim,
                    num_heads,
                    dropout=dropout,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                )
                for _ in range(num_blocks)
            ]
        )
        self.ln_f = _make_norm(model_dim, norm_backend)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = idx.shape
        if t > self.context_length:
            raise ValueError(f"Sequence length {t} exceeds context_length {self.context_length}")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_embed(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
