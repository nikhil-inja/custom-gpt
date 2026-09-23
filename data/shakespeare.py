"""Tiny Shakespeare char-level dataset: download, encode, batch."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@dataclass
class CharVocab:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[int(i)] for i in ids)


def build_char_vocab(text: str) -> CharVocab:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return CharVocab(stoi=stoi, itos=itos)


def download_shakespeare(dest_path: str, url: str = SHAKESPEARE_URL) -> str:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Downloading Shakespeare to {dest_path} ...")
        urllib.request.urlretrieve(url, dest_path)
    with open(dest_path, "r", encoding="utf-8") as f:
        return f.read()


def load_shakespeare(
    data_dir: str = "data/datasets",
    val_fraction: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, CharVocab]:
    path = os.path.join(data_dir, "tinyshakespeare.txt")
    text = download_shakespeare(path)
    vocab = build_char_vocab(text)
    data = vocab.encode(text)
    n = int(len(data) * (1.0 - val_fraction))
    return data[:n], data[n:], vocab


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random next-token windows from a 1D token tensor."""
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(f"Data length {len(data)} too short for block_size {block_size}")
    ix = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
