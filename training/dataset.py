"""
Tokenized-corpus storage and batching for pretraining.

Token ids are stored as a flat binary file of uint16 (fine as long as
vocab_size < 65536, true for every NeuraQuill config so far) and accessed via
np.memmap, so a corpus far larger than RAM can still be trained on --
only the sampled blocks are ever actually read into memory.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def tokenize_and_save(tokenizer, text: str, out_path: str) -> int:
    """Encode `text` with `tokenizer` and write the token ids to `out_path`
    as raw uint16. Returns the number of tokens written."""
    ids = tokenizer.encode(text)
    if tokenizer.vocab_size >= 65536:
        raise ValueError("uint16 storage requires vocab_size < 65536")
    arr = np.array(ids, dtype=np.uint16)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out_path)
    return len(arr)


def load_token_memmap(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def split_train_val(data: np.ndarray, val_fraction: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    n = len(data)
    n_val = max(1, int(n * val_fraction))
    return data[: n - n_val], data[n - n_val :]


def get_batch(
    data: np.ndarray, block_size: int, batch_size: int, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random contiguous windows of length `block_size`
    from `data`. Targets are the same window shifted right by one token --
    the standard next-token-prediction training pair."""
    if len(data) <= block_size:
        raise ValueError(
            f"data has only {len(data)} tokens, need > block_size ({block_size})"
        )
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if device != "cpu":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y
