"""Build a fixed-shape SFT dataset (X, Y) from a list of conversations,
and sample random batches from it. Unlike pretraining's memmapped token
stream, SFT examples are discrete conversations, so the whole dataset is
just two small in-memory numpy arrays (# examples is small compared to a
pretraining corpus)."""
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .chat_template import build_training_example, encode_conversation


def build_sft_dataset(
    tokenizer, conversations: Sequence[List[Dict[str, str]]], block_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    pad_id = tokenizer.vocab["<|pad|>"]
    xs, ys = [], []
    for turns in conversations:
        ids, mask = encode_conversation(tokenizer, turns)
        x, y = build_training_example(ids, mask, block_size, pad_id)
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)


def get_sft_batch(
    X: np.ndarray, Y: np.ndarray, batch_size: int, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(X), size=batch_size)
    x = torch.from_numpy(X[ix])
    y = torch.from_numpy(Y[ix])
    if device != "cpu":
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    return x, y
