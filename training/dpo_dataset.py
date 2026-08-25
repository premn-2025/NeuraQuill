"""Tokenize (question, chosen, rejected) preference pairs into fixed-shape
(X_chosen, Y_chosen, X_rejected, Y_rejected) arrays, reusing the same chat
template + masking as SFT -- loss is only ever computed on assistant
tokens, for both the chosen and rejected completion of the same prompt."""
from typing import List, Tuple

import numpy as np
import torch

from .chat_template import build_training_example, encode_conversation

SYSTEM_PROMPT = "You are a helpful assistant that solves problems step by step and clearly states the final answer."


def build_dpo_dataset(tokenizer, pairs, block_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pad_id = tokenizer.vocab["<|pad|>"]
    Xc, Yc, Xr, Yr = [], [], [], []
    for pair in pairs:
        chosen_turns = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair.question},
            {"role": "assistant", "content": pair.chosen},
        ]
        rejected_turns = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair.question},
            {"role": "assistant", "content": pair.rejected},
        ]
        ids_c, mask_c = encode_conversation(tokenizer, chosen_turns)
        ids_r, mask_r = encode_conversation(tokenizer, rejected_turns)
        xc, yc = build_training_example(ids_c, mask_c, block_size, pad_id)
        xr, yr = build_training_example(ids_r, mask_r, block_size, pad_id)
        Xc.append(xc)
        Yc.append(yc)
        Xr.append(xr)
        Yr.append(yr)
    return (
        np.array(Xc, dtype=np.int64), np.array(Yc, dtype=np.int64),
        np.array(Xr, dtype=np.int64), np.array(Yr, dtype=np.int64),
    )


def get_dpo_batch(
    Xc: np.ndarray, Yc: np.ndarray, Xr: np.ndarray, Yr: np.ndarray, batch_size: int, device: str = "cpu"
):
    ix = np.random.randint(0, len(Xc), size=batch_size)
    tensors = [torch.from_numpy(a[ix]) for a in (Xc, Yc, Xr, Yr)]
    if device != "cpu":
        tensors = [t.to(device, non_blocking=True) for t in tensors]
    return tuple(tensors)
