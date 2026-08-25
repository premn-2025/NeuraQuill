"""Perplexity = exp(average per-token cross-entropy loss). Lower is better;
it's "the model was, on average, this uncertain (as if choosing uniformly
among this many tokens) about what came next."""
import math

import torch

from model.lm import GPT
from training.dataset import get_batch


@torch.no_grad()
def estimate_loss(
    model: GPT, data, block_size: int, batch_size: int, num_batches: int, device: str = "cpu"
) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(data, block_size, batch_size, device=device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def perplexity_from_loss(loss: float) -> float:
    return math.exp(loss)
