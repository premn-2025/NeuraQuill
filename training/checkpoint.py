"""Checkpoint save/resume for pretraining."""
import dataclasses
from pathlib import Path
from typing import Optional

import torch

from model.config import GPTConfig
from model.lm import GPT


def save_checkpoint(
    path: str,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    config: GPTConfig,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "config": dataclasses.asdict(config),
        },
        out_path,
    )


def load_checkpoint(
    path: str, device: str = "cpu", optimizer: Optional[torch.optim.Optimizer] = None
):
    """Returns (model, optimizer, step, best_val_loss, config). If `optimizer`
    is given, its state is loaded in-place and it is also returned; otherwise
    the optimizer field of the return tuple is None."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return model, optimizer, ckpt["step"], ckpt["best_val_loss"], config
