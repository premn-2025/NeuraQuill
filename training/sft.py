"""
The SFT (supervised fine-tuning / instruction tuning) training loop.
Structurally the same shape as training/pretrain.py's `train()` (same LR
schedule, optimizer, AMP, checkpointing) but batches come from a fixed
set of tokenized conversations instead of a continuous pretraining
token stream, and the loss is already masked to assistant-only tokens
via `ignore_index=-1` targets built by chat_template.build_training_example
(GPT.forward's cross_entropy already honors ignore_index=-1).
"""
import time
from typing import Callable, Optional

import numpy as np
import torch

from model.config import GPTConfig
from model.lm import GPT
from training.checkpoint import save_checkpoint
from training.lr_schedule import get_lr
from training.optim import configure_optimizer
from training.pretrain import TrainConfig, TrainHistory
from training.sft_dataset import get_sft_batch


@torch.no_grad()
def estimate_sft_loss(
    model: GPT, X: np.ndarray, Y: np.ndarray, batch_size: int, num_batches: int, device: str
) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_sft_batch(X, Y, min(batch_size, len(X)), device=device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_sft(
    model: GPT,
    config: GPTConfig,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    train_cfg: TrainConfig,
    log_fn: Callable[[str], None] = print,
    start_step: int = 0,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> TrainHistory:
    torch.manual_seed(train_cfg.seed)
    device = train_cfg.device
    model.to(device)
    model.train()

    if optimizer is None:
        optimizer = configure_optimizer(model, train_cfg.weight_decay, train_cfg.max_lr)

    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)

    history = TrainHistory()
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(start_step, train_cfg.max_steps):
        lr = get_lr(step, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.max_lr, train_cfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_sft_batch(X_train, Y_train, train_cfg.batch_size, device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            _, loss = model(x, targets=y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        history.steps.append(step)
        history.train_losses.append(loss.item())

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
            val_loss = estimate_sft_loss(
                model, X_val, Y_val, train_cfg.batch_size, train_cfg.eval_batches, device=device
            )
            history.val_steps.append(step)
            history.val_losses.append(val_loss)
            dt = time.time() - t0
            log_fn(
                f"step {step:6d} | lr {lr:.2e} | train_loss {loss.item():.4f} "
                f"| val_loss {val_loss:.4f} | {dt:.1f}s"
            )
            if train_cfg.checkpoint_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    train_cfg.checkpoint_path, model, optimizer, step, best_val_loss, config
                )

    return history
