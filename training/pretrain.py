"""
The core pretraining loop, factored as a plain function so it's both the
real training entry point and directly unit-testable on tiny synthetic
data (see tests/test_training.py).
"""
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from evaluation.perplexity import estimate_loss
from model.config import GPTConfig
from model.lm import GPT
from training.checkpoint import save_checkpoint
from training.dataset import get_batch
from training.lr_schedule import get_lr
from training.optim import configure_optimizer


@dataclass
class TrainConfig:
    max_steps: int = 1000
    warmup_steps: int = 100
    batch_size: int = 32
    block_size: int = 128
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 500
    checkpoint_path: Optional[str] = None
    device: str = "cpu"
    seed: int = 1337


@dataclass
class TrainHistory:
    steps: List[int] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_steps: List[int] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)


def train(
    model: GPT,
    config: GPTConfig,
    train_data,
    val_data,
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

        x, y = get_batch(train_data, train_cfg.block_size, train_cfg.batch_size, device=device)

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
            val_loss = estimate_loss(
                model, val_data, train_cfg.block_size,
                min(train_cfg.batch_size, len(val_data) // 2 or 1),
                train_cfg.eval_batches, device=device,
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
