"""
DPO (Direct Preference Optimization, Rafailov et al. 2023): train the
policy model directly on (chosen, rejected) preference pairs against a
frozen reference model, with no separate reward model and no RL loop.

    L = -log sigmoid( beta * [ (logpi(chosen) - logpi_ref(chosen))
                              - (logpi(rejected) - logpi_ref(rejected)) ] )

Both logpi terms are the *sum* of log-probs over only the assistant
response tokens (via the same ignore_index=-1 masking SFT uses), given the
same prompt -- so the loss pushes the policy to prefer the chosen
response over the rejected one, relative to how much the reference model
already preferred it (which is what keeps DPO from just collapsing onto
memorizing "chosen" verbatim).
"""
import copy
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from model.lm import GPT
from training.checkpoint import save_checkpoint
from training.dpo_dataset import get_dpo_batch
from training.lr_schedule import get_lr
from training.optim import configure_optimizer

IGNORE_INDEX = -1


def sequence_logprob(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum of log p(y_t | x_<=t) over positions where y != IGNORE_INDEX,
    per example. Shape (B,)."""
    logits, _ = model(x)  # (B, T, V) -- no `targets` so we control masking ourselves
    log_probs = F.log_softmax(logits, dim=-1)

    mask = (y != IGNORE_INDEX)
    y_safe = y.clone()
    y_safe[~mask] = 0  # dummy index for masked positions, zeroed out right after via `mask`

    token_logprobs = torch.gather(log_probs, dim=2, index=y_safe.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    return token_logprobs.sum(dim=1)


def dpo_loss(
    policy: GPT, ref: GPT, x_c, y_c, x_r, y_r, beta: float = 0.1
):
    policy_chosen = sequence_logprob(policy, x_c, y_c)
    policy_rejected = sequence_logprob(policy, x_r, y_r)
    with torch.no_grad():
        ref_chosen = sequence_logprob(ref, x_c, y_c)
        ref_rejected = sequence_logprob(ref, x_r, y_r)

    policy_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = beta * (policy_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    chosen_rewards = beta * (policy_chosen - ref_chosen).detach()
    rejected_rewards = beta * (policy_rejected - ref_rejected).detach()
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    margin = (chosen_rewards - rejected_rewards).mean()

    return loss, {"accuracy": accuracy.item(), "margin": margin.item()}


@dataclass
class DPOTrainConfig:
    max_steps: int = 500
    warmup_steps: int = 20
    batch_size: int = 16
    max_lr: float = 5e-6
    min_lr: float = 5e-7
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    beta: float = 0.1
    eval_interval: int = 50
    eval_batches: int = 5
    checkpoint_path: Optional[str] = None
    device: str = "cpu"
    seed: int = 1337


@dataclass
class DPOHistory:
    steps: List[int] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_steps: List[int] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    val_margins: List[float] = field(default_factory=list)


@torch.no_grad()
def estimate_dpo(policy, ref, Xc, Yc, Xr, Yr, batch_size, num_batches, beta, device):
    policy.eval()
    losses, accs, margins = [], [], []
    for _ in range(num_batches):
        x_c, y_c, x_r, y_r = get_dpo_batch(Xc, Yc, Xr, Yr, min(batch_size, len(Xc)), device=device)
        loss, metrics = dpo_loss(policy, ref, x_c, y_c, x_r, y_r, beta=beta)
        losses.append(loss.item())
        accs.append(metrics["accuracy"])
        margins.append(metrics["margin"])
    policy.train()
    return sum(losses) / len(losses), sum(accs) / len(accs), sum(margins) / len(margins)


def train_dpo(
    policy: GPT,
    config,
    Xc_train, Yc_train, Xr_train, Yr_train,
    Xc_val, Yc_val, Xr_val, Yr_val,
    train_cfg: DPOTrainConfig,
    log_fn: Callable[[str], None] = print,
    ref: Optional[GPT] = None,
) -> DPOHistory:
    torch.manual_seed(train_cfg.seed)
    device = train_cfg.device
    policy.to(device)
    policy.train()

    if ref is None:
        ref = copy.deepcopy(policy)
    ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    optimizer = configure_optimizer(policy, train_cfg.weight_decay, train_cfg.max_lr)

    history = DPOHistory()
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(train_cfg.max_steps):
        lr = get_lr(step, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.max_lr, train_cfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x_c, y_c, x_r, y_r = get_dpo_batch(
            Xc_train, Yc_train, Xr_train, Yr_train, train_cfg.batch_size, device=device
        )
        loss, metrics = dpo_loss(policy, ref, x_c, y_c, x_r, y_r, beta=train_cfg.beta)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), train_cfg.grad_clip)
        optimizer.step()

        history.steps.append(step)
        history.train_losses.append(loss.item())

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
            val_loss, val_acc, val_margin = estimate_dpo(
                policy, ref, Xc_val, Yc_val, Xr_val, Yr_val,
                train_cfg.batch_size, train_cfg.eval_batches, train_cfg.beta, device,
            )
            history.val_steps.append(step)
            history.val_losses.append(val_loss)
            history.val_accuracies.append(val_acc)
            history.val_margins.append(val_margin)
            dt = time.time() - t0
            log_fn(
                f"step {step:6d} | lr {lr:.2e} | train_loss {loss.item():.4f} "
                f"| val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | val_margin {val_margin:.3f} | {dt:.1f}s"
            )
            if train_cfg.checkpoint_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(train_cfg.checkpoint_path, policy, optimizer, step, best_val_loss, policy.config)

    return history
