"""AdamW with weight decay applied only to matrix (2D+) parameters --
biases and LayerNorm gains are excluded, which is standard GPT-2/nanoGPT
practice (decaying a 1D scale/shift parameter toward zero has no principled
justification the way it does for weight matrices)."""
import torch


def configure_optimizer(
    model: torch.nn.Module,
    weight_decay: float,
    lr: float,
    betas=(0.9, 0.95),
) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)
