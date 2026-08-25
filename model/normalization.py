"""Hand-written LayerNorm (PyTorch's nn.LayerNorm doesn't let bias be
optionally disabled pre-2.1, and writing it out makes the math explicit)."""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * weight + bias, normalized over the
    last dimension only (the embedding dimension)."""

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = x_norm * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
