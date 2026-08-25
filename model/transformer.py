import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .config import GPTConfig
from .mlp import MLP
from .normalization import LayerNorm


class Block(nn.Module):
    """One pre-norm transformer block:

        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Pre-norm (norm before the sublayer, not after) is what GPT-2/GPT-3 and
    most modern decoder-only models use -- it keeps gradients much better
    behaved at depth than the original post-norm Transformer.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
