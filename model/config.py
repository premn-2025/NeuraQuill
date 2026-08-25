from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Everything needed to build one NeuraQuill decoder-only transformer."""

    vocab_size: int
    block_size: int = 512   # max context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True       # bias terms in Linear/LayerNorm, GPT-2 style

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
