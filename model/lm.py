"""
NeuraQuill: a GPT-style decoder-only transformer language model, built entirely
from the pieces in this package (no HuggingFace model classes).

    tokens -> token embedding + positional embedding
           -> N x Block (causal self-attention + MLP, pre-norm, residual)
           -> final LayerNorm
           -> LM head (tied with the token embedding) -> vocab logits
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .normalization import LayerNorm
from .transformer import Block


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # learned positional embedding
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: the input embedding and output projection share
        # weights (Press & Wolf, 2016) -- halves embedding-related params
        # and tends to improve small-model quality.
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2 paper: scale residual-branch output projections by
        # 1/sqrt(2 * n_layer) so activations don't blow up as depth grows
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)          # (B, T, n_embd)
        pos_emb = self.wpe(pos)          # (T, n_embd), broadcasts over batch
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        stop_token_ids: Optional[set] = None,
    ) -> torch.Tensor:
        """Autoregressively sample `max_new_tokens` tokens after `idx`
        (B, T) of context. Greedy if temperature -> 0.

        `stop_token_ids`, if given, ends generation as soon as the
        generated token is one of them -- but only when batch size is 1
        (with a batch, sequences would finish at different times, and
        this function doesn't do per-sequence masking, so all rows just
        run to `max_new_tokens` instead)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if stop_token_ids is not None and idx.size(0) == 1 and next_id.item() in stop_token_ids:
                break
        return idx
