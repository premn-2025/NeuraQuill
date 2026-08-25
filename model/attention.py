"""
Causal multi-head self-attention, implemented explicitly from the formula

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + causal_mask) V

No `nn.MultiheadAttention`. Every step (projecting Q/K/V, splitting into
heads, scoring, masking, softmax, weighting V, merging heads, output
projection) is written out so nothing is hidden inside a library call.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # one combined projection for Q, K, V (standard GPT-2 trick: fewer,
        # bigger matmuls instead of three separate small ones)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask: position i may only attend to positions <= i.
        # Registered as a buffer (not a parameter) so it moves with
        # .to(device) but is never trained / never saved as a learned weight.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        qkv = self.c_attn(x)                          # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)        # each (B, T, C)

        # split into heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention scores: (B, n_head, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal masking: forbid attending to future positions
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                                  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back

        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out
