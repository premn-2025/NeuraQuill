"""
Resize a pretrained model's token embedding / LM head to accommodate new
vocabulary added after pretraining (e.g. chat-template special tokens
added for SFT). Existing rows are copied over exactly; new rows get a
fresh random init consistent with the model's own init scheme.
"""
import torch
import torch.nn as nn

from .lm import GPT


def resize_token_embeddings(model: GPT, new_vocab_size: int) -> GPT:
    old_vocab_size, n_embd = model.wte.weight.shape
    if new_vocab_size == old_vocab_size:
        return model
    if new_vocab_size < old_vocab_size:
        raise ValueError(
            f"Shrinking vocab ({old_vocab_size} -> {new_vocab_size}) is not supported"
        )

    device = model.wte.weight.device
    new_wte = nn.Embedding(new_vocab_size, n_embd).to(device)
    nn.init.normal_(new_wte.weight, mean=0.0, std=0.02)
    with torch.no_grad():
        new_wte.weight[:old_vocab_size].copy_(model.wte.weight)

    model.wte = new_wte
    model.lm_head = nn.Linear(n_embd, new_vocab_size, bias=False).to(device)
    model.lm_head.weight = model.wte.weight  # re-tie: lm_head adopts the (copied+extended) wte weight
    model.config.vocab_size = new_vocab_size
    return model
