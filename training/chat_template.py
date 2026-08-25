"""
Chat template for instruction tuning: turns a list of
{"role": "system"|"user"|"assistant", "content": str} dicts into token ids
plus a loss mask that is 1 only on assistant-generated tokens.

Format per turn:  <|{role}|>{content}<|endoftext|>

Only assistant content (and its closing <|endoftext|>) contribute to the
training loss -- the model should learn to *produce* assistant turns, not
to predict the user's or system's words, which is why SFT loss is masked
this way rather than trained like plain next-token pretraining.
"""
from typing import Dict, List, Sequence, Tuple

ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
EOT = "<|endoftext|>"
CHAT_SPECIAL_TOKENS = list(ROLE_TOKENS.values())  # to be added via tokenizer.add_special_tokens


def encode_conversation(tokenizer, turns: Sequence[Dict[str, str]]) -> Tuple[List[int], List[int]]:
    """Returns (ids, loss_mask), same length, loss_mask[i] == 1 iff ids[i]
    belongs to an assistant turn (its content or closing EOT)."""
    ids: List[int] = []
    mask: List[int] = []
    eot_id = tokenizer.vocab[EOT]

    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"Unknown role {role!r}; expected one of {list(ROLE_TOKENS)}")
        is_assistant = role == "assistant"

        role_id = tokenizer.vocab[ROLE_TOKENS[role]]
        ids.append(role_id)
        mask.append(0)  # the role tag itself is never a training target

        content_ids = tokenizer.encode(content)
        ids.extend(content_ids)
        mask.extend([1 if is_assistant else 0] * len(content_ids))

        ids.append(eot_id)
        mask.append(1 if is_assistant else 0)

    return ids, mask


def build_training_example(
    ids: List[int],
    mask: List[int],
    block_size: int,
    pad_id: int,
    ignore_index: int = -1,
) -> Tuple[List[int], List[int]]:
    """Turn (ids, mask) into a fixed-length (x, y) training pair:
    x = ids[:-1] (padded/truncated to block_size), y = ids[1:] with
    non-assistant / padding positions replaced by `ignore_index` so
    cross_entropy skips them."""
    ids = ids[: block_size + 1]
    mask = mask[: block_size + 1]

    pad_len = (block_size + 1) - len(ids)
    if pad_len > 0:
        ids = ids + [pad_id] * pad_len
        mask = mask + [0] * pad_len

    x = ids[:-1]
    y_raw = ids[1:]
    y_mask = mask[1:]
    y = [tok if m else ignore_index for tok, m in zip(y_raw, y_mask)]
    return x, y
