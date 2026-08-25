"""
SFT dataset construction for tool-use training data: reuses the Phase 4
chat template, then additionally masks out any
<|tool_result|>...<|/tool_result|> span. That text is injected by the
orchestrator at inference time (the tool's real output), never generated
by the model -- training on it as a target would teach the model to
hallucinate tool results instead of actually calling the tool.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from inference.tool_format import TOOL_RESULT_END, TOOL_RESULT_START
from training.chat_template import build_training_example, encode_conversation


def mask_tool_results(ids: List[int], mask: List[int], tokenizer) -> List[int]:
    start_id = tokenizer.vocab[TOOL_RESULT_START]
    end_id = tokenizer.vocab[TOOL_RESULT_END]
    new_mask = list(mask)
    in_result = False
    for i, tok_id in enumerate(ids):
        if tok_id == start_id:
            in_result = True
            new_mask[i] = 0
        elif tok_id == end_id:
            in_result = False
            new_mask[i] = 0
        elif in_result:
            new_mask[i] = 0
    return new_mask


def build_tool_sft_dataset(
    tokenizer, conversations: Sequence[List[Dict[str, str]]], block_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    pad_id = tokenizer.vocab["<|pad|>"]
    xs, ys = [], []
    for turns in conversations:
        ids, mask = encode_conversation(tokenizer, turns)
        mask = mask_tool_results(ids, mask, tokenizer)
        x, y = build_training_example(ids, mask, block_size, pad_id)
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)
