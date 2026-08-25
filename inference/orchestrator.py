"""
The generate <-> execute <-> continue loop: let the model generate until
it either finishes (<|endoftext|>) or completes a tool call
(<|/tool_call|>), and in the latter case, parse + execute the call, splice
the result back into the context as fixed (non-generated) tokens, and
resume generation -- up to `max_tool_calls` times per turn, so a model
stuck calling tools forever can't hang the loop.
"""
from typing import List, Optional

import torch

from tools.registry import ToolRegistry

from .tool_format import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    ToolCallParseError,
    format_tool_result,
    parse_tool_call,
)


class ToolCallOrchestrator:
    def __init__(self, model, tokenizer, registry: ToolRegistry, max_tool_calls: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.max_tool_calls = max_tool_calls
        self.last_full_ids: Optional[List[int]] = None  # populated by run(), useful for tests/debugging

    def run(
        self,
        idx: torch.Tensor,
        max_new_tokens_per_turn: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
    ) -> str:
        eot_id = self.tokenizer.vocab["<|endoftext|>"]
        call_end_id = self.tokenizer.vocab[TOOL_CALL_END]
        prompt_len = idx.shape[1]

        for _ in range(self.max_tool_calls + 1):
            idx = self.model.generate(
                idx, max_new_tokens=max_new_tokens_per_turn, temperature=temperature,
                top_k=top_k, stop_token_ids={eot_id, call_end_id},
            )
            last_id = idx[0, -1].item()
            if last_id == eot_id:
                break
            if last_id != call_end_id:
                break  # exhausted max_new_tokens_per_turn without hitting a stop token

            full_text = self.tokenizer.decode(idx[0].tolist())
            call_start_pos = full_text.rfind(TOOL_CALL_START)
            if call_start_pos == -1:
                break  # malformed: saw a call-end without a matching call-start

            call_json_text = full_text[call_start_pos + len(TOOL_CALL_START) : -len(TOOL_CALL_END)]
            try:
                tool_name, arguments = parse_tool_call(call_json_text)
                result = self.registry.execute(tool_name, arguments)
            except ToolCallParseError as e:
                result = f"Error: {e}"

            result_ids = self.tokenizer.encode(format_tool_result(result))
            result_tensor = torch.tensor([result_ids], dtype=torch.long, device=idx.device)
            idx = torch.cat([idx, result_tensor], dim=1)

        self.last_full_ids = idx[0].tolist()

        reply_ids = idx[0, prompt_len:].tolist()
        if reply_ids and reply_ids[-1] == eot_id:
            reply_ids = reply_ids[:-1]
        return self.tokenizer.decode(reply_ids)
