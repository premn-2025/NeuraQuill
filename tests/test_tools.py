"""
Phase 7 test suite: tools (calculator, registry), the tool-call wire
format (parsing/formatting), and the generate<->execute<->continue
orchestrator.

Run with: pytest tests/test_tools.py -v
"""
import sys
from pathlib import Path
from typing import List

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.orchestrator import ToolCallOrchestrator  # noqa: E402
from inference.tool_format import (  # noqa: E402
    TOOL_SPECIAL_TOKENS,
    ToolCallParseError,
    format_tool_call,
    format_tool_result,
    parse_tool_call,
)
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from tools.base import Tool  # noqa: E402
from tools.calculator import CalculatorTool  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402
from training.chat_template import encode_conversation  # noqa: E402
from training.tool_sft import build_tool_sft_dataset, mask_tool_results  # noqa: E402


# --------------------------------------------------------------------------
# CalculatorTool
# --------------------------------------------------------------------------


def test_calculator_correct_results():
    calc = CalculatorTool()
    assert calc.run({"expression": "2 + 2"}) == "4"
    assert calc.run({"expression": "23 * 47"}) == "1081"
    assert calc.run({"expression": "(3 + 4) * 2"}) == "14"


def test_calculator_missing_expression_returns_error_string_not_raise():
    calc = CalculatorTool()
    result = calc.run({})
    assert isinstance(result, str)
    assert result.startswith("Error")


def test_calculator_non_string_expression_returns_error_string():
    calc = CalculatorTool()
    result = calc.run({"expression": 5})
    assert result.startswith("Error")


def test_calculator_unsafe_expression_returns_error_not_raise():
    calc = CalculatorTool()
    result = calc.run({"expression": "__import__('os').system('echo hi')"})
    assert result.startswith("Error")


def test_calculator_division_by_zero_returns_error_not_raise():
    calc = CalculatorTool()
    result = calc.run({"expression": "1 / 0"})
    assert result.startswith("Error")


def test_calculator_malformed_expression_returns_error_not_raise():
    calc = CalculatorTool()
    result = calc.run({"expression": "2 + "})
    assert result.startswith("Error")


# --------------------------------------------------------------------------
# ToolRegistry
# --------------------------------------------------------------------------


def test_registry_dispatches_to_correct_tool():
    registry = ToolRegistry([CalculatorTool()])
    assert registry.execute("calculator", {"expression": "6 * 7"}) == "42"


def test_registry_unknown_tool_returns_error_not_raise():
    registry = ToolRegistry([CalculatorTool()])
    result = registry.execute("search", {"query": "x"})
    assert isinstance(result, str)
    assert "unknown tool" in result.lower()


class _ExplodingTool(Tool):
    name = "exploder"
    description = "always raises, to test the registry catches it"

    def run(self, arguments):
        raise RuntimeError("boom")


def test_registry_catches_tool_exceptions():
    registry = ToolRegistry([_ExplodingTool()])
    result = registry.execute("exploder", {})
    assert isinstance(result, str)
    assert "boom" in result


class _SpyTool(Tool):
    name = "spy"
    description = "records calls"

    def __init__(self):
        self.calls = []

    def run(self, arguments):
        self.calls.append(arguments)
        return "ok"


def test_registry_has_and_names():
    registry = ToolRegistry([CalculatorTool()])
    assert registry.has("calculator")
    assert not registry.has("nonexistent")
    assert "calculator" in registry.names()


# --------------------------------------------------------------------------
# tool-call wire format
# --------------------------------------------------------------------------


def test_format_and_parse_roundtrip():
    call_text = format_tool_call("calculator", {"expression": "2+2"})
    inner = call_text[len("<|tool_call|>") : -len("<|/tool_call|>")]
    tool, args = parse_tool_call(inner)
    assert tool == "calculator"
    assert args == {"expression": "2+2"}


def test_format_tool_result_wraps_correctly():
    assert format_tool_result("4") == "<|tool_result|>4<|/tool_result|>"


@pytest.mark.parametrize(
    "bad_text",
    [
        "not json",
        "[1, 2, 3]",                                  # valid JSON, not an object
        '{"arguments": {"expression": "2+2"}}',        # missing 'tool'
        '{"tool": "calculator"}',                      # missing 'arguments'
        '{"tool": "calculator", "arguments": "x"}',     # arguments not an object
        '{"tool": 5, "arguments": {}}',                 # tool not a string
        "",
    ],
)
def test_parse_tool_call_rejects_malformed_input(bad_text):
    with pytest.raises(ToolCallParseError):
        parse_tool_call(bad_text)


# --------------------------------------------------------------------------
# ToolCallOrchestrator -- tested against a scripted model double so the
# loop logic is exercised deterministically, without needing a trained
# model or GPU
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tool_tokenizer():
    corpus = (
        "I will calculate this now. The answer is correct. Sorry, there was an error. "
        "Let me try again with a different approach. "
    ) * 20
    tok = ByteLevelBPETokenizer.train([corpus], vocab_size=600, special_tokens=("<|endoftext|>", "<|pad|>"))
    tok.add_special_tokens(["<|system|>", "<|user|>", "<|assistant|>"] + TOOL_SPECIAL_TOKENS)
    return tok


class ScriptedModel:
    """Test double for GPT.generate: ignores idx/temperature/top_k
    entirely and just appends the next pre-programmed continuation each
    call, so orchestrator control flow can be tested without a trained
    model. Each continuation's text must end with either <|endoftext|> or
    <|/tool_call|> to exercise the orchestrator's stop-condition logic
    the same way a real model's stop_token_ids-based generate() would."""

    def __init__(self, tokenizer, continuations: List[str]):
        self.tokenizer = tokenizer
        self._continuations = list(continuations)
        self.calls = 0

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token_ids=None):
        text = self._continuations[self.calls]
        self.calls += 1
        new_ids = self.tokenizer.encode(text)
        new_tensor = torch.tensor([new_ids], dtype=torch.long, device=idx.device)
        return torch.cat([idx, new_tensor], dim=1)


def _make_idx(tokenizer, prompt="<|user|>What is 2+2?<|endoftext|><|assistant|>"):
    ids = tokenizer.encode(prompt)
    return torch.tensor([ids], dtype=torch.long)


def test_orchestrator_no_tool_call_passthrough(tool_tokenizer):
    model = ScriptedModel(tool_tokenizer, ["The answer is correct.<|endoftext|>"])
    orch = ToolCallOrchestrator(model, tool_tokenizer, ToolRegistry([CalculatorTool()]))
    reply = orch.run(_make_idx(tool_tokenizer))
    assert reply == "The answer is correct."
    assert model.calls == 1


def test_orchestrator_single_tool_call_roundtrip(tool_tokenizer):
    call_text = format_tool_call("calculator", {"expression": "2+2"})
    model = ScriptedModel(
        tool_tokenizer,
        [
            f"I will calculate this now. {call_text}",
            "The answer is correct.<|endoftext|>",
        ],
    )
    spy = _SpyTool()
    orch = ToolCallOrchestrator(model, tool_tokenizer, ToolRegistry([CalculatorTool(), spy]))
    reply = orch.run(_make_idx(tool_tokenizer))

    assert model.calls == 2
    assert reply.startswith("I will calculate this now.")
    assert reply.endswith("The answer is correct.")
    # the injected tool result ("4") must appear in the full transcript
    full_text = tool_tokenizer.decode(orch.last_full_ids)
    assert "<|tool_result|>4<|/tool_result|>" in full_text


def test_orchestrator_calls_correct_tool_with_correct_arguments(tool_tokenizer):
    call_text = format_tool_call("spy", {"expression": "9*9"})
    model = ScriptedModel(tool_tokenizer, [f"calling. {call_text}", "done.<|endoftext|>"])
    spy = _SpyTool()
    orch = ToolCallOrchestrator(model, tool_tokenizer, ToolRegistry([spy]))
    orch.run(_make_idx(tool_tokenizer))
    assert spy.calls == [{"expression": "9*9"}]


def test_orchestrator_chained_tool_calls(tool_tokenizer):
    call1 = format_tool_call("spy", {"expression": "1"})
    call2 = format_tool_call("spy", {"expression": "2"})
    model = ScriptedModel(
        tool_tokenizer,
        [f"first. {call1}", f"second. {call2}", "done.<|endoftext|>"],
    )
    spy = _SpyTool()
    orch = ToolCallOrchestrator(model, tool_tokenizer, ToolRegistry([spy]), max_tool_calls=4)
    reply = orch.run(_make_idx(tool_tokenizer))
    assert spy.calls == [{"expression": "1"}, {"expression": "2"}]
    assert reply.endswith("done.")


def test_orchestrator_respects_max_tool_calls_cap(tool_tokenizer):
    """A model that calls tools forever must not hang the loop."""
    call_text = format_tool_call("spy", {"expression": "1"})
    # 10 continuations, every single one ending in another tool call -- far
    # more than max_tool_calls allows
    model = ScriptedModel(tool_tokenizer, [call_text] * 10)
    spy = _SpyTool()
    orch = ToolCallOrchestrator(model, tool_tokenizer, ToolRegistry([spy]), max_tool_calls=3)
    reply = orch.run(_make_idx(tool_tokenizer))  # must return, not hang
    assert isinstance(reply, str)
    assert model.calls == 4  # max_tool_calls + 1 generate() invocations, then stops
    assert len(spy.calls) == 4


def test_orchestrator_malformed_tool_call_json_feeds_back_error_not_crash(tool_tokenizer):
    bad_call = "<|tool_call|>not valid json<|/tool_call|>"
    model = ScriptedModel(tool_tokenizer, [f"trying. {bad_call}", "Sorry, there was an error.<|endoftext|>"])
    orch = ToolCallOrchestrator(model, tool_tokenizer, ToolRegistry([CalculatorTool()]))
    reply = orch.run(_make_idx(tool_tokenizer))  # must not raise
    assert reply.endswith("Sorry, there was an error.")
    full_text = tool_tokenizer.decode(orch.last_full_ids)
    assert "<|tool_result|>Error:" in full_text


def test_orchestrator_unknown_tool_name_feeds_back_error_not_crash(tool_tokenizer):
    call_text = format_tool_call("nonexistent_tool", {"x": 1})
    model = ScriptedModel(tool_tokenizer, [f"calling. {call_text}", "done.<|endoftext|>"])
    orch = ToolCallOrchestrator(model, tool_tokenizer, ToolRegistry([CalculatorTool()]))
    reply = orch.run(_make_idx(tool_tokenizer))
    assert reply.endswith("done.")
    full_text = tool_tokenizer.decode(orch.last_full_ids)
    assert "unknown tool" in full_text.lower()


# --------------------------------------------------------------------------
# mask_tool_results / build_tool_sft_dataset: tool results must never be a
# training target, since they're injected by the orchestrator, not
# generated by the model
# --------------------------------------------------------------------------


def test_mask_tool_results_excludes_only_the_result_span(tool_tokenizer):
    call_text = format_tool_call("calculator", {"expression": "2+2"})
    result_text = format_tool_result("4")
    turns = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": f"Let me check.{call_text}{result_text} The answer is 4."},
    ]
    ids, mask = encode_conversation(tool_tokenizer, turns)
    new_mask = mask_tool_results(ids, mask, tool_tokenizer)

    # everything that was masked-in for a non-tool-result reason must stay masked-in
    result_start_id = tool_tokenizer.vocab["<|tool_result|>"]
    result_end_id = tool_tokenizer.vocab["<|/tool_result|>"]
    in_result = False
    for tok_id, old_m, new_m in zip(ids, mask, new_mask):
        if tok_id == result_start_id:
            in_result = True
        if tok_id in (result_start_id, result_end_id) or in_result:
            assert new_m == 0
        else:
            assert new_m == old_m
        if tok_id == result_end_id:
            in_result = False

    # the tool_call span (before the result) must still be trained on
    call_start_id = tool_tokenizer.vocab["<|tool_call|>"]
    assert new_mask[ids.index(call_start_id)] == 1


def test_mask_tool_results_leaves_conversation_without_tool_call_unchanged(tool_tokenizer):
    turns = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there, how can I help?"},
    ]
    ids, mask = encode_conversation(tool_tokenizer, turns)
    new_mask = mask_tool_results(ids, mask, tool_tokenizer)
    assert new_mask == mask


def test_build_tool_sft_dataset_shapes(tool_tokenizer):
    call_text = format_tool_call("calculator", {"expression": "2+2"})
    result_text = format_tool_result("4")
    conversations = [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": f"Let me check.{call_text}{result_text} The answer is 4."},
        ]
    ]
    # block_size must comfortably fit the whole conversation (~92 tokens
    # here) -- too small and truncation would drop trailing content, making
    # the "no truncation" comparison below invalid (same lesson as the
    # analogous DPO test fix)
    block_size = 128
    X, Y = build_tool_sft_dataset(tool_tokenizer, conversations, block_size)
    assert X.shape == (1, block_size)
    assert Y.shape == (1, block_size)
    # no ignore_index target should correspond to a tool_result-span position
    ids, mask = encode_conversation(tool_tokenizer, conversations[0])
    new_mask = mask_tool_results(ids, mask, tool_tokenizer)
    assert sum(1 for m in new_mask if m == 1) == sum(1 for v in Y[0] if v != -1)
