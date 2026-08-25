"""
The tool-call wire format: four special tokens delimit a call and its
injected result.

    <|assistant|>I'll compute that.<|tool_call|>{"tool": "calculator",
    "arguments": {"expression": "23 * 47"}}<|/tool_call|><|tool_result|>
    1081<|/tool_result|> So the answer is 1081.<|endoftext|>

The model only ever generates up through `<|/tool_call|>` itself --
`<|tool_result|>...<|/tool_result|>` is injected by the orchestrator, not
generated, which is what makes the tool's output authoritative rather
than something the model could hallucinate.
"""
import json
from typing import Any, Dict, Tuple

TOOL_CALL_START = "<|tool_call|>"
TOOL_CALL_END = "<|/tool_call|>"
TOOL_RESULT_START = "<|tool_result|>"
TOOL_RESULT_END = "<|/tool_result|>"

TOOL_SPECIAL_TOKENS = [TOOL_CALL_START, TOOL_CALL_END, TOOL_RESULT_START, TOOL_RESULT_END]


class ToolCallParseError(ValueError):
    """Raised when the text between <|tool_call|> and <|/tool_call|> isn't
    a well-formed {"tool": ..., "arguments": {...}} object."""


def parse_tool_call(text: str) -> Tuple[str, Dict[str, Any]]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ToolCallParseError(f"invalid JSON in tool call: {e}") from e
    if not isinstance(obj, dict):
        raise ToolCallParseError("tool call JSON must be an object")
    if "tool" not in obj:
        raise ToolCallParseError("tool call JSON missing required 'tool' field")
    if not isinstance(obj["tool"], str):
        raise ToolCallParseError("'tool' field must be a string")
    if "arguments" not in obj:
        raise ToolCallParseError("tool call JSON missing required 'arguments' field")
    if not isinstance(obj["arguments"], dict):
        raise ToolCallParseError("'arguments' must be a JSON object")
    return obj["tool"], obj["arguments"]


def format_tool_call(tool: str, arguments: Dict[str, Any]) -> str:
    payload = json.dumps({"tool": tool, "arguments": arguments})
    return f"{TOOL_CALL_START}{payload}{TOOL_CALL_END}"


def format_tool_result(result: str) -> str:
    return f"{TOOL_RESULT_START}{result}{TOOL_RESULT_END}"
