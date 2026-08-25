"""
A genuinely restricted arithmetic sandbox -- NOT Python's `eval`/`exec`.

This walks a parsed AST and only ever executes a fixed whitelist of node
types (numeric literals, +, -, *, /, //, %, **, unary +/-, parentheses).
There is no name lookup, no attribute access, no function/method calls, no
imports, no comprehensions, no subscripting -- so there is no way to reach
`__import__`, `os`, `open`, or anything else outside pure arithmetic,
unlike a naive `eval(expr)` which would happily execute those.

This is deliberately scoped to arithmetic only: it's Phase 5's ground-truth
oracle for grading math problems, not a general code-execution sandbox
(that's a separate, much bigger concern -- Phase 7's tool-calling /
code-execution work, which needs process isolation, timeouts, resource
limits, etc. Don't reuse this module for anything beyond arithmetic.)
"""
import ast
import operator
from typing import Union

Number = Union[int, float]

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains anything outside the arithmetic
    whitelist (names, calls, attribute access, etc.)."""


def safe_eval_arithmetic(expr: str) -> Number:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"could not parse expression: {e}") from e
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError(f"disallowed constant: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_BINOPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))

    raise UnsafeExpressionError(f"disallowed expression node: {type(node).__name__}")
