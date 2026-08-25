"""The calculator tool -- reuses Phase 5's verifier sandbox directly, so
the same restricted, `eval`-free arithmetic evaluator that grades
reasoning problems is also what the model gets to call at inference
time. This is exactly the capability Phase 5/6 showed the model itself
lacks (~11% arithmetic accuracy): a real calculator makes that gap moot
for anything the model correctly decides to hand off."""
from typing import Any, Dict

from verifiers.sandbox import UnsafeExpressionError, safe_eval_arithmetic

from .base import Tool


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate an arithmetic expression (+, -, *, /, //, %, **, parentheses) "
        "and return the exact numeric result. Arguments: {'expression': str}."
    )

    def run(self, arguments: Dict[str, Any]) -> str:
        if "expression" not in arguments:
            return "Error: missing required argument 'expression'"
        expr = arguments["expression"]
        if not isinstance(expr, str):
            return f"Error: 'expression' must be a string, got {type(expr).__name__}"
        try:
            result = safe_eval_arithmetic(expr)
        except UnsafeExpressionError as e:
            return f"Error: invalid expression: {e}"
        except ZeroDivisionError:
            return "Error: division by zero"
        return str(result)
