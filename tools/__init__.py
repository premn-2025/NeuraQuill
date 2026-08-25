from .base import Tool
from .calculator import CalculatorTool
from .registry import ToolRegistry


def default_registry() -> ToolRegistry:
    return ToolRegistry([CalculatorTool()])


__all__ = ["Tool", "CalculatorTool", "ToolRegistry", "default_registry"]
