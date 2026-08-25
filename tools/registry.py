"""Name -> Tool dispatch. Never raises: an unknown tool name or a tool
implementation that misbehaves both come back as an error string, not an
exception, so the orchestrator loop never crashes mid-generation."""
from typing import Dict, Iterable, Optional

from .base import Tool


class ToolRegistry:
    def __init__(self, tools: Optional[Iterable[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self):
        return list(self._tools.keys())

    def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available tools: {self.names()}"
        try:
            return tool.run(arguments)
        except Exception as e:  # tool implementations shouldn't raise, but never trust that
            return f"Error: tool '{name}' raised an unexpected exception: {e}"
