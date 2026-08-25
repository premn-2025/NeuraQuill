"""Common interface every tool implements. A tool must never raise --
internal failures are caught and turned into a descriptive error string,
because whatever `run()` returns gets fed straight back into the model's
context; an uncaught exception would crash the whole orchestrator loop
over something as ordinary as a malformed argument."""
from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, arguments: Dict[str, Any]) -> str:
        ...
