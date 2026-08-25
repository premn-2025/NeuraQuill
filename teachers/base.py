"""A common interface over every teacher, regardless of provider -- so
adding a new one (a different Ollama tag, a different API) is a roster
entry, not new call-site code."""
from dataclasses import dataclass, field
from typing import List, Optional

from . import anthropic_client, ollama_client
from .env import get_env


@dataclass
class Teacher:
    name: str          # e.g. "ollama/gemma3:4b" or "anthropic/claude-opus-5"
    provider: str       # "ollama" | "anthropic"
    model: str           # the actual model tag/id passed to that provider
    capabilities: List[str] = field(default_factory=list)  # e.g. ["math", "reasoning", "code", "general"]

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7) -> str:
        if self.provider == "ollama":
            host = get_env("OLLAMA_HOST", "http://localhost:11434")
            return ollama_client.generate(prompt, model=self.model, system=system, host=host, temperature=temperature)
        if self.provider == "anthropic":
            return anthropic_client.generate(prompt, system=system, model=self.model, temperature=temperature)
        raise ValueError(f"unknown teacher provider: {self.provider!r} (teacher {self.name!r})")
