"""
Client for using Claude itself as a teacher model (Phase 11) -- generates
targeted synthetic training data without needing any external API beyond
Anthropic's own. Requires ANTHROPIC_API_KEY (see .env.example); call
teachers.env.load_env() before generate() so the key from .env makes it
into os.environ.
"""
from typing import Optional

import anthropic

from .env import get_env, require_env

DEFAULT_MODEL = "claude-opus-5"


def generate(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.7,
) -> str:
    require_env("ANTHROPIC_API_KEY")  # fails fast with a clear message, not a bare SDK error
    client = anthropic.Anthropic()
    model = model or get_env("CLAUDE_TEACHER_MODEL", DEFAULT_MODEL)

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    text_parts = [block.text for block in response.content if block.type == "text"]
    return "".join(text_parts)
