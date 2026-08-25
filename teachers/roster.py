"""
Builds the teacher roster from .env config (OLLAMA_TEACHER_MODELS,
CLAUDE_TEACHER_MODEL) and assigns each a capability profile, so a data-
generation run can route questions to the teachers suited for that
category instead of asking every teacher every question.

Default profile is deliberately inclusive for "reasoning"/"general" --
Phase 11's first real use is grade-school arithmetic/algebra, which any
general-purpose chat model can reasonably attempt -- while "math"/"code"/
"research" stay narrower, for when a coding- or research-focused data
run gets built later.
"""
from typing import List

from .base import Teacher
from .env import get_env

CAPABILITY_PROFILES = {
    "deepseek-r1": ["math", "reasoning", "general"],
    "qwen": ["math", "code", "reasoning", "general"],
    "gemma": ["reasoning", "general"],
    "llama": ["reasoning", "general"],
    "mistral": ["reasoning", "general"],
    "glm": ["math", "reasoning", "general"],
    "command-r": ["reasoning", "general", "research"],  # Cohere's RAG/tool-use-focused family
    "claude": ["math", "code", "reasoning", "general", "research"],
}
DEFAULT_CAPABILITIES = ["general"]  # unrecognized model -> safest, narrowest default


def capabilities_for(model_tag: str) -> List[str]:
    lower = model_tag.lower()
    for prefix, caps in CAPABILITY_PROFILES.items():
        if lower.startswith(prefix):
            return caps
    return DEFAULT_CAPABILITIES


def build_roster() -> List[Teacher]:
    roster: List[Teacher] = []

    ollama_models = [m.strip() for m in get_env("OLLAMA_TEACHER_MODELS", "").split(",") if m.strip()]
    for model in ollama_models:
        roster.append(
            Teacher(name=f"ollama/{model}", provider="ollama", model=model, capabilities=capabilities_for(model))
        )

    claude_model = get_env("CLAUDE_TEACHER_MODEL")
    if claude_model:
        roster.append(
            Teacher(
                name=f"anthropic/{claude_model}", provider="anthropic", model=claude_model,
                capabilities=capabilities_for(claude_model),
            )
        )

    return roster


def teachers_for_category(roster: List[Teacher], category: str) -> List[Teacher]:
    return [t for t in roster if category in t.capabilities]
