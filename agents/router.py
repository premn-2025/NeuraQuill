"""
Routes a subtask to the worker best suited for it: 'reasoning' (math/logic,
handled via Phase 7's tool-calling orchestrator) or 'knowledge' (handled
via Phase 8's RAG retrieval).

Deliberately rule-based rather than asking the 10M-param LLM to
self-classify: earlier phases (5, 6) showed this model's free-form
judgment is unreliable at this scale, while a small pattern-based
classifier is deterministic and testable. A learned/LLM-driven router is
natural future work once the base model scales up (Phase 10) -- and the
same structured-output machinery Phase 7 built (JSON inside special
tokens) would be the way to do it.
"""
import re

_MATH_PATTERN = re.compile(
    r"\d+\s*[\+\-\*/]\s*\d+"          # e.g. "12 + 7"
    r"|solve for x"
    r"|is\s+\w+\s+taller\s+than\s+\w+",  # comparison-style reasoning problems
    re.IGNORECASE,
)

ROUTES = ("reasoning", "knowledge")


def route(task: str) -> str:
    if _MATH_PATTERN.search(task):
        return "reasoning"
    return "knowledge"
