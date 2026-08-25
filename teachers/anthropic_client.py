"""
Client for using Claude itself as a teacher model (Phase 11) -- generates
targeted synthetic training data without needing any external API beyond
Anthropic's own. Requires ANTHROPIC_API_KEY (see .env.example); call
teachers.env.load_env() before generate() so the key from .env makes it
into os.environ.

Every call is checked against a persistent, hard-enforced budget
(ANTHROPIC_BUDGET_USD in .env, default a conservative $10) BEFORE it's
sent -- a call whose worst-case cost would exceed the remaining budget is
refused with BudgetExceededError, never reaching the network. Actual
spend (from the API's own reported token counts, not the pre-call
estimate) is recorded and persisted after every successful call, so
cumulative spend is accurate and survives across separate script runs.
"""
from typing import Optional

import anthropic

from .env import get_env, require_env
from .spend_tracker import SpendTracker, estimate_cost

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_BUDGET_USD = 10.0  # deliberately conservative; raise ANTHROPIC_BUDGET_USD in .env to spend more

_tracker: Optional[SpendTracker] = None


def get_tracker() -> SpendTracker:
    global _tracker
    if _tracker is None:
        budget = float(get_env("ANTHROPIC_BUDGET_USD", str(DEFAULT_BUDGET_USD)))
        _tracker = SpendTracker(budget_usd=budget)
    return _tracker


def generate(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.7,
) -> str:
    require_env("ANTHROPIC_API_KEY")  # fails fast with a clear message, not a bare SDK error
    model = model or get_env("CLAUDE_TEACHER_MODEL", DEFAULT_MODEL)

    tracker = get_tracker()
    # worst-case pre-call estimate: assume the full max_tokens gets used as
    # output (the true ceiling regardless of how much is actually
    # generated) plus a rough ~4-chars/token input estimate -- deliberately
    # pessimistic so the budget can never be blown past, only under-spent
    # relative to what was reserved
    rough_input_tokens = (len(prompt) + len(system or "")) // 4
    worst_case_cost = estimate_cost(model, rough_input_tokens, max_tokens)
    tracker.check_budget(worst_case_cost)  # raises BudgetExceededError, no call made, if this would exceed budget

    client = anthropic.Anthropic()
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)

    actual_cost = tracker.record(model, response.usage.input_tokens, response.usage.output_tokens)
    print(
        f"    [claude spend] {response.usage.input_tokens} in + {response.usage.output_tokens} out "
        f"= ${actual_cost:.4f} (total spent: ${tracker.total_spent:.4f} / ${tracker.budget_usd:.2f})"
    )

    text_parts = [block.text for block in response.content if block.type == "text"]
    return "".join(text_parts)
