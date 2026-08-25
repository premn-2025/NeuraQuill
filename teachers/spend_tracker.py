"""
Tracks Anthropic API spend, persisted to disk across runs, with a hard
budget cap checked BEFORE every call (not just logged after). "Don't let
it hit zero" is a code guarantee here, not a hope -- a call whose
worst-case cost would exceed the remaining budget is refused before it
ever reaches the network.

Pricing is Anthropic's published first-party per-1M-token rates (input,
output) as of this session; update PRICING if new models are added or
prices change.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_LOG_PATH = "data/anthropic_spend_log.json"

# $ per 1M tokens: (input, output)
PRICING: Dict[str, Tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class BudgetExceededError(RuntimeError):
    pass


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model in PRICING:
        input_rate, output_rate = PRICING[model]
    else:
        # unknown model -- fail safe by assuming the most expensive known
        # rate rather than silently under-counting spend
        input_rate = max(r[0] for r in PRICING.values())
        output_rate = max(r[1] for r in PRICING.values())
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


@dataclass
class SpendTracker:
    budget_usd: float
    log_path: str = DEFAULT_LOG_PATH
    _total_spent: float = field(default=0.0, init=False, repr=False)
    _calls: List[dict] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._load()

    def _load(self) -> None:
        path = Path(self.log_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._total_spent = data.get("total_spent", 0.0)
            self._calls = data.get("calls", [])
        else:
            self._total_spent = 0.0
            self._calls = []

    def _save(self) -> None:
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"total_spent": self._total_spent, "calls": self._calls}, indent=2),
            encoding="utf-8",
        )

    @property
    def total_spent(self) -> float:
        return self._total_spent

    @property
    def remaining(self) -> float:
        return self.budget_usd - self._total_spent

    def check_budget(self, estimated_next_cost: float) -> None:
        """Raises BudgetExceededError -- and makes no API call -- if
        `estimated_next_cost` (a worst-case, pre-call estimate) would
        push cumulative spend past the budget."""
        if self._total_spent + estimated_next_cost > self.budget_usd:
            raise BudgetExceededError(
                f"Budget exceeded: spent ${self._total_spent:.4f} of ${self.budget_usd:.2f}; "
                f"next call estimated at ${estimated_next_cost:.4f} would push total to "
                f"${self._total_spent + estimated_next_cost:.4f}"
            )

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Records the ACTUAL cost of a completed call (using the API's
        own reported token counts, not the pre-call estimate) and
        persists immediately -- so spend survives even if the process is
        killed right after this call."""
        cost = estimate_cost(model, input_tokens, output_tokens)
        self._total_spent += cost
        self._calls.append(
            {"model": model, "input_tokens": input_tokens, "output_tokens": output_tokens, "cost": cost}
        )
        self._save()
        return cost
