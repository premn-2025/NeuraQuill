"""
Synthetic preference pairs for DPO, built on top of Phase 5's generators
and verifier: `chosen` is a problem's real, verified-correct solution;
`rejected` is a plausible-looking but verified-incorrect one, constructed
by corrupting the final step the same way a model actually tends to err
(right procedure, wrong number / flipped conclusion) rather than by
generating obviously garbage text.

This sidesteps the usual DPO bottleneck (needing human preference labels)
the same way Phase 5 sidestepped needing human labels for reasoning data:
correctness is verifiable by construction, so "which one is better" is
just "which one the verifier grades correct."
"""
import random
from dataclasses import dataclass
from typing import List, Optional

from .generators import ReasoningProblem, generate_dataset
from .reward import grade


@dataclass
class PreferencePair:
    question: str
    chosen: str
    rejected: str
    kind: str


def _corrupt_arithmetic(problem: ReasoningProblem, rng: random.Random) -> str:
    correct = int(problem.answer)
    delta = rng.choice([d for d in range(-5, 6) if d != 0])
    wrong = correct + delta
    # same surface format as the correct solution, just a different number
    prefix = problem.solution.rsplit("=", 1)[0]
    return f"{prefix}= {wrong}"


def _corrupt_linear_equation(problem: ReasoningProblem, rng: random.Random) -> str:
    correct = int(problem.answer)
    delta = rng.choice([d for d in range(-5, 6) if d != 0])
    wrong = correct + delta
    lines = problem.solution.split("\n")
    lines[-1] = f"x = {wrong}"  # right procedure, wrong final step -- a realistic model error
    return "\n".join(lines)


def _corrupt_comparison(problem: ReasoningProblem, rng: random.Random) -> str:
    flipped = "No" if problem.answer == "Yes" else "Yes"
    # keep the same facts, just state the opposite (and logically inconsistent) conclusion
    facts = problem.solution.split("Since")[0].strip()
    return f"{facts} The answer is {flipped}."


_CORRUPTORS = {
    "arithmetic": _corrupt_arithmetic,
    "linear_equation": _corrupt_linear_equation,
    "comparison": _corrupt_comparison,
}


def make_preference_pair(problem: ReasoningProblem, rng: random.Random) -> PreferencePair:
    rejected = _CORRUPTORS[problem.kind](problem, rng)
    return PreferencePair(question=problem.question, chosen=problem.solution, rejected=rejected, kind=problem.kind)


def generate_preference_dataset(
    n: int, seed: int = 0, kinds: Optional[List[str]] = None
) -> List[PreferencePair]:
    rng = random.Random(seed)
    problems = generate_dataset(n, seed=seed, kinds=kinds)
    pairs = []
    for problem in problems:
        pair = make_preference_pair(problem, rng)
        # self-check at generation time, same discipline as Phase 5: never
        # emit a pair whose "chosen" doesn't actually verify as correct or
        # whose "rejected" doesn't actually verify as incorrect
        if not grade(problem, pair.chosen).correct:
            continue
        if grade(problem, pair.rejected).correct:
            continue
        pairs.append(pair)
    return pairs
