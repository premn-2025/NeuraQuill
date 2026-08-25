"""
Synthetic, automatically-verifiable reasoning problems: arithmetic, linear
equations, and transitive comparison ("A taller than B taller than C")
puzzles. Every generated problem's `answer` is computed by the sandbox
(or by direct logical construction for comparisons) at generation time, so
the dataset is self-verifying by construction -- no human labeling needed,
which is exactly what makes this kind of data cheap to generate at scale.
"""
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .sandbox import safe_eval_arithmetic

NAMES = ["Alice", "Bob", "Carol", "Dave", "Emma", "Frank", "Grace", "Henry"]


@dataclass
class ReasoningProblem:
    question: str
    answer: str    # canonical string form of the correct final answer
    solution: str  # a worked, step-by-step solution ending in the answer
    kind: str      # "arithmetic" | "linear_equation" | "comparison"


def generate_arithmetic_problem(rng: random.Random, max_value: int = 50) -> ReasoningProblem:
    op = rng.choice(["+", "-", "*"])
    a = rng.randint(1, max_value)
    b = rng.randint(1, max_value)
    expr = f"{a} {op} {b}"
    result = safe_eval_arithmetic(expr)
    return ReasoningProblem(
        question=f"What is {a} {op} {b}?",
        answer=str(result),
        solution=f"{a} {op} {b} = {result}",
        kind="arithmetic",
    )


def generate_linear_equation_problem(rng: random.Random, max_value: int = 20) -> ReasoningProblem:
    x = rng.randint(-max_value, max_value)
    a = rng.randint(1, 10)
    b = rng.randint(-max_value, max_value)
    c = safe_eval_arithmetic(f"{a} * {x} + {b}")
    return ReasoningProblem(
        question=f"Solve for x: {a}x + {b} = {c}",
        answer=str(x),
        solution=f"{a}x + {b} = {c}\n{a}x = {c - b}\nx = {x}",
        kind="linear_equation",
    )


def generate_comparison_problem(rng: random.Random, n_people: int = 3) -> ReasoningProblem:
    order = rng.sample(NAMES, n_people)  # order[0] is tallest ... order[-1] is shortest
    facts = [f"{order[i]} is taller than {order[i + 1]}." for i in range(len(order) - 1)]
    a, b = rng.sample(order, 2)
    is_taller = order.index(a) < order.index(b)
    answer = "Yes" if is_taller else "No"
    return ReasoningProblem(
        question=" ".join(facts) + f" Is {a} taller than {b}?",
        answer=answer,
        solution=(
            " ".join(facts)
            + f" Since {a} comes {'before' if is_taller else 'after'} {b} in the height "
            + f"ordering, the answer is {answer}."
        ),
        kind="comparison",
    )


GENERATORS: Dict[str, Callable[[random.Random], ReasoningProblem]] = {
    "arithmetic": generate_arithmetic_problem,
    "linear_equation": generate_linear_equation_problem,
    "comparison": generate_comparison_problem,
}


def generate_dataset(
    n: int, seed: int = 0, kinds: Optional[Sequence[str]] = None
) -> List[ReasoningProblem]:
    rng = random.Random(seed)
    kinds = list(kinds) if kinds else list(GENERATORS.keys())
    return [GENERATORS[rng.choice(kinds)](rng) for _ in range(n)]


def to_conversation(problem: ReasoningProblem) -> List[Dict[str, str]]:
    """Format a problem as chat turns (compatible with
    training.chat_template.encode_conversation), for use as SFT/RL data."""
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that solves problems step by step and clearly states the final answer.",
        },
        {"role": "user", "content": problem.question},
        {"role": "assistant", "content": problem.solution},
    ]
