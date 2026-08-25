"""Grade a candidate solution against a ReasoningProblem's known-correct
answer, and turn that into a scalar reward -- the piece that lets training
scale past hand-labeled data (Phase 6 DPO / any future RL pass reads
`reward` directly)."""
from dataclasses import dataclass
from typing import Optional

from .extract import extract_answer
from .generators import ReasoningProblem


@dataclass
class GradeResult:
    correct: bool
    reward: float
    extracted_answer: Optional[str]


def _answers_match(extracted: str, expected: str, kind: str) -> bool:
    if kind == "comparison":
        return extracted.strip().lower() == expected.strip().lower()
    try:
        return abs(float(extracted) - float(expected)) < 1e-6
    except ValueError:
        return False


def grade(problem: ReasoningProblem, model_output: str) -> GradeResult:
    extracted = extract_answer(model_output, problem.kind)
    if extracted is None:
        return GradeResult(correct=False, reward=0.0, extracted_answer=None)
    correct = _answers_match(extracted, problem.answer, problem.kind)
    return GradeResult(correct=correct, reward=1.0 if correct else 0.0, extracted_answer=extracted)
