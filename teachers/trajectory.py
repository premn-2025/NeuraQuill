"""
Rich per-(problem, teacher) records instead of flat question->answer
pairs -- keeps provenance, verification, and a disagreement signal, which
is what makes teacher data usable for more than one SFT pass (curriculum
selection, future reward modeling, auditing a specific teacher's error
patterns, etc.).
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

PROMPT_VERSION = "v1"


@dataclass
class Trajectory:
    question: str
    teacher: str
    prompt_version: str
    answer: str
    reasoning_type: str
    tool_calls: List[Any]
    verifier_result: bool
    extracted_answer: Optional[str]
    quality_score: float
    disagreement_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_disagreement_scores(trajectories: List[Trajectory]) -> None:
    """Mutates each trajectory's disagreement_score in place. Defined per
    *question*: 1 - (fraction of teachers who answered that exact
    question correctly), the same value assigned to every trajectory for
    that question. High disagreement flags questions the teacher pool
    broadly struggled with or split on -- a difficulty/ambiguity signal,
    useful even though ground truth is already known here via the
    verifier (it isn't a substitute for verifier_result, it's additional
    context: "everyone missed this" reads differently from "only this
    teacher missed it")."""
    by_question: Dict[str, List[Trajectory]] = {}
    for t in trajectories:
        by_question.setdefault(t.question, []).append(t)

    for group in by_question.values():
        n_correct = sum(1 for t in group if t.verifier_result)
        disagreement = 1.0 - (n_correct / len(group))
        for t in group:
            t.disagreement_score = disagreement
