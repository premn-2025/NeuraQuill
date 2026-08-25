"""
Verification for agent outputs. Reasoning subtasks reuse Phase 5's
grader directly (verifiable by construction). Knowledge subtasks have no
general-purpose fact-checker in this project, so verification there is
intentionally weaker and scoped: does the answer actually contain the
expected fact substring -- i.e. is it grounded in what was retrieved,
for scenarios where the expected fact is known up front (scripted tests,
evaluation harnesses). This is not a claim of general factual
correctness checking.
"""
from verifiers.generators import ReasoningProblem
from verifiers.reward import GradeResult, grade


def verify_reasoning(problem: ReasoningProblem, answer: str) -> GradeResult:
    return grade(problem, answer)


def verify_knowledge_grounded(answer: str, expected_fact_substring: str) -> bool:
    return expected_fact_substring.lower() in answer.lower()
