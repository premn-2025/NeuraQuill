"""
Phase 5 test suite: synthetic reasoning problem generators + the
Python-sandbox arithmetic verifier + answer extraction + grading/reward.

Run with: pytest tests/test_verifiers.py -v
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiers.extract import extract_answer, extract_final_number, extract_yes_no  # noqa: E402
from verifiers.generators import (  # noqa: E402
    ReasoningProblem,
    generate_arithmetic_problem,
    generate_comparison_problem,
    generate_dataset,
    generate_linear_equation_problem,
    to_conversation,
)
from verifiers.reward import grade  # noqa: E402
from verifiers.sandbox import UnsafeExpressionError, safe_eval_arithmetic  # noqa: E402


# --------------------------------------------------------------------------
# sandbox: correctness on legitimate arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2 + 2", 4),
        ("10 - 3", 7),
        ("6 * 7", 42),
        ("10 / 4", 2.5),
        ("10 // 4", 2),
        ("10 % 3", 1),
        ("2 ** 10", 1024),
        ("-5 + 3", -2),
        ("(2 + 3) * 4", 20),
        ("2 + 3 * 4", 14),
        ("-(2 + 3)", -5),
        ("3.5 + 1.5", 5.0),
    ],
)
def test_sandbox_computes_correct_arithmetic(expr, expected):
    assert safe_eval_arithmetic(expr) == pytest.approx(expected)


# --------------------------------------------------------------------------
# sandbox: security -- must reject everything outside pure arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo pwned')",
        "open('secret.txt').read()",
        "a + 1",                      # name lookup
        "(1).__class__",              # attribute access
        "[1, 2, 3]",                  # list literal
        "{1: 2}",                     # dict literal
        "1 if True else 2",           # conditional
        "print(1)",                   # function call
        "1; 2",                       # statement, not an expression
        "lambda: 1",                  # lambda
        "[x for x in range(10)]",     # comprehension
        "True + 1",                   # bool disguised as int (excluded deliberately)
        "'hello'",                    # string constant
    ],
)
def test_sandbox_rejects_anything_non_arithmetic(expr):
    with pytest.raises(UnsafeExpressionError):
        safe_eval_arithmetic(expr)


def test_sandbox_rejects_syntax_errors():
    with pytest.raises(UnsafeExpressionError):
        safe_eval_arithmetic("2 +")


# --------------------------------------------------------------------------
# generators: self-verifying by construction -- cross-check each problem's
# recorded answer against an independent computation
# --------------------------------------------------------------------------


def test_arithmetic_problem_answer_matches_sandbox_recompute():
    rng = random.Random(1)
    for _ in range(200):
        p = generate_arithmetic_problem(rng)
        # re-derive the expression from the question text and recompute
        # independently, rather than trusting the generator's own math
        expr = p.question[len("What is ") : -1]
        assert str(safe_eval_arithmetic(expr)) == p.answer


def test_linear_equation_problem_is_algebraically_consistent():
    rng = random.Random(2)
    for _ in range(200):
        p = generate_linear_equation_problem(rng)
        # question is "Solve for x: {a}x + {b} = {c}" -- parse it back out
        # and verify a*answer + b == c independently via the sandbox
        body = p.question[len("Solve for x: ") :]
        lhs, rhs = body.split("=")
        c = int(rhs.strip())
        a_str, b_str = lhs.replace("x", " ").split("+")
        a, b = int(a_str.strip()), int(b_str.strip())
        assert safe_eval_arithmetic(f"{a} * {p.answer} + {b}") == c


def test_comparison_problem_answer_is_logically_consistent():
    rng = random.Random(3)
    for _ in range(200):
        p = generate_comparison_problem(rng)
        assert p.answer in ("Yes", "No")
        # the solution text must explicitly state the same answer as p.answer
        assert p.answer in p.solution


def test_generate_dataset_is_deterministic_given_seed():
    ds1 = generate_dataset(50, seed=42)
    ds2 = generate_dataset(50, seed=42)
    assert [p.question for p in ds1] == [p.question for p in ds2]
    assert [p.answer for p in ds1] == [p.answer for p in ds2]


def test_generate_dataset_respects_requested_kinds():
    ds = generate_dataset(30, seed=0, kinds=["arithmetic"])
    assert all(p.kind == "arithmetic" for p in ds)


def test_to_conversation_shape():
    p = ReasoningProblem(question="What is 2+2?", answer="4", solution="2+2=4", kind="arithmetic")
    turns = to_conversation(p)
    roles = [t["role"] for t in turns]
    assert roles == ["system", "user", "assistant"]
    assert turns[1]["content"] == p.question
    assert turns[2]["content"] == p.solution


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("5", "5"),
        ("The answer is 5.", "5"),
        ("answer: 5", "5"),
        ("First I get 3, then 7, so the answer is 12.", "12"),
        ("x = 5", "5"),
        ("no numbers here", None),
        ("-7", "-7"),
        ("3.14", "3.14"),
    ],
)
def test_extract_final_number(text, expected):
    assert extract_final_number(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yes, that is correct.", "Yes"),
        ("no, it is not.", "No"),
        ("I don't know", None),
        ("First no, wait, actually yes", "Yes"),
    ],
)
def test_extract_yes_no(text, expected):
    assert extract_yes_no(text) == expected


def test_extract_answer_dispatches_by_kind():
    assert extract_answer("The answer is 42.", "arithmetic") == "42"
    assert extract_answer("Yes.", "comparison") == "Yes"


# --------------------------------------------------------------------------
# grading / reward: the core requirement -- correctly grade known
# right/wrong solutions
# --------------------------------------------------------------------------


def test_grade_known_correct_and_incorrect_solutions():
    arithmetic = ReasoningProblem(question="What is 2 + 2?", answer="4", solution="2 + 2 = 4", kind="arithmetic")
    comparison = ReasoningProblem(
        question="A is taller than B. Is A taller than B?", answer="Yes", solution="...", kind="comparison"
    )
    cases = [
        (arithmetic, "Let me compute: 2 + 2 = 4. The answer is 4.", True),
        (arithmetic, "I think it's 5.", False),
        (arithmetic, "", False),
        (comparison, "Based on the facts, yes, A is taller.", True),
        (comparison, "No, that's not correct.", False),
        (comparison, "unclear", False),
    ]
    for problem, output, expected_correct in cases:
        result = grade(problem, output)
        assert result.correct == expected_correct
        assert result.reward == (1.0 if expected_correct else 0.0)


def test_grade_every_problems_own_solution_is_graded_correct():
    """The strongest end-to-end check: for a whole generated batch, each
    problem's own recorded step-by-step solution must grade as correct
    against its own answer -- if this ever fails, either the generator or
    the verifier has a bug."""
    dataset = generate_dataset(300, seed=7)
    for problem in dataset:
        result = grade(problem, problem.solution)
        assert result.correct, f"{problem.kind} problem graded its own solution wrong: {problem}"


def test_grade_handles_missing_answer_gracefully():
    p = ReasoningProblem(question="What is 1 + 1?", answer="2", solution="1 + 1 = 2", kind="arithmetic")
    result = grade(p, "I have no idea what the answer might be.")
    assert result.correct is False
    assert result.reward == 0.0
    assert result.extracted_answer is None


def test_grade_uses_float_tolerance():
    p = ReasoningProblem(question="What is 5 / 2?", answer="2.5", solution="5 / 2 = 2.5", kind="arithmetic")
    assert grade(p, "The answer is 2.5").correct
    assert grade(p, "The answer is 2.5000001").correct  # within tolerance
    assert not grade(p, "The answer is 2.6").correct
