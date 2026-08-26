"""
Phase 12D test suite: normalizing verified teacher explanations into one
consistent presentation style.

Run with: pytest tests/test_normalize.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teachers.normalize import (  # noqa: E402
    ensure_explicit_answer,
    looks_truncated,
    normalize_explanation,
    strip_formatting,
)


# --------------------------------------------------------------------------
# strip_formatting
# --------------------------------------------------------------------------


def test_strip_formatting_removes_bold():
    assert strip_formatting("The **answer** is 4.") == "The answer is 4."


def test_strip_formatting_removes_italic():
    assert strip_formatting("This is *important* text.") == "This is important text."


def test_strip_formatting_removes_headers():
    assert strip_formatting("## Step 1\nDo the thing.") == "Step 1\nDo the thing."


def test_strip_formatting_removes_latex_display():
    assert strip_formatting("So $$50 - 44 = 6$$ is the result.") == "So 50 - 44 = 6 is the result."


def test_strip_formatting_removes_latex_inline():
    assert strip_formatting("We get $x = 5$ here.") == "We get x = 5 here."


def test_strip_formatting_removes_latex_paren_delimiters():
    assert strip_formatting(r"The equation \(10x + 3 = 203\) simplifies.") == "The equation 10x + 3 = 203 simplifies."


def test_strip_formatting_removes_latex_bracket_delimiters():
    assert strip_formatting(r"\[6x = -84\] so x = -14.") == "6x = -84 so x = -14."


def test_strip_formatting_collapses_excess_blank_lines():
    assert strip_formatting("A\n\n\n\n\nB") == "A\n\nB"


def test_strip_formatting_collapses_excess_spaces():
    assert strip_formatting("A    B") == "A B"


def test_strip_formatting_preserves_plain_text():
    text = "6x + 20 = -82\n6x = -102\nx = -17"
    assert strip_formatting(text) == text


# --------------------------------------------------------------------------
# looks_truncated
# --------------------------------------------------------------------------


def test_looks_truncated_true_for_midsentence_cutoff():
    assert looks_truncated("To get x by itself, we need to get rid of the -11. We do this by")


def test_looks_truncated_false_for_proper_ending():
    assert not looks_truncated("The answer is 6.")


def test_looks_truncated_false_for_closing_paren():
    assert not looks_truncated("(the answer is 6)")


def test_looks_truncated_true_for_empty_string():
    assert looks_truncated("")
    assert looks_truncated("   ")


# --------------------------------------------------------------------------
# ensure_explicit_answer
# --------------------------------------------------------------------------


def test_ensure_explicit_answer_appends_when_missing():
    result = ensure_explicit_answer("6x = -102 so x = -17", "-17")
    assert result.endswith("The answer is -17.")


def test_ensure_explicit_answer_leaves_existing_statement_alone():
    text = "So the answer is 6, confirmed by substitution."
    assert ensure_explicit_answer(text, "6") == text


def test_ensure_explicit_answer_case_insensitive_detection():
    text = "Therefore the Answer Is 6."
    assert ensure_explicit_answer(text, "6") == text


# --------------------------------------------------------------------------
# normalize_explanation: the full pipeline
# --------------------------------------------------------------------------


def test_normalize_explanation_drops_truncated_text():
    truncated = "To get x by itself, we need to get rid of the -11. We do this by"
    assert normalize_explanation(truncated, "5") is None


def test_normalize_explanation_strips_and_ensures_answer():
    text = "We compute **50 - 44 = 6**."
    result = normalize_explanation(text, "6")
    assert result is not None
    assert "**" not in result
    assert "answer is 6" in result.lower()


def test_normalize_explanation_does_not_duplicate_existing_answer_statement():
    text = "Step 1: subtract. The answer is 6."
    result = normalize_explanation(text, "6")
    assert result.lower().count("answer is") == 1


def test_normalize_explanation_real_command_r_truncation_example():
    """Regression test using an actual shape of truncated output found in
    the Phase 12A dataset (command-r7b, ~19% of its accepted examples)."""
    text = (
        "To solve this problem, we need to follow the order of operations (PEMDAS):\n\n"
        "1. **P**arentheses (none in this case)\n2. **E**xponents (none in this case)\n"
        "3. **M**ultiplication and Division (none in this case)\n4. **A**ddition and Subtraction\n\n"
        "The problem is simply 18 + 3, so we add the two numbers:\n\n18"
    )
    assert normalize_explanation(text, "21") is None
