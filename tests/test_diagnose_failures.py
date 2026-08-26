"""
Phase 12C test suite: the failure-classification logic in
scripts/diagnose_failures.py.

Run with: pytest tests/test_diagnose_failures.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_failures import classify_failure, has_repetition  # noqa: E402


def test_has_repetition_detects_repeated_chunk():
    text = "write down the multiplication write down the multiplication"
    assert has_repetition(text, n=4)


def test_has_repetition_detects_exact_boundary_case():
    """Regression test: a text that is *exactly* two adjacent n-word
    chunks (len == 2n) must still be detected at i=0 -- an off-by-one in
    the range excluded this boundary case entirely."""
    text = "a b c d a b c d"
    assert has_repetition(text, n=4)


def test_has_repetition_false_for_normal_text():
    text = "6x plus 20 equals negative 82 so 6x equals negative 102 so x equals negative 17"
    assert not has_repetition(text, n=4)


def test_has_repetition_false_for_short_text():
    assert not has_repetition("The answer is 4", n=4)


def test_classify_failure_no_answer_extracted():
    assert classify_failure("4", None, "I don't know") == "no_answer_extracted"


def test_classify_failure_degenerate_repetition_takes_priority_when_no_number():
    text = "write down the multiplication write down the multiplication"
    assert classify_failure("4", None, text) == "no_answer_extracted"  # no number at all -> extraction wins


def test_classify_failure_degenerate_repetition_with_a_number_present():
    text = "one two three four one two three four answer is 99"
    assert classify_failure("4", "99", text) == "degenerate_repetition"


def test_classify_failure_close_miss():
    assert classify_failure("10", "13", "10x = 13, so x = 13") == "close_miss (<=5)"


def test_classify_failure_wild_miss():
    assert classify_failure("10", "9999", "the answer is 9999") == "wild_miss (>5)"


def test_classify_failure_boundary_exactly_five_is_close():
    assert classify_failure("10", "15", "the answer is 15") == "close_miss (<=5)"


def test_classify_failure_boundary_six_is_wild():
    assert classify_failure("10", "16", "the answer is 16") == "wild_miss (>5)"


def test_classify_failure_unparseable_extraction():
    assert classify_failure("Yes", "maybe", "the answer is maybe") == "unparseable_mismatch"
