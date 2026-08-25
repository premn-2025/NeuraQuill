"""
Phase 3 (data quality) test suite: the basic content filter.

Run with: pytest tests/test_content_filter.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.content_filter import (  # noqa: E402
    BLOCKLIST,
    filter_text,
    paragraph_contains_blocked_term,
)


def test_paragraph_with_blocked_term_is_flagged():
    for term in BLOCKLIST:
        assert paragraph_contains_blocked_term(f"He used the word {term} in the story.")


def test_paragraph_without_blocked_term_is_not_flagged():
    assert not paragraph_contains_blocked_term("The quick brown fox jumps over the lazy dog.")


def test_matching_is_case_insensitive():
    assert paragraph_contains_blocked_term("He shouted NIGGER at the crowd.")
    assert paragraph_contains_blocked_term("He shouted Nigger at the crowd.")


def test_matching_uses_word_boundaries_no_false_positive_on_substrings():
    """A word-list filter without \\b boundaries would false-positive on
    innocent words that merely contain a blocked substring. None of our
    blocklist terms should fire inside these unrelated words."""
    safe_sentences = [
        "The assassin crept through the assembly hall.",
        "Chinchilla and chinstrap are unrelated words.",
        # 'coonhound' contains 'coon' only as a substring, not a standalone
        # word (no boundary between 'coon' and 'hound') -- must not match.
        "The coonhound followed the scent through the woods.",
    ]
    for s in safe_sentences:
        assert not paragraph_contains_blocked_term(s)


def test_filter_text_drops_only_offending_paragraphs():
    text = (
        "This is a perfectly fine paragraph about the weather.\n\n"
        "This paragraph contains the word nigger and should be dropped.\n\n"
        "This is another fine paragraph about gardening."
    )
    cleaned, n_total, n_dropped = filter_text(text)
    assert n_total == 3
    assert n_dropped == 1
    assert "weather" in cleaned
    assert "gardening" in cleaned
    assert "nigger" not in cleaned.lower()


def test_filter_text_preserves_clean_text_exactly():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    cleaned, n_total, n_dropped = filter_text(text)
    assert n_dropped == 0
    assert cleaned == text


def test_filter_text_empty_input():
    cleaned, n_total, n_dropped = filter_text("")
    assert cleaned == ""
    assert n_dropped == 0


def test_filter_text_all_paragraphs_offensive_yields_empty_output():
    text = "\n\n".join(f"A paragraph with {term} in it." for term in BLOCKLIST[:3])
    cleaned, n_total, n_dropped = filter_text(text)
    assert n_total == 3
    assert n_dropped == 3
    assert cleaned == ""
