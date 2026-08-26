"""
Phase 12D: normalize verified teacher explanations into one consistent
presentation style before training on them again.

Two concrete, evidenced problems this targets (not a guess):
  1. Markdown (**bold**, ## headers) and LaTeX ($$...$$, \\(...\\)) syntax
     is common across teacher output but doesn't match this project's
     tokenizer training distribution (Gutenberg prose + plain
     instructions -- no markdown/LaTeX corpus), a plausible contributor
     to Phase 12B's coherence collapse.
  2. ~3% of Phase 12A's accepted examples (concentrated almost entirely
     in one teacher, command-r7b at 19%) were verified "correct" only
     because the "last number in the text" extraction fallback happened
     to land on the right value in a response that was actually cut off
     mid-thought (max_tokens truncation) -- a real data-quality bug, not
     a style issue. Content, not just format, was already broken for
     these; normalize_explanation drops them rather than patching over
     an interrupted reasoning chain with a bolted-on answer sentence.
"""
import re
from typing import Optional

_MARKDOWN_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MARKDOWN_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_LATEX_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\$(.+?)\$")
_LATEX_PAREN_RE = re.compile(r"\\\((.+?)\\\)")
_LATEX_BRACKET_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

ANSWER_STATEMENT_RE = re.compile(r"answer\s+is", re.IGNORECASE)


def strip_formatting(text: str) -> str:
    """Removes markdown bold/italic/headers and LaTeX math delimiters,
    keeping the enclosed content."""
    text = _LATEX_DISPLAY_RE.sub(lambda m: m.group(1).strip(), text)
    text = _LATEX_BRACKET_RE.sub(lambda m: m.group(1).strip(), text)
    text = _LATEX_PAREN_RE.sub(lambda m: m.group(1).strip(), text)
    text = _LATEX_INLINE_RE.sub(lambda m: m.group(1).strip(), text)
    text = _MARKDOWN_BOLD_RE.sub(r"\1", text)
    text = _MARKDOWN_ITALIC_RE.sub(r"\1", text)
    text = _MARKDOWN_HEADER_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def looks_truncated(text: str) -> bool:
    text = text.strip()
    if not text:
        return True
    return text[-1] not in ".!?)"


def ensure_explicit_answer(text: str, answer: str) -> str:
    """Appends an explicit 'The answer is X.' sentence if the text
    doesn't already state one, so a training target never depends on the
    'last number in the text' extraction fallback to happen to be
    correct."""
    if ANSWER_STATEMENT_RE.search(text):
        return text
    text = text.rstrip()
    if text and text[-1] not in ".!?":
        text += "."
    return f"{text} The answer is {answer}."


def normalize_explanation(text: str, answer: str) -> Optional[str]:
    """Full pipeline: strip formatting artifacts, drop (return None)
    rather than patch anything that still looks truncated afterward, and
    guarantee an explicit answer statement on what remains."""
    cleaned = strip_formatting(text)
    if looks_truncated(cleaned):
        return None
    return ensure_explicit_answer(cleaned, answer)
