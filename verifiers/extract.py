"""Pull a final answer out of free-form model output text."""
import re
from typing import Optional

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_EXPLICIT_ANSWER_RE = re.compile(r"answer\s*(?:is|:)\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def extract_final_number(text: str) -> Optional[str]:
    """Prefer an explicit 'answer is X' / 'answer: X' marker if present
    (models often state their conclusion this way); otherwise fall back to
    the last bare number in the text, since a worked solution typically
    ends with the final value (e.g. '... so x = 5')."""
    explicit = _EXPLICIT_ANSWER_RE.findall(text)
    if explicit:
        return explicit[-1]
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def extract_yes_no(text: str) -> Optional[str]:
    matches = _YES_NO_RE.findall(text)
    return matches[-1].capitalize() if matches else None


def extract_answer(text: str, kind: str) -> Optional[str]:
    if kind == "comparison":
        return extract_yes_no(text)
    return extract_final_number(text)
