"""
Basic content filter for the pretraining corpus.

This is intentionally simple: a curated blocklist of slurs/severe
profanity, matched with word boundaries (case-insensitive), applied at
*paragraph* granularity -- a paragraph containing a blocked term is
dropped, the rest of the book is kept. This is NOT a general toxicity
classifier (that's Phase 5's job per PLAN.md) -- it's a first, cheap gate
to stop the most clearly unacceptable terms from passing straight through
into training data, which a plain "strip the Gutenberg license header"
step does nothing about.

Known limitations (be aware of these, don't oversell this filter):
  - word-list based -> no context awareness, no coverage of every slur or
    of profanity used non-abusively; a real pipeline would layer a trained
    toxicity/PII classifier on top of this, not rely on a list.
  - paragraph-level granularity -> can still remove non-offensive
    sentences that happen to share a paragraph with an offending one, and
    conversely can't catch a slur split across paragraph-adjacent context.
  - the blocklist here is short and focused on unambiguous slurs, not an
    exhaustive profanity list -- expand it deliberately as needed.
"""
import re
from typing import List, Tuple

# Deliberately short and focused on unambiguous, high-confidence slurs
# rather than an exhaustive profanity list -- easy to extend.
BLOCKLIST: List[str] = [
    "nigger",
    "nigga",
    "kike",
    "spic",
    "chink",
    "gook",
    "wetback",
    "coon",
    "faggot",
    "retard",
    "redskin",
    "tranny",
]

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BLOCKLIST) + r")\b", re.IGNORECASE
)


def paragraph_contains_blocked_term(paragraph: str) -> bool:
    return bool(_PATTERN.search(paragraph))


def filter_text(text: str) -> Tuple[str, int, int]:
    """Split `text` on blank-line paragraph boundaries and drop any
    paragraph containing a blocklisted term.

    Returns (filtered_text, n_paragraphs_total, n_paragraphs_dropped).
    """
    paragraphs = re.split(r"\n\s*\n", text)
    kept = []
    dropped = 0
    for p in paragraphs:
        if paragraph_contains_blocked_term(p):
            dropped += 1
            continue
        kept.append(p)
    return "\n\n".join(kept), len(paragraphs), dropped
