"""
Splits a (possibly compound) task into an ordered list of subtasks, e.g.
"What is 12 + 7, and what does the X200 weigh?" -> two independent
subtasks, one per worker.

Rule-based, same reasoning as the router: reliable task decomposition
isn't something a 10M-param model should be trusted to do freely at this
scale. Splitting on explicit connective phrases is simple and correct for
the compound-question shape this project's scripted scenarios use.
"""
import re
from typing import List

_SPLIT_RE = re.compile(r"\s*(?:,?\s+and\s+also\s+|,?\s+and\s+|;\s*)\s*", re.IGNORECASE)


def plan(task: str) -> List[str]:
    parts = [p.strip() for p in _SPLIT_RE.split(task) if p.strip()]
    return parts if parts else [task]
