"""
Phase 12D: apply teachers/normalize.py to a verified teacher dataset --
strip markdown/LaTeX formatting, drop examples that still look truncated
after cleanup, and guarantee an explicit answer statement on what
remains. Re-verifies every kept example against the known-correct answer
after normalization (should never fail, since normalization only trims/
appends around content -- but re-checked rather than assumed).

Usage:
    python scripts/normalize_teacher_data.py --data data/teacher_reasoning_dataset_large.jsonl \
        --out data/teacher_reasoning_dataset_normalized.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teachers.normalize import normalize_explanation  # noqa: E402
from verifiers.generators import ReasoningProblem  # noqa: E402
from verifiers.reward import grade  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} examples from {args.data}")

    kept = []
    n_dropped_truncated = 0
    n_dropped_reverify_failed = 0
    by_source_dropped = {}

    for row in rows:
        question = row["turns"][1]["content"]
        raw_answer_text = row["turns"][-1]["content"]
        answer = row["answer"]
        kind = row["kind"]

        normalized = normalize_explanation(raw_answer_text, answer)
        if normalized is None:
            n_dropped_truncated += 1
            src = row.get("source_model", "unknown")
            by_source_dropped[src] = by_source_dropped.get(src, 0) + 1
            continue

        # re-verify after normalization -- should always still be correct
        # since normalization only strips formatting/appends a statement,
        # but checked rather than assumed
        problem = ReasoningProblem(question=question, answer=answer, solution="", kind=kind)
        if not grade(problem, normalized).correct:
            n_dropped_reverify_failed += 1
            continue

        new_row = dict(row)
        new_row["turns"] = list(row["turns"])
        new_row["turns"][-1] = {"role": "assistant", "content": normalized}
        kept.append(new_row)

    print(f"Dropped (still truncated after cleanup): {n_dropped_truncated}")
    if by_source_dropped:
        print(f"  by source: {by_source_dropped}")
    print(f"Dropped (failed re-verification after normalization): {n_dropped_reverify_failed}")
    print(f"Kept: {len(kept)}/{len(rows)} ({len(kept) / len(rows) * 100:.1f}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
