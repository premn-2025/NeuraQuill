"""
Generate a synthetic, self-verifying reasoning dataset and save it as
chat-formatted JSONL (ready to feed into SFT, same shape as Dolly).

Usage:
    python scripts/generate_reasoning_data.py --n 2000 --out data/reasoning_dataset.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiers import generate_dataset, grade  # noqa: E402
from verifiers.generators import to_conversation  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/reasoning_dataset.jsonl")
    args = parser.parse_args()

    dataset = generate_dataset(args.n, seed=args.seed)

    # sanity-check the whole batch against its own verifier before saving --
    # never ship generated data without exercising the grader on it
    n_self_correct = sum(1 for p in dataset if grade(p, p.solution).correct)
    print(f"Self-verification: {n_self_correct}/{len(dataset)} problems' own "
          f"solutions grade as correct")
    if n_self_correct != len(dataset):
        raise RuntimeError("generator/verifier mismatch -- refusing to save a bad dataset")

    kinds = {}
    for p in dataset:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    print(f"Kind distribution: {kinds}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in dataset:
            row = {"turns": to_conversation(p), "answer": p.answer, "kind": p.kind}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset):,} examples -> {out_path}")


if __name__ == "__main__":
    main()
