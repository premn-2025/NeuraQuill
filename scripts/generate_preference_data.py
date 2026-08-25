"""
Generate a synthetic preference dataset (chosen = verified-correct,
rejected = verified-incorrect) for DPO.

Usage:
    python scripts/generate_preference_data.py --n 3000 --out data/preference_dataset.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiers.preference import generate_preference_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/preference_dataset.jsonl")
    args = parser.parse_args()

    pairs = generate_preference_dataset(args.n, seed=args.seed)
    print(f"Generated {len(pairs):,} self-verified preference pairs (requested {args.n:,})")

    kinds = {}
    for p in pairs:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    print(f"Kind distribution: {kinds}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(
                {"question": p.question, "chosen": p.chosen, "rejected": p.rejected, "kind": p.kind},
                ensure_ascii=False,
            ) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
