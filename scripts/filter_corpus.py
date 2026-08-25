"""
Apply the basic content filter (data/content_filter.py) to a raw corpus
file and write the cleaned result.

Usage:
    python scripts/filter_corpus.py --in data/pretrain_corpus_raw.txt \
        --out data/pretrain_corpus_clean.txt
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.content_filter import filter_text  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--out", dest="out_path", type=str, required=True)
    args = parser.parse_args()

    text = Path(args.in_path).read_text(encoding="utf-8")
    cleaned, n_total, n_dropped = filter_text(text)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(cleaned, encoding="utf-8")

    print(f"Paragraphs: {n_total:,} total, {n_dropped:,} dropped "
          f"({n_dropped / max(1, n_total) * 100:.3f}%)")
    print(f"Chars: {len(text):,} -> {len(cleaned):,}")
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
