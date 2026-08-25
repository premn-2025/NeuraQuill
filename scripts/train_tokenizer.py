"""
CLI entry point for Phase 1: train the byte-level BPE tokenizer on a corpus
and save the resulting vocab/merges to disk.

Usage:
    python scripts/train_tokenizer.py --corpus data/sample_corpus.txt \
        --vocab-size 1024 --out tokenizer/artifacts
"""
import argparse
import sys
from pathlib import Path

# Windows terminals default to cp1252, which can't print some of the
# printable-unicode byte markers (e.g. 'Ġ') used internally by the
# tokenizer's verbose merge logging -- force utf-8 stdout so this script
# works the same on Windows/macOS/Linux.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train NeuraQuill's byte-level BPE tokenizer")
    parser.add_argument("--corpus", type=str, required=True, help="Path to a text corpus file")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--out", type=str, default="tokenizer/artifacts")
    args = parser.parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")
    # one "text" per non-empty line keeps pretokenization from merging across
    # unrelated lines' whitespace in odd ways; blank lines act as separators
    lines = [line for line in text.split("\n")]
    corpus_texts = ["\n".join(lines)]

    print(f"Training on {len(text)} chars from {args.corpus} -> vocab_size={args.vocab_size}")
    tok = ByteLevelBPETokenizer.train(corpus_texts, vocab_size=args.vocab_size, verbose=True)
    tok.save(args.out)
    print(f"Saved tokenizer to {args.out} ({tok.vocab_size} tokens, {len(tok.merges)} merges)")

    sample = "The quick brown fox jumps over the lazy dog. def foo(): return 42"
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    print(f"\nSample: {sample!r}")
    print(f"Encoded ({len(ids)} tokens): {ids}")
    print(f"Decoded: {decoded!r}")
    print(f"Round-trip OK: {decoded == sample}")


if __name__ == "__main__":
    main()
