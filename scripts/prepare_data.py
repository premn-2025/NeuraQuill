"""
Train the tokenizer on the pretraining corpus and tokenize it into a
binary token stream ready for training.

Usage:
    python scripts/prepare_data.py --corpus data/pretrain_corpus_raw.txt \
        --vocab-size 8192 --tokenizer-out tokenizer/artifacts \
        --tokens-out data/pretrain_tokens.bin
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.content_filter import filter_text  # noqa: E402
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from training.dataset import tokenize_and_save  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--tokenizer-out", type=str, default="tokenizer/artifacts")
    parser.add_argument("--tokens-out", type=str, default="data/pretrain_tokens.bin")
    parser.add_argument(
        "--no-filter", action="store_true",
        help="skip the content filter (data/content_filter.py) -- off by default, "
             "i.e. filtering runs unless you pass this",
    )
    args = parser.parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")
    print(f"Corpus: {len(text):,} chars from {args.corpus}")

    if not args.no_filter:
        text, n_total, n_dropped = filter_text(text)
        print(f"Content filter: dropped {n_dropped:,}/{n_total:,} paragraphs "
              f"({n_dropped / max(1, n_total) * 100:.3f}%) -> {len(text):,} chars remain")
    else:
        print("Content filter: SKIPPED (--no-filter)")

    print(f"Training tokenizer (vocab_size={args.vocab_size}) ...")
    tok = ByteLevelBPETokenizer.train(
        [text], vocab_size=args.vocab_size, special_tokens=("<|endoftext|>", "<|pad|>")
    )
    tok.save(args.tokenizer_out)
    print(f"Saved tokenizer -> {args.tokenizer_out} ({tok.vocab_size} tokens, {len(tok.merges)} merges)")

    print("Tokenizing corpus ...")
    n_tokens = tokenize_and_save(tok, text, args.tokens_out)
    print(f"Wrote {n_tokens:,} tokens -> {args.tokens_out} "
          f"({n_tokens * 2 / 1e6:.1f} MB as uint16); "
          f"compression: {len(text) / n_tokens:.2f} chars/token")


if __name__ == "__main__":
    main()
