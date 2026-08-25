"""
Generic prep step: tokenize any {"turns": [...]} JSONL (e.g. the output of
generate_reasoning_data.py) into a fixed-shape SFT dataset .npz, using an
already-chat-extended tokenizer (no vocab extension needed here).

Usage:
    python scripts/prepare_chat_jsonl.py --data data/reasoning_dataset.jsonl \
        --tokenizer tokenizer/artifacts_chat --block-size 512 \
        --out data/reasoning_sft_dataset.npz
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from training.sft_dataset import build_sft_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/artifacts_chat")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    tok = ByteLevelBPETokenizer.load(args.tokenizer)

    conversations = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            conversations.append(row["turns"])
    print(f"Loaded {len(conversations):,} conversations from {args.data}")

    X, Y = build_sft_dataset(tok, conversations, args.block_size)
    print(f"Dataset shape: X {X.shape}, Y {Y.shape}")

    n_val = max(1, int(len(X) * args.val_fraction))
    perm = np.random.RandomState(0).permutation(len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        X_train=X[train_idx], Y_train=Y[train_idx],
        X_val=X[val_idx], Y_val=Y[val_idx],
    )
    print(f"Saved -> {args.out} ({len(train_idx):,} train / {len(val_idx):,} val)")


if __name__ == "__main__":
    main()
