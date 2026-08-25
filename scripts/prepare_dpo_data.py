"""
Tokenize a preference-pair JSONL (from generate_preference_data.py) into
a fixed-shape DPO dataset .npz.

Usage:
    python scripts/prepare_dpo_data.py --data data/preference_dataset.jsonl \
        --tokenizer tokenizer/artifacts_chat --block-size 512 \
        --out data/dpo_dataset.npz
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
from training.dpo_dataset import build_dpo_dataset  # noqa: E402


class _Pair:
    def __init__(self, question, chosen, rejected):
        self.question, self.chosen, self.rejected = question, chosen, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/artifacts_chat")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    tok = ByteLevelBPETokenizer.load(args.tokenizer)

    pairs = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pairs.append(_Pair(row["question"], row["chosen"], row["rejected"]))
    print(f"Loaded {len(pairs):,} pairs from {args.data}")

    Xc, Yc, Xr, Yr = build_dpo_dataset(tok, pairs, args.block_size)
    print(f"Dataset shape: Xc {Xc.shape}")

    n_val = max(1, int(len(Xc) * args.val_fraction))
    perm = np.random.RandomState(0).permutation(len(Xc))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        Xc_train=Xc[train_idx], Yc_train=Yc[train_idx], Xr_train=Xr[train_idx], Yr_train=Yr[train_idx],
        Xc_val=Xc[val_idx], Yc_val=Yc[val_idx], Xr_val=Xr[val_idx], Yr_val=Yr[val_idx],
    )
    print(f"Saved -> {args.out} ({len(train_idx):,} train / {len(val_idx):,} val)")


if __name__ == "__main__":
    main()
