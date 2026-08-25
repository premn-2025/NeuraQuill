"""
CLI entry point for Phase 4: instruction-tune NeuraQuill-10M starting from the
Phase 3 pretrained (vocab-resized) checkpoint.

Usage:
    python scripts/sft_train.py --dataset data/sft_dataset.npz \
        --checkpoint checkpoints/neuraquill-10m-resized.pt \
        --out checkpoints/neuraquill-10m-sft.pt --max-steps 3000
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from training.checkpoint import load_checkpoint  # noqa: E402
from training.pretrain import TrainConfig  # noqa: E402
from training.sft import train_sft  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/sft_dataset.npz")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/neuraquill-10m-resized.pt")
    parser.add_argument("--out", type=str, default="checkpoints/neuraquill-10m-sft.pt")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--eval-interval", type=int, default=150)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = np.load(args.dataset)
    X_train, Y_train, X_val, Y_val = data["X_train"], data["Y_train"], data["X_val"], data["Y_val"]
    print(f"Dataset: {len(X_train):,} train / {len(X_val):,} val examples, block_size={X_train.shape[1]}")

    model, _, start_step, best_val, config = load_checkpoint(args.checkpoint, device=device)
    print(f"Loaded {args.checkpoint}: vocab_size={config.vocab_size}, {model.num_parameters():,} params")

    train_cfg = TrainConfig(
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        block_size=config.block_size,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        eval_interval=args.eval_interval,
        checkpoint_path=args.out,
        device=device,
    )

    train_sft(model, config, X_train, Y_train, X_val, Y_val, train_cfg)


if __name__ == "__main__":
    main()
