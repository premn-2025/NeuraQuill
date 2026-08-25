"""
CLI entry point for Phase 6: DPO-tune a checkpoint (typically the Phase 5
reasoning-SFT'd model) on a preference dataset.

Usage:
    python scripts/dpo_train.py --dataset data/dpo_dataset.npz \
        --checkpoint checkpoints/neuraquill-10m-reasoning.pt \
        --out checkpoints/neuraquill-10m-dpo.pt --max-steps 1000
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
from training.dpo import DPOTrainConfig, train_dpo  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/dpo_dataset.npz")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/neuraquill-10m-reasoning.pt")
    parser.add_argument("--out", type=str, default="checkpoints/neuraquill-10m-dpo.pt")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-lr", type=float, default=5e-6)
    parser.add_argument("--min-lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = np.load(args.dataset)
    print(f"Dataset: {len(data['Xc_train']):,} train / {len(data['Xc_val']):,} val pairs")

    policy, _, start_step, best_val, config = load_checkpoint(args.checkpoint, device=device)
    print(f"Loaded {args.checkpoint}: vocab_size={config.vocab_size}, {policy.num_parameters():,} params")

    train_cfg = DPOTrainConfig(
        max_steps=args.max_steps, warmup_steps=args.warmup_steps, batch_size=args.batch_size,
        max_lr=args.max_lr, min_lr=args.min_lr, beta=args.beta, eval_interval=args.eval_interval,
        checkpoint_path=args.out, device=device,
    )

    train_dpo(
        policy, config,
        data["Xc_train"], data["Yc_train"], data["Xr_train"], data["Yr_train"],
        data["Xc_val"], data["Yc_val"], data["Xr_val"], data["Yr_val"],
        train_cfg,
    )


if __name__ == "__main__":
    main()
