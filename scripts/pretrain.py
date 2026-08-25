"""
CLI entry point for Phase 3: pretrain NeuraQuill-10M from random initialization
on the tokenized corpus.

Usage:
    python scripts/pretrain.py --config configs/10m.yaml \
        --tokens data/pretrain_tokens.bin --max-steps 5000
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from model import GPT, load_config  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from training.dataset import load_token_memmap, split_train_val  # noqa: E402
from training.optim import configure_optimizer  # noqa: E402
from training.pretrain import TrainConfig, train  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/10m.yaml")
    parser.add_argument("--tokens", type=str, default="data/pretrain_tokens.bin")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/neuraquill-10m.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="accumulate gradients over this many micro-batches per optimizer step "
             "(effective batch size = batch_size * grad_accum_steps), for training "
             "larger configs within limited VRAM",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = load_token_memmap(args.tokens)
    train_data, val_data = split_train_val(data, val_fraction=0.02)
    print(f"Tokens: {len(data):,} total ({len(train_data):,} train / {len(val_data):,} val)")

    start_step = 0
    optimizer = None
    if args.resume and Path(args.checkpoint).exists():
        model, optimizer, start_step, best_val, config = load_checkpoint(args.checkpoint, device=device)
        optimizer = configure_optimizer(model, 0.1, args.max_lr)  # fresh optimizer, model weights kept
        print(f"Resumed from {args.checkpoint} at step {start_step} (best_val_loss={best_val:.4f})")
    else:
        model = GPT(config)
        print(f"New model: {model.num_parameters():,} params")

    train_cfg = TrainConfig(
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        block_size=config.block_size,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        eval_interval=args.eval_interval,
        checkpoint_path=args.checkpoint,
        device=device,
        grad_accum_steps=args.grad_accum_steps,
    )

    train(model, config, train_data, val_data, train_cfg, start_step=start_step, optimizer=optimizer)


if __name__ == "__main__":
    main()
