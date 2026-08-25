"""
Phase 7 data prep: extend the chat tokenizer with the tool-call special
tokens, resize a base checkpoint's embeddings to match, and tokenize the
tool-use dataset (with tool-result spans correctly masked out of the loss).

Usage:
    python scripts/prepare_tool_sft_data.py \
        --data data/tool_use_dataset.jsonl \
        --base-tokenizer tokenizer/artifacts_chat \
        --base-checkpoint checkpoints/neuraquill-10m-reasoning.pt \
        --tokenizer-out tokenizer/artifacts_tools \
        --checkpoint-out checkpoints/neuraquill-10m-tools-resized.pt \
        --dataset-out data/tool_sft_dataset.npz
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from inference.tool_format import TOOL_SPECIAL_TOKENS  # noqa: E402
from model.resize import resize_token_embeddings  # noqa: E402
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from training.optim import configure_optimizer  # noqa: E402
from training.tool_sft import build_tool_sft_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/tool_use_dataset.jsonl")
    parser.add_argument("--base-tokenizer", type=str, default="tokenizer/artifacts_chat")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/neuraquill-10m-reasoning.pt")
    parser.add_argument("--tokenizer-out", type=str, default="tokenizer/artifacts_tools")
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/neuraquill-10m-tools-resized.pt")
    parser.add_argument("--dataset-out", type=str, default="data/tool_sft_dataset.npz")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading base tokenizer from {args.base_tokenizer} ...")
    tok = ByteLevelBPETokenizer.load(args.base_tokenizer)
    old_vocab_size = tok.vocab_size
    new_ids = tok.add_special_tokens(TOOL_SPECIAL_TOKENS)
    print(f"Extended tokenizer vocab: {old_vocab_size} -> {tok.vocab_size} "
          f"(added {TOOL_SPECIAL_TOKENS} -> ids {new_ids})")
    tok.save(args.tokenizer_out)

    print(f"\nLoading base checkpoint from {args.base_checkpoint} ...")
    model, _, step, best_val_loss, config = load_checkpoint(args.base_checkpoint, device="cpu")
    print(f"Base model: step {step}, best_val_loss {best_val_loss:.4f}, vocab_size {config.vocab_size}")
    resize_token_embeddings(model, tok.vocab_size)
    print(f"Resized model vocab_size -> {model.config.vocab_size}")

    fresh_optimizer = configure_optimizer(model, weight_decay=0.1, lr=1e-4)
    save_checkpoint(args.checkpoint_out, model, fresh_optimizer, step=0, best_val_loss=float("inf"), config=model.config)
    print(f"Saved resized checkpoint -> {args.checkpoint_out}")

    print(f"\nLoading {args.data} ...")
    conversations = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            conversations.append(json.loads(line)["turns"])
    print(f"Loaded {len(conversations):,} conversations")

    X, Y = build_tool_sft_dataset(tok, conversations, config.block_size)
    print(f"Dataset shape: X {X.shape}")

    n_val = max(1, int(len(X) * args.val_fraction))
    perm = np.random.RandomState(0).permutation(len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Path(args.dataset_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.dataset_out,
        X_train=X[train_idx], Y_train=Y[train_idx],
        X_val=X[val_idx], Y_val=Y[val_idx],
    )
    print(f"Saved -> {args.dataset_out} ({len(train_idx):,} train / {len(val_idx):,} val)")


if __name__ == "__main__":
    main()
