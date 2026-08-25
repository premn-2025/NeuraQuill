"""
Phase 4 data prep: load Dolly-15k, content-filter it, format it with the
chat template, extend the pretraining tokenizer with chat special tokens,
resize the pretrained NeuraQuill-10M checkpoint's embeddings to match, and
tokenize everything into a fixed-shape SFT dataset.

Usage:
    python scripts/prepare_sft_data.py \
        --data data/dolly15k.jsonl \
        --base-tokenizer tokenizer/artifacts \
        --base-checkpoint checkpoints/neuraquill-10m.pt \
        --tokenizer-out tokenizer/artifacts_chat \
        --checkpoint-out checkpoints/neuraquill-10m-resized.pt \
        --dataset-out data/sft_dataset.npz
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from data.content_filter import paragraph_contains_blocked_term  # noqa: E402
from model.resize import resize_token_embeddings  # noqa: E402
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from training.chat_template import CHAT_SPECIAL_TOKENS, encode_conversation  # noqa: E402
from training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from training.sft_dataset import build_sft_dataset  # noqa: E402

SYSTEM_PROMPT = "You are a helpful assistant."


def load_conversations(path: str, block_size: int, tokenizer, max_examples: int = None):
    conversations = []
    n_skipped_filter = 0
    n_skipped_length = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            instruction = row["instruction"].strip()
            context = row.get("context", "").strip()
            response = row["response"].strip()
            if not instruction or not response:
                continue

            if (
                paragraph_contains_blocked_term(instruction)
                or paragraph_contains_blocked_term(context)
                or paragraph_contains_blocked_term(response)
            ):
                n_skipped_filter += 1
                continue

            user_content = instruction if not context else f"{instruction}\n\nContext: {context}"
            turns = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ]

            ids, _ = encode_conversation(tokenizer, turns)
            if len(ids) > block_size + 1:
                n_skipped_length += 1
                continue

            conversations.append(turns)
            if max_examples and len(conversations) >= max_examples:
                break

    print(f"Loaded {len(conversations):,} conversations "
          f"(skipped {n_skipped_filter} filtered, {n_skipped_length} too long)")
    return conversations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dolly15k.jsonl")
    parser.add_argument("--base-tokenizer", type=str, default="tokenizer/artifacts")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/neuraquill-10m.pt")
    parser.add_argument("--tokenizer-out", type=str, default="tokenizer/artifacts_chat")
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/neuraquill-10m-resized.pt")
    parser.add_argument("--dataset-out", type=str, default="data/sft_dataset.npz")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading base tokenizer from {args.base_tokenizer} ...")
    tok = ByteLevelBPETokenizer.load(args.base_tokenizer)
    old_vocab_size = tok.vocab_size
    new_ids = tok.add_special_tokens(CHAT_SPECIAL_TOKENS)
    print(f"Extended tokenizer vocab: {old_vocab_size} -> {tok.vocab_size} "
          f"(added {CHAT_SPECIAL_TOKENS} -> ids {new_ids})")
    tok.save(args.tokenizer_out)
    print(f"Saved extended tokenizer -> {args.tokenizer_out}")

    print(f"\nLoading base checkpoint from {args.base_checkpoint} ...")
    model, _, step, best_val_loss, config = load_checkpoint(args.base_checkpoint, device="cpu")
    print(f"Base model: step {step}, best_val_loss {best_val_loss:.4f}, "
          f"vocab_size {config.vocab_size}, block_size {config.block_size}")

    resize_token_embeddings(model, tok.vocab_size)
    print(f"Resized model vocab_size -> {model.config.vocab_size}")

    import torch
    from training.optim import configure_optimizer

    fresh_optimizer = configure_optimizer(model, weight_decay=0.1, lr=1e-4)
    save_checkpoint(args.checkpoint_out, model, fresh_optimizer, step=0, best_val_loss=float("inf"), config=model.config)
    print(f"Saved resized checkpoint -> {args.checkpoint_out}")

    print(f"\nLoading + filtering + formatting {args.data} ...")
    conversations = load_conversations(args.data, config.block_size, tok, max_examples=args.max_examples)

    print("Tokenizing SFT dataset ...")
    X, Y = build_sft_dataset(tok, conversations, config.block_size)
    print(f"Dataset shape: X {X.shape}, Y {Y.shape}")

    n_val = max(1, int(len(X) * 0.02))
    perm = np.random.RandomState(0).permutation(len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Path(args.dataset_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.dataset_out,
        X_train=X[train_idx], Y_train=Y[train_idx],
        X_val=X[val_idx], Y_val=Y[val_idx],
    )
    print(f"Saved dataset -> {args.dataset_out} "
          f"({len(train_idx):,} train / {len(val_idx):,} val examples)")


if __name__ == "__main__":
    main()
