"""
One-shot chat inference against an SFT'd NeuraQuill checkpoint.

Usage:
    python scripts/chat.py --checkpoint checkpoints/neuraquill-10m-sft.pt \
        --tokenizer tokenizer/artifacts_chat --prompt "What is the capital of France?"
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/artifacts_chat")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--system", type=str, default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, step, best_val_loss, config = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"Loaded checkpoint from step {step} (best_val_loss={best_val_loss:.4f}) on {device}")

    tok = ByteLevelBPETokenizer.load(args.tokenizer)
    eot_id = tok.vocab["<|endoftext|>"]

    prompt_text = (
        f"<|system|>{args.system}<|endoftext|>"
        f"<|user|>{args.prompt}<|endoftext|>"
        f"<|assistant|>"
    )
    ids = tok.encode(prompt_text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, stop_token_ids={eot_id},
    )
    reply_ids = out[0, len(ids):].tolist()
    if reply_ids and reply_ids[-1] == eot_id:
        reply_ids = reply_ids[:-1]
    reply = tok.decode(reply_ids)

    print(f"\nUser: {args.prompt}")
    print(f"Assistant: {reply}")


if __name__ == "__main__":
    main()
