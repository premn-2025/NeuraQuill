"""
Sample text from a trained NeuraQuill checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/neuraquill-10m.pt \
        --tokenizer tokenizer/artifacts --prompt "Once upon a time" \
        --max-new-tokens 200 --temperature 0.8 --top-k 40
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
    parser.add_argument("--tokenizer", type=str, default="tokenizer/artifacts")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, step, best_val_loss, config = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"Loaded checkpoint from step {step} (best_val_loss={best_val_loss:.4f}) on {device}")

    tok = ByteLevelBPETokenizer.load(args.tokenizer)

    prompt = args.prompt or "<|endoftext|>"
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k
    )
    text = tok.decode(out[0].tolist())
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


if __name__ == "__main__":
    main()
