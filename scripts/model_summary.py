"""
Print a parameter-count / shape summary for a given model config, and run
one forward pass on random input as a smoke test.

Usage:
    python scripts/model_summary.py --config configs/10m.yaml
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import GPT, load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/10m.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model = GPT(config)

    n_params = model.num_parameters()
    n_params_non_emb = model.num_parameters(non_embedding=True)
    print(f"Config: {config}")
    print(f"Total parameters:              {n_params:,}  ({n_params / 1e6:.2f}M)")
    print(f"Non-embedding parameters:      {n_params_non_emb:,}  ({n_params_non_emb / 1e6:.2f}M)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = torch.randint(0, config.vocab_size, (2, 16), device=device)
    logits, loss = model(x, targets=x)
    print(f"\nDevice: {device}")
    print(f"Forward pass OK. logits shape: {tuple(logits.shape)}, loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
