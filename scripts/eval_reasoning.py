"""
Real-world Phase 5 check: generate reasoning problems, ask the SFT'd
NeuraQuill checkpoint to solve them via chat-style inference, and grade its
answers with the verifier. Reports accuracy by problem kind.

Usage:
    python scripts/eval_reasoning.py --checkpoint checkpoints/neuraquill-10m-sft.pt \
        --tokenizer tokenizer/artifacts_chat --n 30
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
from verifiers import generate_dataset, grade  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/artifacts_chat")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--show", type=int, default=5, help="print this many full Q/A/verdict examples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, step, best_val_loss, config = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"Loaded checkpoint step {step} (best_val_loss={best_val_loss:.4f}) on {device}")

    tok = ByteLevelBPETokenizer.load(args.tokenizer)
    eot_id = tok.vocab["<|endoftext|>"]

    dataset = generate_dataset(args.n, seed=args.seed)

    results = []
    for problem in dataset:
        prompt_text = (
            "<|system|>You are a helpful assistant that solves problems step by step "
            "and clearly states the final answer.<|endoftext|>"
            f"<|user|>{problem.question}<|endoftext|>"
            "<|assistant|>"
        )
        ids = tok.encode(prompt_text)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=40, stop_token_ids={eot_id},
        )
        reply_ids = out[0, len(ids):].tolist()
        if reply_ids and reply_ids[-1] == eot_id:
            reply_ids = reply_ids[:-1]
        reply = tok.decode(reply_ids)

        result = grade(problem, reply)
        results.append((problem, reply, result))

    n_correct = sum(1 for _, _, r in results if r.correct)
    print(f"\nOverall accuracy: {n_correct}/{len(results)} ({n_correct / len(results) * 100:.1f}%)")

    by_kind = {}
    for problem, _, r in results:
        s = by_kind.setdefault(problem.kind, [0, 0])
        s[0] += int(r.correct)
        s[1] += 1
    for kind, (correct, total) in by_kind.items():
        print(f"  {kind:16s}: {correct}/{total} ({correct / total * 100:.1f}%)")

    print(f"\n--- {min(args.show, len(results))} example transcripts ---")
    for problem, reply, r in results[: args.show]:
        print(f"\nQ: {problem.question}")
        print(f"Model: {reply}")
        print(f"Expected: {problem.answer} | Extracted: {r.extracted_answer} | Correct: {r.correct}")


if __name__ == "__main__":
    main()
