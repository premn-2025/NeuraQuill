"""
Real-world Phase 7 check: same idea as eval_reasoning.py, but routes
generation through the actual ToolCallOrchestrator + a real CalculatorTool,
so any arithmetic the model hands off gets an exact answer instead of a
mentally-computed one.

Usage:
    python scripts/eval_tool_reasoning.py --checkpoint checkpoints/neuraquill-10m-tools.pt \
        --tokenizer tokenizer/artifacts_tools --n 60
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from inference.orchestrator import ToolCallOrchestrator  # noqa: E402
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from tools import default_registry  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from verifiers import generate_dataset, grade  # noqa: E402

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step by step and clearly "
    "states the final answer. For arithmetic, use the calculator tool instead of "
    "computing by hand."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/artifacts_tools")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, step, best_val_loss, config = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"Loaded checkpoint step {step} (best_val_loss={best_val_loss:.4f}) on {device}")

    tok = ByteLevelBPETokenizer.load(args.tokenizer)
    orchestrator = ToolCallOrchestrator(model, tok, default_registry(), max_tool_calls=2)

    dataset = generate_dataset(args.n, seed=args.seed)

    results = []
    for problem in dataset:
        prompt_text = (
            f"<|system|>{SYSTEM_PROMPT}<|endoftext|>"
            f"<|user|>{problem.question}<|endoftext|>"
            "<|assistant|>"
        )
        ids = tok.encode(prompt_text)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        reply = orchestrator.run(
            idx, max_new_tokens_per_turn=args.max_new_tokens, temperature=args.temperature, top_k=40
        )
        result = grade(problem, reply)
        results.append((problem, reply, result))

    n_correct = sum(1 for _, _, r in results if r.correct)
    print(f"\nOverall accuracy (with tool calling): {n_correct}/{len(results)} ({n_correct / len(results) * 100:.1f}%)")

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
