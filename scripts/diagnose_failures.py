"""
Phase 12C: a failure taxonomy for a reasoning checkpoint -- classify
*where* the accuracy gap actually comes from before generating more
training data to try to close it blindly.

Usage:
    python scripts/diagnose_failures.py --checkpoint checkpoints/neuraquill-100m-reasoning.pt \
        --tokenizer tokenizer/artifacts_chat --n 100 --seed 2024 --out data/diagnostic_run.json
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from verifiers import generate_dataset, grade  # noqa: E402

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step by step and "
    "clearly states the final answer."
)


def has_repetition(text: str, n: int = 4) -> bool:
    """Cheap degenerate-loop detector: any n-word chunk immediately
    followed by an identical n-word chunk."""
    words = text.split()
    # +1 so a text that is *exactly* two adjacent n-word chunks (len == 2n)
    # still gets checked at i=0 -- range(len(words) - 2*n) would exclude
    # that boundary case entirely
    for i in range(len(words) - 2 * n + 1):
        if words[i : i + n] == words[i + n : i + 2 * n]:
            return True
    return False


def classify_failure(expected: str, extracted, reply: str) -> str:
    if extracted is None:
        return "no_answer_extracted"
    if has_repetition(reply):
        return "degenerate_repetition"
    try:
        diff = abs(float(extracted) - float(expected))
    except (ValueError, TypeError):
        return "unparseable_mismatch"
    return "close_miss (<=5)" if diff <= 5 else "wild_miss (>5)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/artifacts_chat")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--out", type=str, default="data/diagnostic_run.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, step, best_val, config = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    tok = ByteLevelBPETokenizer.load(args.tokenizer)
    eot_id = tok.vocab["<|endoftext|>"]

    dataset = generate_dataset(args.n, seed=args.seed)
    results = []
    for problem in dataset:
        prompt = (
            f"<|system|>{SYSTEM_PROMPT}<|endoftext|>"
            f"<|user|>{problem.question}<|endoftext|><|assistant|>"
        )
        ids = tok.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=0.3, top_k=40, stop_token_ids={eot_id})
        reply_ids = out[0, len(ids):].tolist()
        if reply_ids and reply_ids[-1] == eot_id:
            reply_ids = reply_ids[:-1]
        reply = tok.decode(reply_ids)
        result = grade(problem, reply)
        results.append({
            "question": problem.question, "kind": problem.kind, "expected": problem.answer,
            "reply": reply, "extracted": result.extracted_answer, "correct": result.correct,
        })

    n_correct = sum(r["correct"] for r in results)
    print(f"Accuracy: {n_correct}/{args.n} ({n_correct / args.n * 100:.1f}%)")

    failures = [r for r in results if not r["correct"]]
    print(f"\nFailures: {len(failures)}/{args.n}")

    by_kind = {}
    by_class = {}
    for r in failures:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        cls = classify_failure(r["expected"], r["extracted"], r["reply"])
        by_class[cls] = by_class.get(cls, 0) + 1

    print(f"\nFailures by kind: {by_kind}")
    print(f"Failures by class: {by_class}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
