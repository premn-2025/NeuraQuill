"""
Phase 12: run the Phase 10.5 eval categories against a candidate
checkpoint and decide, by explicit policy (evaluation/gate.py), whether
it should be adopted over a reference checkpoint/report -- not by
whether it "seems better."

Usage:
    python scripts/evaluate_and_gate.py \
        --candidate-checkpoint checkpoints/neuraquill-100m-teacher-reasoning.pt \
        --candidate-tokenizer tokenizer/artifacts_chat \
        --reference-report reports/neuraquill-100m-baseline.json \
        --category reasoning
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from evaluation.gate import flatten_baseline_report, gate_checkpoint  # noqa: E402
from scripts.baseline_eval import eval_reasoning_no_tools, eval_tools  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", type=str, required=True)
    parser.add_argument("--candidate-tokenizer", type=str, required=True)
    parser.add_argument("--reference-report", type=str, required=True,
                         help="a scripts/baseline_eval.py JSON report to compare against")
    parser.add_argument("--category", type=str, choices=["reasoning", "tools"], default="reasoning")
    parser.add_argument("--n-eval", type=int, default=60)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    reference_report = json.loads(Path(args.reference_report).read_text(encoding="utf-8"))
    reference_metrics = flatten_baseline_report(reference_report)
    print(f"Reference ({args.reference_report}): {reference_metrics}")

    print(f"Evaluating candidate ({args.candidate_checkpoint}) on category={args.category!r} ...")
    if args.category == "reasoning":
        result = eval_reasoning_no_tools(args.candidate_checkpoint, args.candidate_tokenizer, device, args.n_eval, args.seed)
        candidate_metrics = {"reasoning_accuracy": result["accuracy"]}
        required = ["reasoning_accuracy"]
    else:
        result = eval_tools(args.candidate_checkpoint, args.candidate_tokenizer, device, args.n_eval, args.seed)
        candidate_metrics = {"tools_accuracy": result["accuracy"]}
        required = ["tools_accuracy"]
    print(f"Candidate metrics: {candidate_metrics}")

    decision = gate_checkpoint(
        candidate_metrics, reference_metrics, required_metrics=required,
        min_improvement=args.min_improvement,
    )

    print(f"\n{'=' * 60}")
    print("Gate decision:")
    for reason in decision.reasons:
        print(f"  {reason}")
    print(f"\n{'ADOPTED' if decision.adopted else 'REJECTED'}")
    if not decision.adopted:
        print(f"Reference checkpoint remains the best available for this category.")

    return 0 if decision.adopted else 1


if __name__ == "__main__":
    sys.exit(main())
