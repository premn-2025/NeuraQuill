"""
Evaluation gate: a new checkpoint is only "adopted" (becomes what the
next stage builds on) if it demonstrably beats the reference on the
metrics that matter -- an explicit, testable policy instead of "it
looked better so I kept it." Phase 11's negative result (a checkpoint
that clearly regressed reasoning accuracy, and would have been silently
adopted without this check) is exactly the failure mode this exists to
catch.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class GateDecision:
    adopted: bool
    reasons: List[str]
    candidate_metrics: Dict[str, float]
    reference_metrics: Dict[str, float]


def gate_checkpoint(
    candidate_metrics: Dict[str, float],
    reference_metrics: Dict[str, float],
    required_metrics: List[str],
    min_improvement: float = 0.0,
    allow_regression: Optional[Dict[str, float]] = None,
    lower_is_better: Optional[Set[str]] = None,
) -> GateDecision:
    """A candidate is adopted only if every metric in `required_metrics`
    is present in both dicts and the candidate is at least as good as
    the reference (no regression by default, `min_improvement` to
    require strictly better). `lower_is_better` names metrics where a
    smaller value wins (e.g. perplexity); everything else assumes
    higher-is-better (e.g. accuracy). `allow_regression` gives a
    per-metric tolerance for a small, acceptable wobble (e.g. perplexity
    jittering slightly) while metrics not listed there must not regress
    at all -- Phase 11's lesson was specifically that reasoning accuracy
    must never be allowed to regress silently.

    Every required metric is evaluated (not short-circuited) so the
    reasons list is a complete picture, not just the first failure.
    """
    allow_regression = allow_regression or {}
    lower_is_better = lower_is_better or set()
    reasons: List[str] = []
    adopted = True

    for metric in required_metrics:
        if metric not in candidate_metrics:
            reasons.append(f"{metric}: missing from candidate metrics")
            adopted = False
            continue
        if metric not in reference_metrics:
            reasons.append(f"{metric}: missing from reference metrics (nothing to compare against)")
            adopted = False
            continue

        cand = candidate_metrics[metric]
        ref = reference_metrics[metric]
        tolerance = allow_regression.get(metric, 0.0)

        if metric in lower_is_better:
            threshold = ref - min_improvement + tolerance
            ok = cand <= threshold
            comparator = "<="
        else:
            threshold = ref + min_improvement - tolerance
            ok = cand >= threshold
            comparator = ">="

        status = "OK" if ok else "FAIL"
        reasons.append(
            f"{metric}: candidate={cand:.4f} {comparator} threshold={threshold:.4f} "
            f"(reference={ref:.4f}) [{status}]"
        )
        adopted = adopted and ok

    return GateDecision(
        adopted=adopted, reasons=reasons,
        candidate_metrics=dict(candidate_metrics), reference_metrics=dict(reference_metrics),
    )


def flatten_baseline_report(report: dict) -> Dict[str, float]:
    """Flattens a scripts/baseline_eval.py report into the flat metric
    dict gate_checkpoint expects."""
    flat: Dict[str, float] = {}
    if report.get("perplexity"):
        flat["perplexity"] = report["perplexity"]["perplexity"]
        flat["val_loss"] = report["perplexity"]["val_loss"]
    if report.get("reasoning"):
        flat["reasoning_accuracy"] = report["reasoning"]["accuracy"]
    if report.get("tools"):
        flat["tools_accuracy"] = report["tools"]["accuracy"]
    if report.get("rag_retrieval"):
        flat["rag_precision_at_1"] = report["rag_retrieval"]["precision_at_1"]
    return flat
