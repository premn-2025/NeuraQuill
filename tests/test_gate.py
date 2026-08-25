"""
Phase 12 test suite: the evaluation gate that decides whether a new
checkpoint gets adopted.

Run with: pytest tests/test_gate.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.gate import flatten_baseline_report, gate_checkpoint  # noqa: E402


def test_gate_adopts_when_candidate_improves():
    decision = gate_checkpoint(
        candidate_metrics={"reasoning_accuracy": 0.45},
        reference_metrics={"reasoning_accuracy": 0.40},
        required_metrics=["reasoning_accuracy"],
    )
    assert decision.adopted is True
    assert "OK" in decision.reasons[0]


def test_gate_rejects_regression():
    """The exact Phase 11 scenario: 0.367 -> 0.200 must be rejected."""
    decision = gate_checkpoint(
        candidate_metrics={"reasoning_accuracy": 0.200},
        reference_metrics={"reasoning_accuracy": 0.367},
        required_metrics=["reasoning_accuracy"],
    )
    assert decision.adopted is False
    assert "FAIL" in decision.reasons[0]


def test_gate_equal_metrics_pass_with_zero_min_improvement():
    decision = gate_checkpoint(
        candidate_metrics={"reasoning_accuracy": 0.40},
        reference_metrics={"reasoning_accuracy": 0.40},
        required_metrics=["reasoning_accuracy"],
    )
    assert decision.adopted is True


def test_gate_equal_metrics_fail_when_strict_improvement_required():
    decision = gate_checkpoint(
        candidate_metrics={"reasoning_accuracy": 0.40},
        reference_metrics={"reasoning_accuracy": 0.40},
        required_metrics=["reasoning_accuracy"],
        min_improvement=0.01,
    )
    assert decision.adopted is False


def test_gate_lower_is_better_metric():
    # perplexity: lower candidate value is an improvement
    decision = gate_checkpoint(
        candidate_metrics={"perplexity": 30.0},
        reference_metrics={"perplexity": 45.0},
        required_metrics=["perplexity"],
        lower_is_better={"perplexity"},
    )
    assert decision.adopted is True

    decision_worse = gate_checkpoint(
        candidate_metrics={"perplexity": 50.0},
        reference_metrics={"perplexity": 45.0},
        required_metrics=["perplexity"],
        lower_is_better={"perplexity"},
    )
    assert decision_worse.adopted is False


def test_gate_allow_regression_tolerance():
    # a small regression within tolerance still passes
    decision = gate_checkpoint(
        candidate_metrics={"reasoning_accuracy": 0.38},
        reference_metrics={"reasoning_accuracy": 0.40},
        required_metrics=["reasoning_accuracy"],
        allow_regression={"reasoning_accuracy": 0.03},
    )
    assert decision.adopted is True

    # beyond tolerance fails
    decision2 = gate_checkpoint(
        candidate_metrics={"reasoning_accuracy": 0.30},
        reference_metrics={"reasoning_accuracy": 0.40},
        required_metrics=["reasoning_accuracy"],
        allow_regression={"reasoning_accuracy": 0.03},
    )
    assert decision2.adopted is False


def test_gate_missing_metric_in_candidate_rejects():
    decision = gate_checkpoint(
        candidate_metrics={},
        reference_metrics={"reasoning_accuracy": 0.40},
        required_metrics=["reasoning_accuracy"],
    )
    assert decision.adopted is False
    assert "missing from candidate" in decision.reasons[0]


def test_gate_missing_metric_in_reference_rejects():
    decision = gate_checkpoint(
        candidate_metrics={"reasoning_accuracy": 0.40},
        reference_metrics={},
        required_metrics=["reasoning_accuracy"],
    )
    assert decision.adopted is False
    assert "missing from reference" in decision.reasons[0]


def test_gate_evaluates_all_metrics_not_short_circuited():
    """One failing metric among several must not stop the others from
    being evaluated and reported -- the reasons list should be a
    complete picture, not just the first failure."""
    decision = gate_checkpoint(
        candidate_metrics={"reasoning_accuracy": 0.20, "tools_accuracy": 0.99},
        reference_metrics={"reasoning_accuracy": 0.40, "tools_accuracy": 0.98},
        required_metrics=["reasoning_accuracy", "tools_accuracy"],
    )
    assert decision.adopted is False
    assert len(decision.reasons) == 2
    assert "FAIL" in decision.reasons[0]
    assert "OK" in decision.reasons[1]


def test_gate_multiple_metrics_all_must_pass():
    decision = gate_checkpoint(
        candidate_metrics={"reasoning_accuracy": 0.45, "tools_accuracy": 0.99},
        reference_metrics={"reasoning_accuracy": 0.40, "tools_accuracy": 0.98},
        required_metrics=["reasoning_accuracy", "tools_accuracy"],
    )
    assert decision.adopted is True


def test_gate_decision_preserves_full_metric_dicts():
    decision = gate_checkpoint(
        candidate_metrics={"reasoning_accuracy": 0.45, "extra": 1.0},
        reference_metrics={"reasoning_accuracy": 0.40},
        required_metrics=["reasoning_accuracy"],
    )
    assert decision.candidate_metrics == {"reasoning_accuracy": 0.45, "extra": 1.0}
    assert decision.reference_metrics == {"reasoning_accuracy": 0.40}


# --------------------------------------------------------------------------
# flatten_baseline_report
# --------------------------------------------------------------------------


def test_flatten_baseline_report_full():
    report = {
        "perplexity": {"perplexity": 33.5, "val_loss": 3.51},
        "reasoning": {"accuracy": 0.367},
        "tools": {"accuracy": 0.983},
        "rag_retrieval": {"precision_at_1": 1.0},
    }
    flat = flatten_baseline_report(report)
    assert flat == {
        "perplexity": 33.5, "val_loss": 3.51,
        "reasoning_accuracy": 0.367, "tools_accuracy": 0.983, "rag_precision_at_1": 1.0,
    }


def test_flatten_baseline_report_partial():
    report = {"reasoning": {"accuracy": 0.40}}
    flat = flatten_baseline_report(report)
    assert flat == {"reasoning_accuracy": 0.40}


def test_flatten_baseline_report_empty():
    assert flatten_baseline_report({}) == {}


def test_gate_real_phase11_scenario_end_to_end():
    """Reproduces the actual Phase 11 numbers end to end: the teacher-
    tuned checkpoint (20.0%) against the reference 100M reasoning
    checkpoint (36.7%) must be rejected by the gate exactly as it was
    rejected by hand in PLAN.md."""
    reference_report = {"reasoning": {"accuracy": 0.367}}
    candidate_report = {"reasoning": {"accuracy": 0.200}}

    reference_metrics = flatten_baseline_report(reference_report)
    candidate_metrics = flatten_baseline_report(candidate_report)

    decision = gate_checkpoint(
        candidate_metrics, reference_metrics,
        required_metrics=["reasoning_accuracy"],
    )
    assert decision.adopted is False
