"""
Real-world Phase 9 check: run the full planner -> router -> worker ->
verifier Agent loop with actual trained checkpoints -- the tool-tuned
model (Phase 7) as the reasoning worker, the general SFT model (Phase 4)
as the knowledge/RAG worker (Phase 8's retrieval index) -- on a set of
scripted single- and multi-step scenarios, and report end-to-end task
completion.

Usage:
    python scripts/agent_demo.py
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from agents.agent import Agent  # noqa: E402
from agents.verifier import verify_knowledge_grounded, verify_reasoning  # noqa: E402
from agents.workers import RAGWorker, ReasoningWorker  # noqa: E402
from rag.retriever import build_index  # noqa: E402
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from tools import default_registry  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from verifiers.generators import ReasoningProblem  # noqa: E402

KNOWLEDGE_BASE = {
    "zorblatt": "The Zorblatt is a fictional creature that lives in deep caves and eats crystals for energy.",
    "x200": "The X200 gadget has a battery life of 14 hours and weighs exactly 300 grams.",
    "starforge_engine": "The Starforge engine converts solidified starlight into propulsion energy for deep space travel.",
    "moonpetal_tea": "Moonpetal tea is brewed from dried moonpetal flowers and is said to help with restful sleep.",
}

# each scenario: (task, [ (kind, expected) for each planned subtask ])
# kind is "reasoning" (expected = ReasoningProblem) or "knowledge" (expected = fact substring)
SCENARIOS = [
    (
        "What is 23 + 19?",
        [("reasoning", ReasoningProblem("What is 23 + 19?", "42", "23 + 19 = 42", "arithmetic"))],
    ),
    (
        "What does a Zorblatt eat?",
        [("knowledge", "crystals")],
    ),
    (
        "Solve for x: 4x + 3 = 19",
        [("reasoning", ReasoningProblem("Solve for x: 4x + 3 = 19", "4", "4x + 3 = 19\n4x = 16\nx = 4", "linear_equation"))],
    ),
    (
        "What is 45 - 12, and how much does the X200 weigh?",
        [
            ("reasoning", ReasoningProblem("What is 45 - 12?", "33", "45 - 12 = 33", "arithmetic")),
            ("knowledge", "300 grams"),
        ],
    ),
    (
        "What powers the Starforge engine, and what is 8 * 7?",
        [
            ("knowledge", "starlight"),
            ("reasoning", ReasoningProblem("What is 8 * 7?", "56", "8 * 7 = 56", "arithmetic")),
        ],
    ),
    (
        "What is Moonpetal tea used for?",
        [("knowledge", "sleep")],
    ),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading reasoning worker (Phase 7 tool-tuned checkpoint) ...")
    reasoning_model, _, step_r, val_r, _ = load_checkpoint(
        "checkpoints/neuraquill-10m-tools.pt", device=device
    )
    reasoning_model.eval()
    reasoning_tok = ByteLevelBPETokenizer.load("tokenizer/artifacts_tools")
    reasoning_worker = ReasoningWorker(reasoning_model, reasoning_tok, default_registry())
    print(f"  step {step_r}, best_val_loss {val_r:.4f}")

    print("Loading knowledge worker (Phase 4 SFT checkpoint) + RAG index (Phase 8) ...")
    rag_model, _, step_k, val_k, _ = load_checkpoint(
        "checkpoints/neuraquill-10m-sft.pt", device=device
    )
    rag_model.eval()
    rag_tok = ByteLevelBPETokenizer.load("tokenizer/artifacts_chat")
    store, vectorizer, _ = build_index(KNOWLEDGE_BASE, chunk_size=50, overlap=10)
    rag_worker = RAGWorker(rag_model, rag_tok, store, vectorizer, top_k=1)
    print(f"  step {step_k}, best_val_loss {val_k:.4f}")

    agent = Agent(reasoning_worker, rag_worker)

    n_scenarios_passed = 0
    n_subtasks_total = 0
    n_subtasks_passed = 0

    for task, expected_subtasks in SCENARIOS:
        verifiers = []
        for kind, expected in expected_subtasks:
            if kind == "reasoning":
                verifiers.append(lambda ans, p=expected: verify_reasoning(p, ans).correct)
            else:
                verifiers.append(lambda ans, e=expected: verify_knowledge_grounded(ans, e))

        run = agent.run(task, device=device, verifiers=verifiers)

        print(f"\nTask: {task}")
        for step, (exp_kind, expected) in zip(run.steps, expected_subtasks):
            n_subtasks_total += 1
            n_subtasks_passed += int(bool(step.verified))
            route_ok = "OK" if step.route == exp_kind else "ROUTE-MISMATCH"
            print(f"  [{route_ok}] subtask: {step.task!r}")
            print(f"       route={step.route} answer={step.answer!r} verified={step.verified}")

        scenario_passed = run.all_verified is True
        n_scenarios_passed += int(scenario_passed)
        print(f"  => scenario {'PASSED' if scenario_passed else 'FAILED'}")

    print(f"\n{'=' * 60}")
    print(f"Scenarios passed: {n_scenarios_passed}/{len(SCENARIOS)}")
    print(f"Subtasks passed:  {n_subtasks_passed}/{n_subtasks_total}")


if __name__ == "__main__":
    main()
