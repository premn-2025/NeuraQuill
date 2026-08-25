"""
Phase 9 test suite: router, planner, verifier, workers, and end-to-end
agent runs on scripted scenarios (per PLAN.md's explicit requirement).

Run with: pytest tests/test_agents.py -v
"""
import sys
from pathlib import Path
from typing import List

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.agent import Agent  # noqa: E402
from agents.planner import plan  # noqa: E402
from agents.router import route  # noqa: E402
from agents.verifier import verify_knowledge_grounded, verify_reasoning  # noqa: E402
from agents.workers import RAGWorker, ReasoningWorker  # noqa: E402
from inference.tool_format import TOOL_SPECIAL_TOKENS, format_tool_call  # noqa: E402
from rag.retriever import build_index  # noqa: E402
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from tools.calculator import CalculatorTool  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402
from verifiers.generators import ReasoningProblem  # noqa: E402


# --------------------------------------------------------------------------
# router
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task,expected",
    [
        ("What is 12 + 7?", "reasoning"),
        ("Solve for x: 3x + 2 = 11", "reasoning"),
        ("Is Alice taller than Bob?", "reasoning"),
        ("How much does the X200 weigh?", "knowledge"),
        ("What is the capital of France?", "knowledge"),
        ("Tell me about the Zorblatt creature.", "knowledge"),
    ],
)
def test_route_classifies_correctly(task, expected):
    assert route(task) == expected


# --------------------------------------------------------------------------
# planner
# --------------------------------------------------------------------------


def test_plan_splits_compound_task():
    subtasks = plan("What is 12 + 7, and what does the X200 weigh?")
    assert len(subtasks) == 2
    assert "12 + 7" in subtasks[0]
    assert "X200" in subtasks[1]


def test_plan_single_task_passthrough():
    assert plan("What is 12 + 7?") == ["What is 12 + 7?"]


def test_plan_splits_on_semicolon():
    subtasks = plan("What is 2 + 2?; What is the capital of France?")
    assert len(subtasks) == 2


def test_plan_empty_task_returns_itself():
    assert plan("") == [""]


# --------------------------------------------------------------------------
# verifier
# --------------------------------------------------------------------------


def test_verify_reasoning_correct_and_incorrect():
    problem = ReasoningProblem(question="What is 2 + 2?", answer="4", solution="2 + 2 = 4", kind="arithmetic")
    assert verify_reasoning(problem, "The answer is 4.").correct
    assert not verify_reasoning(problem, "The answer is 5.").correct


def test_verify_knowledge_grounded():
    assert verify_knowledge_grounded("The X200 weighs 300 grams.", "300 grams")
    assert not verify_knowledge_grounded("I don't know.", "300 grams")
    assert verify_knowledge_grounded("THE ANSWER IS YES", "yes")  # case-insensitive


# --------------------------------------------------------------------------
# shared fixtures: a tokenizer with both chat + tool tokens, and a
# scripted model double (same pattern as test_tools.py)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent_tokenizer():
    corpus = (
        "I will calculate this now. The answer is correct. The X200 weighs 300 grams. "
        "Sorry, there was an error. Let me try again with a different approach. "
    ) * 20
    tok = ByteLevelBPETokenizer.train([corpus], vocab_size=700, special_tokens=("<|endoftext|>", "<|pad|>"))
    tok.add_special_tokens(["<|system|>", "<|user|>", "<|assistant|>"] + TOOL_SPECIAL_TOKENS)
    return tok


class ScriptedModel:
    """Test double for GPT.generate -- see tests/test_tools.py for the
    full rationale. Appends one pre-programmed continuation per call."""

    def __init__(self, tokenizer, continuations: List[str]):
        self.tokenizer = tokenizer
        self._continuations = list(continuations)
        self.calls = 0

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token_ids=None):
        text = self._continuations[self.calls]
        self.calls += 1
        new_ids = self.tokenizer.encode(text)
        new_tensor = torch.tensor([new_ids], dtype=torch.long, device=idx.device)
        return torch.cat([idx, new_tensor], dim=1)


# --------------------------------------------------------------------------
# ReasoningWorker
# --------------------------------------------------------------------------


def test_reasoning_worker_solves_via_tool_call(agent_tokenizer):
    call_text = format_tool_call("calculator", {"expression": "12 + 7"})
    model = ScriptedModel(
        agent_tokenizer,
        [f"I will calculate this now. {call_text}", "The answer is correct.<|endoftext|>"],
    )
    worker = ReasoningWorker(model, agent_tokenizer, ToolRegistry([CalculatorTool()]))
    result = worker.run("What is 12 + 7?")
    assert result.worker == "reasoning"
    assert model.calls == 2


# --------------------------------------------------------------------------
# RAGWorker
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def knowledge_index():
    documents = {
        "x200": "The X200 gadget has a battery life of 14 hours and weighs exactly 300 grams.",
        "other": "This document is about something completely unrelated to gadgets.",
    }
    store, vectorizer, chunks = build_index(documents, chunk_size=50, overlap=10)
    return store, vectorizer


def test_rag_worker_retrieves_relevant_context(agent_tokenizer, knowledge_index):
    store, vectorizer = knowledge_index
    model = ScriptedModel(agent_tokenizer, ["The X200 weighs 300 grams.<|endoftext|>"])
    worker = RAGWorker(model, agent_tokenizer, store, vectorizer, top_k=1)
    result = worker.run("How much does the X200 weigh?")
    assert result.worker == "knowledge"
    assert "300 grams" in result.metadata["retrieved"][0]


# --------------------------------------------------------------------------
# Agent: end-to-end scripted scenarios (the explicit PLAN.md requirement)
# --------------------------------------------------------------------------


def test_agent_single_reasoning_task_verified_correct(agent_tokenizer):
    call_text = format_tool_call("calculator", {"expression": "12 + 7"})
    reasoning_model = ScriptedModel(
        agent_tokenizer, [f"Calculating. {call_text}", "The answer is 19.<|endoftext|>"]
    )
    dummy_rag_model = ScriptedModel(agent_tokenizer, [])  # never called in this scenario
    reasoning_worker = ReasoningWorker(reasoning_model, agent_tokenizer, ToolRegistry([CalculatorTool()]))
    rag_worker = RAGWorker(dummy_rag_model, agent_tokenizer, *build_index({"x": "irrelevant"}, chunk_size=10, overlap=2))

    agent = Agent(reasoning_worker, rag_worker)
    problem = ReasoningProblem(question="What is 12 + 7?", answer="19", solution="12 + 7 = 19", kind="arithmetic")
    run = agent.run("What is 12 + 7?", verifiers=[lambda ans: verify_reasoning(problem, ans).correct])

    assert len(run.steps) == 1
    assert run.steps[0].route == "reasoning"
    assert run.steps[0].verified is True
    assert run.all_verified is True


def test_agent_single_knowledge_task_verified_correct(agent_tokenizer, knowledge_index):
    store, vectorizer = knowledge_index
    rag_model = ScriptedModel(agent_tokenizer, ["The X200 weighs 300 grams.<|endoftext|>"])
    dummy_reasoning_model = ScriptedModel(agent_tokenizer, [])
    reasoning_worker = ReasoningWorker(dummy_reasoning_model, agent_tokenizer, ToolRegistry([CalculatorTool()]))
    rag_worker = RAGWorker(rag_model, agent_tokenizer, store, vectorizer, top_k=1)

    agent = Agent(reasoning_worker, rag_worker)
    run = agent.run(
        "How much does the X200 weigh?",
        verifiers=[lambda ans: verify_knowledge_grounded(ans, "300 grams")],
    )
    assert run.steps[0].route == "knowledge"
    assert run.steps[0].verified is True
    assert run.all_verified is True


def test_agent_compound_task_routes_and_verifies_both_subtasks(agent_tokenizer, knowledge_index):
    store, vectorizer = knowledge_index
    call_text = format_tool_call("calculator", {"expression": "12 + 7"})
    reasoning_model = ScriptedModel(
        agent_tokenizer, [f"Calculating. {call_text}", "The answer is 19.<|endoftext|>"]
    )
    rag_model = ScriptedModel(agent_tokenizer, ["The X200 weighs 300 grams.<|endoftext|>"])
    reasoning_worker = ReasoningWorker(reasoning_model, agent_tokenizer, ToolRegistry([CalculatorTool()]))
    rag_worker = RAGWorker(rag_model, agent_tokenizer, store, vectorizer, top_k=1)

    agent = Agent(reasoning_worker, rag_worker)
    problem = ReasoningProblem(question="What is 12 + 7?", answer="19", solution="12 + 7 = 19", kind="arithmetic")
    run = agent.run(
        "What is 12 + 7, and how much does the X200 weigh?",
        verifiers=[
            lambda ans: verify_reasoning(problem, ans).correct,
            lambda ans: verify_knowledge_grounded(ans, "300 grams"),
        ],
    )

    assert len(run.steps) == 2
    assert run.steps[0].route == "reasoning"
    assert run.steps[1].route == "knowledge"
    assert run.all_verified is True
    assert "19" in run.final_answer
    assert "300 grams" in run.final_answer


def test_agent_reports_failure_when_a_subtask_is_wrong(agent_tokenizer, knowledge_index):
    store, vectorizer = knowledge_index
    call_text = format_tool_call("calculator", {"expression": "12 + 7"})
    # deliberately wrong final answer, to prove the verifier actually catches failure
    reasoning_model = ScriptedModel(
        agent_tokenizer, [f"Calculating. {call_text}", "The answer is 20.<|endoftext|>"]
    )
    dummy_rag_model = ScriptedModel(agent_tokenizer, [])
    reasoning_worker = ReasoningWorker(reasoning_model, agent_tokenizer, ToolRegistry([CalculatorTool()]))
    rag_worker = RAGWorker(dummy_rag_model, agent_tokenizer, store, vectorizer, top_k=1)

    agent = Agent(reasoning_worker, rag_worker)
    problem = ReasoningProblem(question="What is 12 + 7?", answer="19", solution="12 + 7 = 19", kind="arithmetic")
    run = agent.run("What is 12 + 7?", verifiers=[lambda ans: verify_reasoning(problem, ans).correct])

    assert run.steps[0].verified is False
    assert run.all_verified is False


def test_agent_verifiers_length_mismatch_raises(agent_tokenizer, knowledge_index):
    store, vectorizer = knowledge_index
    reasoning_worker = ReasoningWorker(
        ScriptedModel(agent_tokenizer, []), agent_tokenizer, ToolRegistry([CalculatorTool()])
    )
    rag_worker = RAGWorker(ScriptedModel(agent_tokenizer, []), agent_tokenizer, store, vectorizer)
    agent = Agent(reasoning_worker, rag_worker)
    with pytest.raises(ValueError):
        agent.run("What is 12 + 7, and how much does the X200 weigh?", verifiers=[lambda ans: True])


def test_agent_run_without_verifiers_has_none_verified_status(agent_tokenizer, knowledge_index):
    store, vectorizer = knowledge_index
    call_text = format_tool_call("calculator", {"expression": "1 + 1"})
    reasoning_model = ScriptedModel(agent_tokenizer, [f"Calculating. {call_text}", "The answer is 2.<|endoftext|>"])
    reasoning_worker = ReasoningWorker(reasoning_model, agent_tokenizer, ToolRegistry([CalculatorTool()]))
    rag_worker = RAGWorker(ScriptedModel(agent_tokenizer, []), agent_tokenizer, store, vectorizer)
    agent = Agent(reasoning_worker, rag_worker)

    run = agent.run("What is 1 + 1?")
    assert run.steps[0].verified is None
    assert run.all_verified is None
