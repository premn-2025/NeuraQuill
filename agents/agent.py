"""
The planner -> router -> worker -> verifier loop: split a (possibly
compound) task into subtasks, route each to the worker suited for it,
execute, and optionally verify each subtask's answer against a supplied
verifier -- built entirely on the Phase 7 (tools) and Phase 8 (RAG)
infrastructure already in place.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .planner import plan
from .router import route
from .workers import RAGWorker, ReasoningWorker


@dataclass
class AgentStep:
    task: str
    route: str
    answer: str
    verified: Optional[bool]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRun:
    task: str
    steps: List[AgentStep]
    final_answer: str

    @property
    def all_verified(self) -> Optional[bool]:
        """None if no step was verified; otherwise True only if every
        verified step passed."""
        verdicts = [s.verified for s in self.steps if s.verified is not None]
        if not verdicts:
            return None
        return all(verdicts)


class Agent:
    def __init__(self, reasoning_worker: ReasoningWorker, rag_worker: RAGWorker):
        self.reasoning_worker = reasoning_worker
        self.rag_worker = rag_worker

    def run(
        self,
        task: str,
        device: str = "cpu",
        verifiers: Optional[List[Optional[Callable[[str], bool]]]] = None,
    ) -> AgentRun:
        subtasks = plan(task)
        if verifiers is not None and len(verifiers) != len(subtasks):
            raise ValueError(
                f"verifiers list length ({len(verifiers)}) must match the "
                f"number of planned subtasks ({len(subtasks)}): {subtasks}"
            )

        steps: List[AgentStep] = []
        for i, subtask in enumerate(subtasks):
            chosen_route = route(subtask)
            worker = self.reasoning_worker if chosen_route == "reasoning" else self.rag_worker
            result = worker.run(subtask, device=device)
            verifier = verifiers[i] if verifiers is not None else None
            verified = verifier(result.answer) if verifier is not None else None
            steps.append(
                AgentStep(
                    task=subtask, route=chosen_route, answer=result.answer,
                    verified=verified, metadata=result.metadata,
                )
            )

        final_answer = " ".join(s.answer for s in steps)
        return AgentRun(task=task, steps=steps, final_answer=final_answer)
