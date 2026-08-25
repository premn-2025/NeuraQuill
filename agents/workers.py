"""
Workers: each wraps existing Phase 7 (tool-calling) / Phase 8 (RAG)
infrastructure to actually perform one kind of subtask. Each worker owns
its own (model, tokenizer) pair rather than sharing one -- in the real
run, the reasoning worker uses the tool-tuned checkpoint (its own,
larger, tool-token-extended vocab) while the knowledge worker uses the
general SFT checkpoint (a different, smaller vocab); mixing them would
be a real bug, not a simplification.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from inference.orchestrator import ToolCallOrchestrator
from rag.retriever import retrieve
from rag.vector_store import VectorStore
from rag.embeddings import TfidfVectorizer
from tools.registry import ToolRegistry


@dataclass
class WorkerResult:
    answer: str
    worker: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningWorker:
    """Solves arithmetic/logic subtasks via the tool-calling orchestrator."""

    name = "reasoning"

    def __init__(
        self,
        model,
        tokenizer,
        registry: ToolRegistry,
        system_prompt: str = (
            "You are a helpful assistant that solves problems step by step and clearly "
            "states the final answer. For arithmetic, use the calculator tool instead of "
            "computing by hand."
        ),
        max_new_tokens: int = 80,
        temperature: float = 0.3,
        max_tool_calls: int = 2,
    ):
        self.orchestrator = ToolCallOrchestrator(model, tokenizer, registry, max_tool_calls=max_tool_calls)
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def run(self, task: str, device: str = "cpu") -> WorkerResult:
        prompt_text = (
            f"<|system|>{self.system_prompt}<|endoftext|>"
            f"<|user|>{task}<|endoftext|>"
            "<|assistant|>"
        )
        ids = self.tokenizer.encode(prompt_text)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        answer = self.orchestrator.run(
            idx, max_new_tokens_per_turn=self.max_new_tokens, temperature=self.temperature
        )
        return WorkerResult(answer=answer, worker=self.name, metadata={})


class RAGWorker:
    """Answers knowledge subtasks by retrieving relevant context (Phase 8)
    and generating over it."""

    name = "knowledge"

    def __init__(
        self,
        model,
        tokenizer,
        store: VectorStore,
        vectorizer: TfidfVectorizer,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer the question.",
        top_k: int = 2,
        max_new_tokens: int = 80,
        temperature: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.store = store
        self.vectorizer = vectorizer
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def run(self, task: str, device: str = "cpu") -> WorkerResult:
        results = retrieve(task, self.store, self.vectorizer, top_k=self.top_k)
        retrieved_chunks: List[str] = [chunk.text for chunk, _ in results]
        context = " ".join(retrieved_chunks)

        prompt_text = (
            f"<|system|>{self.system_prompt}<|endoftext|>"
            f"<|user|>Context: {context}\n\nQuestion: {task}<|endoftext|>"
            "<|assistant|>"
        )
        ids = self.tokenizer.encode(prompt_text)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        eot_id = self.tokenizer.vocab["<|endoftext|>"]
        out = self.model.generate(
            idx, max_new_tokens=self.max_new_tokens, temperature=self.temperature,
            top_k=40, stop_token_ids={eot_id},
        )
        reply_ids = out[0, len(ids):].tolist()
        if reply_ids and reply_ids[-1] == eot_id:
            reply_ids = reply_ids[:-1]
        answer = self.tokenizer.decode(reply_ids)
        return WorkerResult(answer=answer, worker=self.name, metadata={"retrieved": retrieved_chunks})
