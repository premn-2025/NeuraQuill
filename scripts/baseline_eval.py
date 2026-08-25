"""
Phase 10.5: a permanent, reusable baseline evaluation report. Runs every
category this project currently has an objective measurement for against
a given (base, sft, reasoning, tools) checkpoint set, and writes one
timestamped JSON report -- so every future experiment (Phase 11+) gets
compared against a recorded number, not a vibe.

Categories covered right now: perplexity (base LM), reasoning (Phase 5's
verifier, no tools), tool-calling (Phase 7's real orchestrator), RAG
retrieval (Phase 8, model-independent). Categories NOT yet implemented,
listed honestly rather than faked: general knowledge Q&A / summarization,
coding, hallucination/uncertainty calibration -- see PLAN.md Phase 10.5.

Usage:
    python scripts/baseline_eval.py --name neuraquill-100m \
        --base-checkpoint checkpoints/neuraquill-100m.pt \
        --base-tokens data/pretrain_tokens_100m.bin \
        --tools-checkpoint checkpoints/neuraquill-100m-tools.pt \
        --tools-tokenizer tokenizer/artifacts_tools \
        --out reports/neuraquill-100m-baseline.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from evaluation.perplexity import estimate_loss, perplexity_from_loss  # noqa: E402
from inference.orchestrator import ToolCallOrchestrator  # noqa: E402
from rag.retriever import build_index, retrieve  # noqa: E402
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from tools import default_registry  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from training.dataset import load_token_memmap, split_train_val  # noqa: E402
from verifiers import generate_dataset, grade  # noqa: E402

NOT_YET_IMPLEMENTED = [
    "general_knowledge_qa",   # no benchmark harness exists yet
    "coding",                 # needs a real code-execution sandbox (Phase 12's job)
    "hallucination_calibration",  # needs known/unknown/contradictory question sets
]


def eval_perplexity(checkpoint_path: str, tokens_path: str, device: str) -> Optional[dict]:
    if not Path(checkpoint_path).exists() or not Path(tokens_path).exists():
        return None
    model, _, step, best_val, config = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    data = load_token_memmap(tokens_path)
    _, val_data = split_train_val(data, val_fraction=0.02)
    loss = estimate_loss(model, val_data, config.block_size, batch_size=16, num_batches=30, device=device)
    return {
        "checkpoint_step": step,
        "val_loss": loss,
        "perplexity": perplexity_from_loss(loss),
        "params": model.num_parameters(),
    }


def eval_reasoning_no_tools(checkpoint_path: str, tokenizer_path: str, device: str, n: int, seed: int) -> Optional[dict]:
    if not Path(checkpoint_path).exists() or not Path(tokenizer_path).exists():
        return None
    model, _, step, _, config = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    tok = ByteLevelBPETokenizer.load(tokenizer_path)
    eot_id = tok.vocab["<|endoftext|>"]

    dataset = generate_dataset(n, seed=seed)
    by_kind = {}
    n_correct = 0
    for problem in dataset:
        prompt = (
            "<|system|>You are a helpful assistant that solves problems step by step "
            "and clearly states the final answer.<|endoftext|>"
            f"<|user|>{problem.question}<|endoftext|><|assistant|>"
        )
        ids = tok.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=80, temperature=0.3, top_k=40, stop_token_ids={eot_id})
        reply_ids = out[0, len(ids):].tolist()
        if reply_ids and reply_ids[-1] == eot_id:
            reply_ids = reply_ids[:-1]
        reply = tok.decode(reply_ids)
        result = grade(problem, reply)
        n_correct += int(result.correct)
        s = by_kind.setdefault(problem.kind, [0, 0])
        s[1] += 1
        s[0] += int(result.correct)

    return {
        "n": n, "seed": seed, "accuracy": n_correct / n,
        "by_kind": {k: {"correct": v[0], "total": v[1]} for k, v in by_kind.items()},
    }


def eval_tools(checkpoint_path: str, tokenizer_path: str, device: str, n: int, seed: int) -> Optional[dict]:
    if not Path(checkpoint_path).exists() or not Path(tokenizer_path).exists():
        return None
    model, _, step, _, config = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    tok = ByteLevelBPETokenizer.load(tokenizer_path)
    orchestrator = ToolCallOrchestrator(model, tok, default_registry(), max_tool_calls=2)

    dataset = generate_dataset(n, seed=seed)
    by_kind = {}
    n_correct = 0
    for problem in dataset:
        prompt = (
            "<|system|>You are a helpful assistant that solves problems step by step and "
            "clearly states the final answer. For arithmetic, use the calculator tool "
            "instead of computing by hand.<|endoftext|>"
            f"<|user|>{problem.question}<|endoftext|><|assistant|>"
        )
        ids = tok.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        reply = orchestrator.run(idx, max_new_tokens_per_turn=80, temperature=0.3, top_k=40)
        result = grade(problem, reply)
        n_correct += int(result.correct)
        s = by_kind.setdefault(problem.kind, [0, 0])
        s[1] += 1
        s[0] += int(result.correct)

    return {
        "n": n, "seed": seed, "accuracy": n_correct / n,
        "by_kind": {k: {"correct": v[0], "total": v[1]} for k, v in by_kind.items()},
    }


def eval_rag_retrieval() -> dict:
    """Model-independent -- retrieval quality doesn't depend on which
    checkpoint you'd generate an answer with, so this always runs."""
    facts = {
        "zorblatt": "The Zorblatt is a fictional creature that lives in deep caves and eats crystals for energy.",
        "x200": "The X200 gadget has a battery life of 14 hours and weighs exactly 300 grams.",
        "glimmerfern": "The Glimmerfern is a rare plant found only on the slopes of Mount Kavara, blooming at midnight.",
    }
    queries = [
        ("What does a Zorblatt eat?", "zorblatt"),
        ("How much does the X200 weigh?", "x200"),
        ("Where does the Glimmerfern plant grow?", "glimmerfern"),
    ]
    store, vectorizer, _ = build_index(facts, chunk_size=50, overlap=10)
    correct = sum(
        1 for q, expected in queries
        if retrieve(q, store, vectorizer, top_k=1)[0][0].metadata["doc_id"] == expected
    )
    return {"precision_at_1": correct / len(queries)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--base-tokens", type=str, default=None)
    parser.add_argument("--reasoning-checkpoint", type=str, default=None)
    parser.add_argument("--reasoning-tokenizer", type=str, default=None)
    parser.add_argument("--tools-checkpoint", type=str, default=None)
    parser.add_argument("--tools-tokenizer", type=str, default=None)
    parser.add_argument("--n-eval", type=int, default=60)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    report = {"name": args.name, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "device": device}

    if args.base_checkpoint and args.base_tokens:
        print("Evaluating: perplexity ...")
        report["perplexity"] = eval_perplexity(args.base_checkpoint, args.base_tokens, device)
        print(f"  {report['perplexity']}")

    if args.reasoning_checkpoint and args.reasoning_tokenizer:
        print("Evaluating: reasoning (no tools) ...")
        report["reasoning"] = eval_reasoning_no_tools(
            args.reasoning_checkpoint, args.reasoning_tokenizer, device, args.n_eval, args.seed
        )
        print(f"  accuracy={report['reasoning']['accuracy']:.3f}")

    if args.tools_checkpoint and args.tools_tokenizer:
        print("Evaluating: tool calling ...")
        report["tools"] = eval_tools(
            args.tools_checkpoint, args.tools_tokenizer, device, args.n_eval, args.seed
        )
        print(f"  accuracy={report['tools']['accuracy']:.3f}")

    print("Evaluating: RAG retrieval (model-independent) ...")
    report["rag_retrieval"] = eval_rag_retrieval()
    print(f"  {report['rag_retrieval']}")

    report["not_yet_implemented"] = NOT_YET_IMPLEMENTED

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote baseline report -> {out_path}")


if __name__ == "__main__":
    main()
