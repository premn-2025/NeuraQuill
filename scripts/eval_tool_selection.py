"""
Tool-selection accuracy: does the model correctly decide *when* to call a
tool, not just execute correctly once it does? Phase 7-9's numbers all
measure final-answer accuracy through the orchestrator, which conflates
"chose not to call a tool and reasoned correctly" (comparison problems)
with "chose to call a tool and used it correctly" (arithmetic problems) --
this measures the choice itself, separately from execution correctness.

Ground truth: arithmetic/linear_equation problems SHOULD trigger a tool
call (that's what Phase 7 trained the model to do); comparison problems
should NOT (pure reasoning is already ~100% there, per every phase from
5 onward -- calling a tool would be an unnecessary, wrong choice even if
it happened to still produce the right final answer).

Usage:
    python scripts/eval_tool_selection.py --checkpoint checkpoints/neuraquill-100m-tools.pt \
        --tokenizer tokenizer/artifacts_tools --n 60
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from inference.tool_format import TOOL_CALL_START  # noqa: E402
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from verifiers import generate_dataset  # noqa: E402

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step by step and clearly "
    "states the final answer. For arithmetic, use the calculator tool instead "
    "of computing by hand."
)

SHOULD_USE_TOOL = {"arithmetic": True, "linear_equation": True, "comparison": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/artifacts_tools")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, step, best_val, config = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    tok = ByteLevelBPETokenizer.load(args.tokenizer)
    eot_id = tok.vocab["<|endoftext|>"]
    call_start_id = tok.vocab.get(TOOL_CALL_START)

    dataset = generate_dataset(args.n, seed=args.seed)

    n_correct_choice = 0
    by_kind = {}
    for problem in dataset:
        prompt = (
            f"<|system|>{SYSTEM_PROMPT}<|endoftext|>"
            f"<|user|>{problem.question}<|endoftext|><|assistant|>"
        )
        ids = tok.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        # single-turn generation (no orchestrator/tool execution) -- we're
        # only measuring whether the model *chooses* to open a tool call,
        # not whether the call succeeds
        out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=0.3, top_k=40, stop_token_ids={eot_id})
        reply_ids = out[0, len(ids):].tolist()
        called_tool = call_start_id in reply_ids

        should_call = SHOULD_USE_TOOL[problem.kind]
        correct_choice = called_tool == should_call
        n_correct_choice += int(correct_choice)

        s = by_kind.setdefault(problem.kind, [0, 0])
        s[1] += 1
        s[0] += int(correct_choice)

    print(f"Tool-selection accuracy: {n_correct_choice}/{args.n} ({n_correct_choice / args.n * 100:.1f}%)")
    for kind, (correct, total) in by_kind.items():
        expected = "should call tool" if SHOULD_USE_TOOL[kind] else "should NOT call tool"
        print(f"  {kind:16s} ({expected}): {correct}/{total} ({correct / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
