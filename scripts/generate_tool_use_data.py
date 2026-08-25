"""
Generate synthetic tool-use training data: arithmetic and linear-equation
problems reformatted so the assistant's solution calls the calculator
tool for the actual computation instead of doing it "by hand" (comparison
problems are left as plain reasoning -- Phase 5/6 already showed the
model is good at those without a tool).

Usage:
    python scripts/generate_tool_use_data.py --n 3000 --out data/tool_use_dataset.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.tool_format import format_tool_call, format_tool_result  # noqa: E402
from verifiers.generators import generate_dataset  # noqa: E402
from verifiers.reward import grade  # noqa: E402

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step by step and clearly "
    "states the final answer. For arithmetic, use the calculator tool instead of "
    "computing by hand."
)


def _parse_arithmetic_expr(question: str) -> str:
    # "What is {a} {op} {b}?" -> "{a} {op} {b}"
    return question[len("What is ") : -1]


def _parse_linear_equation_expr(question: str) -> str:
    # "Solve for x: {a}x + {b} = {c}" -> "({c} - {b}) // {a}"
    body = question[len("Solve for x: ") :]
    lhs, rhs = body.split("=")
    c = int(rhs.strip())
    a_str, b_str = lhs.replace("x", " ").split("+")
    a, b = int(a_str.strip()), int(b_str.strip())
    return f"({c} - {b}) // {a}"


def to_tool_conversation(problem):
    if problem.kind == "arithmetic":
        expr = _parse_arithmetic_expr(problem.question)
    elif problem.kind == "linear_equation":
        expr = _parse_linear_equation_expr(problem.question)
    else:
        # comparison: no arithmetic involved, keep the plain reasoning solution
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.question},
            {"role": "assistant", "content": problem.solution},
        ], True

    call = format_tool_call("calculator", {"expression": expr})
    result_block = format_tool_result(problem.answer)
    assistant_content = f"Let me calculate that.{call}{result_block} The answer is {problem.answer}."
    turns = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.question},
        {"role": "assistant", "content": assistant_content},
    ]
    # self-check: the expression we handed the tool must actually evaluate
    # to the problem's real answer -- never ship training data where the
    # "worked example" would produce the wrong tool result
    from verifiers.sandbox import safe_eval_arithmetic

    ok = str(safe_eval_arithmetic(expr)) == problem.answer
    return turns, ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/tool_use_dataset.jsonl")
    args = parser.parse_args()

    problems = generate_dataset(args.n, seed=args.seed)
    rows = []
    n_bad = 0
    for problem in problems:
        turns, ok = to_tool_conversation(problem)
        if not ok:
            n_bad += 1
            continue
        rows.append({"turns": turns, "answer": problem.answer, "kind": problem.kind})

    print(f"Generated {len(rows):,} tool-use conversations (rejected {n_bad} self-check failures)")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"Kind distribution: {kinds}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
