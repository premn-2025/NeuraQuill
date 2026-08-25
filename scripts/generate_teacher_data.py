"""
Phase 11: Teacher Data Factory.

Design, stated plainly: teachers are NOT trusted to invent new
problems+answers from scratch (a free-form word problem from a local
model isn't reliably verifiable). Instead, problems come from Phase 5's
self-verifying generators (which already know the correct answer), and
each teacher is asked to write a *step-by-step explanation* for that
exact problem. The teacher's stated final answer is graded against the
known-correct answer with Phase 5's verifier -- only explanations that
check out become SFT data. This buys linguistic diversity and (for
stronger teachers) better reasoning prose, without trusting anything
unverifiable.

Every (problem, teacher) pair is recorded as a rich Trajectory (question,
teacher, prompt_version, answer, reasoning_type, verifier_result,
extracted_answer, quality_score, disagreement_score) -- not just a flat
question->answer pair -- and the full trajectory log is kept even for
rejected answers, since which teachers get which kinds of problems wrong
is itself useful signal for later phases (curriculum, reward modeling).

Teachers are routed by category (see teachers/roster.py) -- not every
teacher answers every question. Ollama models cycle pull -> generate ->
delete -> next; a model already present before this script started is
used as-is and is NEVER deleted (only models this run pulls itself get
cleaned up). The Anthropic teacher needs no pull/delete -- it's an API.

Usage:
    python scripts/generate_teacher_data.py --n-problems 25 --category reasoning
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teachers import build_roster, compute_disagreement_scores, load_env, teachers_for_category  # noqa: E402
from teachers.base import Teacher  # noqa: E402
from teachers.ollama_manager import OllamaManagerError, delete_model, is_present, pull_model  # noqa: E402
from teachers.trajectory import PROMPT_VERSION, Trajectory  # noqa: E402
from verifiers.generators import generate_dataset, to_conversation  # noqa: E402
from verifiers.reward import grade  # noqa: E402

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the given problem step by step, "
    "showing your work clearly, and end your response with a line reading "
    "exactly 'The answer is <number>.'"
)


def build_prompt(question: str) -> str:
    return f"Problem: {question}\n\nSolve it step by step, then state the final answer."


def run_teacher_on_problems(teacher: Teacher, problems) -> List[Trajectory]:
    trajectories = []
    for problem in problems:
        try:
            text = teacher.generate(build_prompt(problem.question), system=SYSTEM_PROMPT, temperature=0.7)
        except Exception as e:
            print(f"    [error] {teacher.name} on {problem.question!r}: {e}")
            continue
        result = grade(problem, text)
        trajectories.append(
            Trajectory(
                question=problem.question, teacher=teacher.name, prompt_version=PROMPT_VERSION,
                answer=text, reasoning_type=problem.kind, tool_calls=[],
                verifier_result=result.correct, extracted_answer=result.extracted_answer,
                quality_score=1.0 if result.correct else 0.0,
            )
        )
    return trajectories


def write_outputs(all_trajectories: List[Trajectory], problems, out_path: Path, traj_path: Path) -> None:
    """Serializes current results to disk. Called after *every* teacher
    (not just once at the end) so a later teacher's failure -- pull
    timeout, API error, anything unanticipated -- can never wipe out
    results already earned from the teachers that succeeded."""
    compute_disagreement_scores(all_trajectories)  # cheap to recompute; trajectory counts stay small

    traj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traj_path, "w", encoding="utf-8") as f:
        for t in all_trajectories:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

    problems_by_question = {p.question: p for p in problems}
    accepted = [t for t in all_trajectories if t.verifier_result]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in accepted:
            problem = problems_by_question[t.question]
            turns = to_conversation(problem)
            turns[-1] = {"role": "assistant", "content": t.answer}
            f.write(json.dumps(
                {"turns": turns, "answer": problem.answer, "kind": problem.kind, "source_model": t.teacher},
                ensure_ascii=False,
            ) + "\n")


def run_one_teacher(teacher: Teacher, problems) -> List[Trajectory]:
    """Pull (if needed) -> generate -> delete (if freshly pulled) for one
    teacher. Never raises -- any failure at any stage (pull, generation,
    delete) is caught and logged, and this returns whatever trajectories
    were collected before the failure (possibly none), so one teacher
    going wrong can't take the rest of the roster down with it."""
    print(f"\n=== {teacher.name} ===")
    was_present = True
    try:
        if teacher.provider == "ollama":
            was_present = is_present(teacher.model)
            if not was_present:
                print("  not present locally, pulling ...")
                t0 = time.time()
                pull_model(teacher.model)
                print(f"  pulled in {time.time() - t0:.1f}s")
            else:
                print("  already present locally (pre-existing -- will NOT be deleted)")

        trajectories = run_teacher_on_problems(teacher, problems)
        n_correct = sum(1 for t in trajectories if t.verifier_result)
        print(f"  {n_correct}/{len(trajectories)} verified correct "
              f"({n_correct / max(1, len(trajectories)) * 100:.0f}%)")
    except Exception as e:
        print(f"  [skip] {teacher.name} failed entirely: {e}")
        return []

    if teacher.provider == "ollama" and not was_present:
        print(f"  deleting {teacher.model} (freshly pulled by this run) ...")
        try:
            delete_model(teacher.model)
        except OllamaManagerError as e:
            print(f"  [warning] delete failed: {e}")

    return trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-problems", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--category", type=str, default="reasoning",
                         help="only teachers with this capability are used (see teachers/roster.py)")
    parser.add_argument("--out", type=str, default="data/teacher_reasoning_dataset.jsonl")
    parser.add_argument("--trajectories-out", type=str, default="data/teacher_trajectories.jsonl")
    parser.add_argument("--skip-claude", action="store_true", help="exclude the Anthropic teacher from this run")
    parser.add_argument("--skip-ollama", action="store_true", help="exclude all Ollama teachers from this run")
    args = parser.parse_args()

    load_env()
    roster = build_roster()
    if args.skip_claude:
        roster = [t for t in roster if t.provider != "anthropic"]
    if args.skip_ollama:
        roster = [t for t in roster if t.provider != "ollama"]
    teachers = teachers_for_category(roster, args.category)
    print(f"Full roster: {[t.name for t in roster]}")
    print(f"Teachers for category={args.category!r}: {[t.name for t in teachers]}")
    if not teachers:
        print("No teachers match this category -- nothing to do.")
        return

    problems = generate_dataset(args.n_problems, seed=args.seed, kinds=["arithmetic", "linear_equation"])
    print(f"Generated {len(problems)} verifiable problems (weighted toward the Phase 10.5 baseline's weak spot)")

    out_path = Path(args.out)
    traj_path = Path(args.trajectories_out)
    all_trajectories: List[Trajectory] = []

    for teacher in teachers:
        all_trajectories.extend(run_one_teacher(teacher, problems))
        write_outputs(all_trajectories, problems, out_path, traj_path)  # checkpoint after every teacher

    print(f"\n{'=' * 60}")
    print("Per-teacher acceptance rate:")
    by_teacher = {}
    for t in all_trajectories:
        s = by_teacher.setdefault(t.teacher, [0, 0])
        s[1] += 1
        s[0] += int(t.verifier_result)
    for name, (correct, total) in by_teacher.items():
        print(f"  {name:28s}: {correct}/{total} ({correct / max(1, total) * 100:.0f}%)")

    n_accepted = sum(1 for t in all_trajectories if t.verifier_result)
    print(f"\nWrote {len(all_trajectories)} raw trajectories (incl. rejected) -> {traj_path}")
    print(f"Wrote {n_accepted} verified SFT examples -> {out_path}")


if __name__ == "__main__":
    main()
