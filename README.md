# NeuraQuill — a custom LLM built from scratch

This repo implements a GPT-style decoder-only transformer LLM entirely from
first principles (no HuggingFace `transformers`/`tokenizers` model code),
then progressively layers on instruction tuning, reasoning training, tool
use, RAG, and agents.

Build methodology: **each phase is implemented, then fully tested, before
the next phase begins.** See [PLAN.md](PLAN.md) for the full roadmap and
current status.

## Status

| Phase | Component | Status |
|---|---|---|
| 1 | Byte-level BPE tokenizer | ✅ implemented + tested (37 tests) |
| 2 | Transformer architecture (attention, MLP, blocks) | ✅ implemented + tested (22 tests) |
| 3 | Training loop (NeuraQuill-10M pretraining) | ✅ implemented + tested (16 tests) + real run |
| 4 | Instruction tuning (SFT + chat) | ✅ implemented + tested (19 tests) + real run |
| 5 | Reasoning data + verifiers | ✅ implemented + tested (49 tests) + real run |
| 6 | Preference optimization (DPO) | ✅ implemented + tested (13 tests) + real run (negative result) |
| 7 | Tool calling | ✅ implemented + tested (29 tests) + real run |
| 8 | RAG | ✅ implemented + tested (23 tests) + real run |
| 9 | Agents | ✅ implemented + tested (20 tests) + real run |
| 10 | Scale up (100M) | ✅ implemented + tested (3 tests) + real run |
| 10.5 | Baseline evaluation | ✅ `scripts/baseline_eval.py` + `reports/*.json` |
| 11 | Teacher data factory | ✅ implemented + tested (48 tests) + real run (mixed result, honestly reported) |
| 12A | Larger verified dataset + spend-tracked Claude | ✅ real run (925 examples, 7 local teachers) |
| 12B | Controlled SFT experiments (baseline/original/teacher/mixed) | ✅ real run (informative negative result) |
| 12C | Failure taxonomy (diagnose before training more) | ✅ real run (clean, decisive result) |
| 12D | Teacher data normalization | ✅ real run (901/925 kept, caught a real data bug) |
| 12E | Controlled distillation sweep (0/10/20/30%) | ✅ real run (monotonic negative result) |
| 12G | External intelligence, confirmed on the 12C set | ✅ real run (99% vs. 38-44%) |

**348/348 tests passing.**

## Phase 11: teacher data factory

A `Teacher` interface over local Ollama models (no API key) and Claude
(API), routed by capability, generating verified reasoning explanations
-- see `teachers/` and `scripts/generate_teacher_data.py`. Real run: 25
problems x 6 teachers, **109/124 local-teacher examples verified correct
(87.9%)**, several teachers clearing NeuraQuill's own raw reasoning
accuracy by a wide margin (qwen3.5:9b and deepseek-r1:8b both hit 100%).
Claude Opus 5 contributed 0 examples -- API billing/credits issue, not a
code problem. Fine-tuning NeuraQuill-100M on the 109 examples was then a
**real negative result**: reasoning accuracy went 36.7% → 20.0%, an
overfitting regression from too little data spread across too many
writing styles -- reported as-is, checkpoint not adopted. Full write-up,
including two real infra bugs found and fixed mid-run, in PLAN.md Phase 11.

## Phase 12A/12B: more data didn't fix it -- a sharper finding

Added 2 new local teachers (`glm4:9b`, `command-r7b`) and a hard-enforced,
persistent Anthropic spend tracker (`teachers/spend_tracker.py` -- checked
*before* every call, never overspends). Scaled the teacher run to 150
problems x 7 teachers: **925/1044 verified (88.6%)**, 8.5x Phase 11's
volume. Then ran a controlled 4-way SFT comparison (same checkpoint, same
recipe, only the data mixture varied) via `scripts/evaluate_and_gate.py`:

| condition | reasoning accuracy |
|---|---|
| no reasoning tuning | 1.7% |
| original data only | 25-26.7% |
| teacher data only (833 examples) | **0.0%** |
| mixed (3729 examples) | 10-16.7% |
| reference (original recipe) | **36.7%** |

All four rejected by the evaluation gate. **More teacher data made things
worse, not better** (0.0% vs. Phase 11's 20.0% with 8x less data), ruling
out "just needed more volume." Real generation transcripts confirm actual
incoherence, not an extraction artifact -- and the training loss curve
looked perfectly healthy throughout, meaning a clean val-loss curve does
not certify a small model retained the skill being measured. Likely
mechanism and next steps in PLAN.md Phase 12B. Reference checkpoint
(36.7%) remains the best, confirmed twice now.

## Phase 12C: diagnose before training more

Before generating any more teacher data, `scripts/diagnose_failures.py`
classified every miss the reference checkpoint made on 100 fresh held-out
problems (two independent runs, 38% and 44% accuracy, same taxonomy both
times): **100% of failures were arithmetic/linear_equation** (comparison
stayed ~100% correct, as always); **zero** were unparseable output or
degenerate repetition -- the model always produces a syntactically valid
number. ~57% were close misses (within 5), ~43% wild misses. Reading the
transcripts: the *procedure* is consistently right, the *arithmetic
executed at each step* isn't -- sometimes not even internally consistent
with the model's own stated intermediate value. **This checkpoint's
entire remaining gap is one thing: unreliable exact arithmetic, not
comprehension, format, or logic** -- exactly what Phase 7's tool-calling
already fixes (98.3%). More SFT data isn't the lever for this particular
gap; the diagnostic just proved it with evidence instead of leaving it as
an assumption.

## Phase 12D/12E/12G: normalize, sweep, confirm

**12D** (`teachers/normalize.py`) strips markdown/LaTeX formatting from
verified teacher explanations and guarantees an explicit answer
statement. Building it surfaced a real, separate data-quality bug: ~3% of
Phase 12A's accepted examples (19% of `command-r7b`'s specifically) were
verified "correct" only because a `max_tokens`-truncated response
happened to contain the right number somewhere -- dropped rather than
patched. 901/925 examples survived (97.4%).

**12E** swept 0% / 10% / 20% / 30% *normalized* teacher-data blended into
the original dataset, same recipe, same held-out set:

| teacher % | accuracy |
|---|---|
| 0% | 25-27% |
| 10% | 20.0% |
| 20% | 15.0% |
| 30% | 15.0% |
| 100% (raw) | 0.0% |
| reference | **36.7%** |

**Monotonic degradation at every ratio tested**, even with 12D's genuine
cleanup applied. No sweet spot exists in this sweep. That shifts the most
likely explanation from "fixable data-engineering problem" toward "100M
is near its capacity ceiling for absorbing stylistically diverse
reasoning data" -- exactly what a future capacity-scaling experiment
would test directly instead of guessing.

**12G** confirmed the actual fix on the *identical* 100-problem set 12C's
diagnostic used: the Phase 7 tool-tuned checkpoint scores **99%** where
the raw reasoning checkpoint scored 38-44%. Not a different measurement
compared against an old number -- the same test, closing the loop.

## NeuraQuill-10M vs NeuraQuill-100M (Phase 10.5 baseline)

Same tokenizer, same datasets, same training pipeline for every stage
(pretrain → SFT → reasoning → tools) -- only the model config changed.
Full report JSON in `reports/`.

| metric | 10M | 100M |
|---|---|---|
| perplexity | 45.8 | **33.5** |
| reasoning accuracy (no tools) | 40.0% | 36.7% |
| tool-calling accuracy | 98.3% | 98.3% |
| RAG retrieval precision@1 | 100% | 100% |

Scale bought a real perplexity win, but **not** better raw reasoning or
more reliable tool use -- ~8M pretraining tokens is far too little for a
99M-param model to reach fluent generation (parameters and tokens both
need to scale together), and tool-calling accuracy is identical at both
sizes because it only requires learning a templated call format, not
exact computation. See PLAN.md Phase 10.5 for the full write-up.

NeuraQuill-10M's real runs (RTX 4060):
- Pretraining: 4000 steps on 2.375M filtered tokens from 15 public-domain
  books, val_loss 8.98 → 3.82 (perplexity ≈ 46), ~15 min.
- SFT: 3000 steps on 13,378 filtered Dolly-15k conversations, val_loss
  8.05 → 3.70, ~12 min. `scripts/chat.py` gives correctly-formatted,
  correctly-terminated chat responses that aren't yet coherent/factual --
  expected at this scale, not a pipeline bug.
- Reasoning: fine-tuned further on 3000 self-verifying synthetic
  problems, then evaluated on 60 fresh unseen problems: **6.7% → 40.0%**
  accuracy. Logic/comparison reasoning hit 100% (genuine transitive
  reasoning, not memorization); arithmetic stayed weak (~11%) because the
  model learned the *procedure* but not exact computation -- the gap
  tool-calling (Phase 7) is meant to close.
- DPO: fine-tuned the reasoning checkpoint on 2850 self-verified
  preference pairs. The preference objective itself moved correctly
  (val_loss 0.693 → ~0.35-0.42, accuracy 75% → ~85%, margin 0 → ~2-3),
  but held-out reasoning accuracy went **40.0% → 33.3%** -- a real
  negative result (reward-proxy gains not transferring to task
  accuracy), reported as-is rather than smoothed over. See PLAN.md.
- Tool calling: trained the model to hand arithmetic off to a real
  calculator tool (a 4-token call/result wire format + an orchestrator
  loop). Evaluated through the *real* orchestrator on 60 fresh problems:
  **98.3%** (59/60), vs. 40.0% (Phase 5) / 33.3% (post-DPO) without tool
  use -- directly closing the gap those phases identified.
- RAG: from-scratch TF-IDF retrieval (no sklearn), 100% precision@1 on a
  known 10-doc corpus, still 100% at 30,177 chunks after mixing in the
  real pretraining corpus as noise. Generation over retrieved context is
  weak/garbled -- the same 10M-param ceiling Phase 4 established, not a
  RAG-specific issue.
- Agents: a planner/router/worker/verifier loop dispatching to two
  *different* specialized checkpoints (Phase 7's tool-tuned model for
  reasoning, Phase 4's SFT model + Phase 8's index for knowledge).
  Routing 100% correct (8/8 subtasks); reasoning-via-tools 3/4;
  knowledge-via-RAG 0/4 -- the architecture works end-to-end, model
  scale/fluency is the limiter, consistent with earlier phases.
- Content filter verified with zero blocklist-term matches in both
  training corpora and in sampled generations.

## Repo layout

```
neuraquill/
├── tokenizer/       # Phase 1 — byte-level BPE, pure Python, no deps
├── model/           # Phase 2 — transformer architecture (PyTorch)
├── training/         # Phase 3+ — pretraining / SFT / DPO loops
├── data/             # corpora
├── tests/            # pytest suite, one file per phase
├── scripts/          # CLI entry points (train_tokenizer.py, etc.)
└── configs/           # model size configs (10m.yaml, 100m.yaml, ...)
```

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```
