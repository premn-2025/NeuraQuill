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

**241/241 tests passing.**

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
