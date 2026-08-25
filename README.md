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
| 5+ | Reasoning/verifiers, DPO, tools, RAG, agents | not started |

**102/102 tests passing.** NeuraQuill-10M's real runs (RTX 4060):
- Pretraining: 4000 steps on 2.375M filtered tokens from 15 public-domain
  books, val_loss 8.98 → 3.82 (perplexity ≈ 46), ~15 min.
- SFT: 3000 steps on 13,378 filtered Dolly-15k conversations, val_loss
  8.05 → 3.70, ~12 min. `scripts/chat.py` gives correctly-formatted,
  correctly-terminated chat responses that aren't yet coherent/factual --
  expected at this scale, not a pipeline bug.
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
