# NeuraQuill Roadmap

Methodology: implement a phase → write a full test suite for it → all tests
green → only then move to the next phase. No phase is "done" without tests.

- [x] **Phase 1 — Tokenizer**: byte-level BPE, trained from scratch on a
      corpus, no external tokenizer libs. Tests: round-trip correctness on
      ASCII/unicode/emoji/code, determinism, save/load, special tokens,
      fuzzing.
- [x] **Phase 2 — Transformer architecture**: embeddings, multi-head causal
      self-attention, MLP, LayerNorm, residual blocks, LM head, all
      hand-written in PyTorch (no `nn.MultiheadAttention`, no HF). Tests:
      shape checks, causal-mask correctness (no leakage from future
      tokens), gradient flow, parameter count matches config, overfit test
      on a tiny batch (loss → ~0). NeuraQuill-10M config verified at 10.18M
      params, runs forward on the RTX 4060 via CUDA.
- [x] **Phase 3 — Pretraining loop (NeuraQuill-10M)**: dataloader, AdamW +
      cosine LR schedule, checkpointing, CUDA training on the RTX 4060,
      perplexity eval, sampling/generation (temperature, top-k). Tests (16,
      on synthetic data): batching/target-shift correctness, LR schedule
      shape, decay/no-decay optimizer split, checkpoint save/resume
      fidelity, loss-decreases end-to-end, perplexity math. Real run: 15
      public-domain Gutenberg books (8.5MB) -> 2.38M tokens -> 4000 steps
      on the RTX 4060 (~15 min), val_loss 9.01 -> 3.79 (perplexity ~44).
      **Resolved**: added `data/content_filter.py` (paragraph-level
      blocklist filter, 8 tests) wired into `scripts/prepare_data.py` by
      default. Dropped 166/133,383 paragraphs (0.12%) from the corpus.
      Retrained NeuraQuill-10M from scratch on the cleaned data: val_loss 8.98
      -> 3.82 (perplexity ~46), consistent with the unfiltered run. Verified
      zero blocklist matches in the cleaned corpus and in 1500 tokens of
      sampled generation across 5 prompts. The filter is still just a word
      list, not a real toxicity classifier -- treat it as a first gate,
      not a guarantee, and revisit with a trained classifier if this ever
      needs to be a serious quality bar (e.g. before Phase 4 SFT data or
      anything user-facing).
- [x] **Phase 4 — Instruction tuning (SFT)**: chat template
      (`<|system|>`/`<|user|>`/`<|assistant|>` + `<|endoftext|>` per turn),
      masked loss (`ignore_index=-1` on every non-assistant token),
      post-pretraining vocab extension + embedding resize (old rows
      copied exactly, new rows freshly initialized, weight tying
      preserved), early generation stopping at a stop token. 19 tests:
      masking correctness, padding/truncation, resize correctness
      (old-vocab logits provably unchanged), stop-token generation,
      overfit-a-tiny-instruction-set end-to-end. Real run: Dolly-15k
      (13,378 conversations after content-filtering + length-filtering) ->
      SFT'd NeuraQuill-10M from the Phase 3 checkpoint, val_loss 8.05 -> 3.70
      over 3000 steps (~12 min, RTX 4060). `scripts/chat.py` confirms the
      *mechanics* work end-to-end: correct chat-template formatting,
      correct early-stopping at `<|endoftext|>`, topic-relevant
      vocabulary in responses -- but content is not yet coherent/factual,
      exactly as expected for a 10M-param model with ~15min pretrain +
      ~12min SFT. Scale (Phase 10) is what closes that gap, not this
      phase's job.
- [ ] **Phase 5 — Reasoning data + verifiers**: synthetic verifiable math/
      logic problems, a Python-sandbox verifier, reward computation. Tests:
      verifier correctly grades known right/wrong solutions.
- [ ] **Phase 6 — Preference optimization (DPO)**: chosen/rejected pair
      training. Tests: DPO loss decreases, chosen logprob rises relative
      to rejected on a toy pair set.
- [ ] **Phase 7 — Tool calling**: structured tool-call format + orchestrator
      that executes tools and feeds results back. Tests: correct tool JSON
      parsing, execution round-trip, error handling on bad tool output.
- [ ] **Phase 8 — RAG**: chunking, embeddings, a vector store (pgvector or
      local), retrieval. Tests: retrieval precision on a known small
      corpus, chunking boundary correctness.
- [ ] **Phase 9 — Agents**: planner/worker/verifier loop built on the tool
      + RAG infrastructure. Tests: end-to-end task completion on scripted
      scenarios.
- [ ] **Phase 10 — Scale up**: 10M → 100M → beyond, as compute allows.

Each phase's tests live in `tests/test_phase<N>_<name>.py` and must pass
(`pytest tests/ -v`) before the corresponding checkbox is marked done.
