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
- [x] **Phase 5 — Reasoning data + verifiers**: synthetic self-verifying
      problem generators (arithmetic, linear equations, transitive
      comparisons), a genuine AST-walk arithmetic sandbox (whitelist of
      +-*/ etc. only -- no `eval`, no name/attribute/call access, can't
      reach `__import__`/`open`/anything), answer extraction, and
      reward/grading. 49 tests: sandbox correctness + adversarial-input
      rejection, generator self-consistency (each kind cross-checked by
      independently recomputing the answer), extraction edge cases,
      known-right/known-wrong grading, and a 300-problem batch where every
      problem's own solution must grade correct. Real run: generated 3000
      problems (100% self-verified), baseline-evaluated the Phase 4 SFT
      checkpoint on 30 held-out problems (6.7% -- near chance, it never
      saw math-formatted data), then fine-tuned further on the 3000
      reasoning conversations (val_loss 3.17 -> 0.15) and re-evaluated on
      60 *fresh* problems (different seed, unseen during fine-tuning):
      **40.0% overall** (comparison/logic 100% [19/19] -- genuine
      transitive reasoning on novel name combinations, not memorization;
      linear_equation 13.0%, arithmetic 11.1% -- it learned the *format*
      of showing work perfectly but gets the actual arithmetic wrong,
      e.g. computing `-86 - (-14)` as `-78` instead of `-72`). This is a
      well-documented small-transformer failure mode (procedure learned
      before exact computation) and is exactly the gap Phase 7's
      tool-calling (a real calculator) is meant to close -- not something
      to try to train away at this scale.
- [x] **Phase 6 — Preference optimization (DPO)**: chosen/rejected pair
      training against a frozen reference model, no reward model, no RL
      loop (`training/dpo.py`). Preference pairs built on Phase 5's
      verifier (`verifiers/preference.py`): chosen = a problem's real
      verified-correct solution, rejected = the same solution corrupted
      at the final step (wrong number / flipped conclusion) -- so
      "better" is verifiable by construction, same as Phase 5's data.
      13 tests: preference-pair self-consistency, `sequence_logprob`
      matches independent manual log-softmax-gather computation and
      correctly excludes masked (non-assistant) positions, the core DPO
      invariant (loss = ln(2) exactly when policy==reference, regardless
      of data), and an end-to-end overfit test. 164/164 tests passing
      overall. **Real run and an honest negative result**: DPO'd the
      Phase 5 reasoning-tuned checkpoint on 2850 preference pairs, 1000
      steps -- the *preference* objective moved exactly as it should
      (val_loss ln(2)=0.693 -> ~0.34-0.42, accuracy 75% -> ~85%, margin
      0.0 -> ~2-3), but re-evaluating actual generation accuracy on 60
      fresh reasoning problems went **40.0% -> 33.3%** (arithmetic
      specifically dropped 11.1% -> 0%). Optimizing the preference proxy
      did not transfer to better task accuracy here, and arguably hurt
      it slightly -- a real, well-documented DPO/RLHF failure mode
      (reward/preference metrics improving while held-out task
      performance doesn't follow), not a bug in the implementation. At
      10M params with a narrow, synthetic preference set, this isn't
      surprising; noted here rather than smoothed over. The Phase 5
      reasoning checkpoint (`neuraquill-10m-reasoning.pt`, 40.0%) remains
      the better checkpoint to build on going forward.
- [x] **Phase 7 — Tool calling**: a 4-token wire format
      (`<|tool_call|>{json}<|/tool_call|>` generated by the model,
      `<|tool_result|>...<|/tool_result|>` injected by the orchestrator,
      never generated), a `Tool`/`ToolRegistry` interface where nothing
      ever raises (bad args, unknown tool, internal exceptions all become
      an error string fed back to the model instead of crashing), and a
      `ToolCallOrchestrator` generate<->execute<->continue loop capped at
      `max_tool_calls`. The calculator tool reuses Phase 5's AST-walk
      sandbox directly. SFT masking extended (`training/tool_sft.py`) so
      the model is trained to produce the call but never trained to
      *predict* the injected result. 29 tests, including a scripted-model
      test double that exercises the orchestrator's control flow
      (chained calls, the max-calls cap, malformed JSON, unknown tool
      name) deterministically without needing a trained model. 193/193
      tests passing overall. **Real run**: generated 3000 tool-use
      conversations (arithmetic/linear-equation solved via the tool,
      comparison left as plain reasoning), extended the tokenizer +
      resized the Phase 5 reasoning checkpoint, trained (val_loss ->
      ~0.0001 -- the templated call format is easy to fit), then
      evaluated through the *real* orchestrator + real calculator on 60
      fresh problems: **98.3% (59/60)**, vs. 40.0% (Phase 5) / 33.3%
      (post-DPO) without tool use. arithmetic 94.4%, linear_equation
      100%, comparison 100%. The one miss is itself informative: for
      "39 * 30" the tool call correctly computed 1170 and the result was
      injected correctly, but the model's closing sentence said "1717"
      -- a copy/transcription slip *after* getting the right answer, not
      a computation error, a different and much smaller failure mode
      than the arithmetic errors seen pre-tool-use.
- [ ] **Phase 8 — RAG**: chunking, embeddings, a vector store (pgvector or
      local), retrieval. Tests: retrieval precision on a known small
      corpus, chunking boundary correctness.
- [ ] **Phase 9 — Agents**: planner/worker/verifier loop built on the tool
      + RAG infrastructure. Tests: end-to-end task completion on scripted
      scenarios.
- [ ] **Phase 10 — Scale up**: 10M → 100M → beyond, as compute allows.

Each phase's tests live in `tests/test_phase<N>_<name>.py` and must pass
(`pytest tests/ -v`) before the corresponding checkbox is marked done.
