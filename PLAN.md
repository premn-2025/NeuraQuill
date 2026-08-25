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
- [x] **Phase 8 — RAG**: word-based chunking with overlap
      (`rag/chunking.py`), a from-scratch TF-IDF vectorizer + cosine
      similarity (`rag/embeddings.py`, no sklearn -- deliberately not
      NeuraQuill's own embeddings, which would be a noisy signal at 10M
      params), a local in-memory vector store, and a retriever tying them
      together. 23 tests: chunking boundary correctness (full coverage,
      exact overlap math, no duplicate/degenerate spans), TF-IDF
      correctness (identical docs -> similarity 1.0, common terms get
      lower idf than rare ones, OOV query terms ignored not erroring),
      and **retrieval precision@1 = 100%, recall@3 = 100%** on a
      10-document known corpus (the explicit PLAN.md requirement) plus a
      chunk-level (not document-level) granularity test. 216/216 tests
      passing overall. **Real run**: mixed those same 10 invented facts
      (things NeuraQuill could not possibly know) into 754 pseudo-
      documents of the real Phase 3 pretraining corpus -> a 30,177-chunk
      haystack -- retrieval precision@1 stayed **100%** at that scale.
      Then a qualitative end-to-end demo (`scripts/rag_chat.py`):
      retrieval correctly surfaces the exact right fact every time
      (e.g. "X200 ... weighs exactly 300 grams" at similarity 0.308 vs.
      0.085 for the next result), but the SFT checkpoint's *composition*
      of that context into a fluent answer is weak/garbled -- the same
      generation-quality ceiling Phase 4 already established at this
      model scale, not a new RAG-specific problem. Retrieval mechanism:
      solid. Generation quality: still the 10M-param bottleneck.
- [x] **Phase 9 — Agents**: a planner (`agents/planner.py`, splits
      compound tasks on connective phrases) -> router (`agents/router.py`,
      classifies each subtask "reasoning" vs "knowledge") -> worker
      (`ReasoningWorker` wraps Phase 7's tool orchestrator,
      `RAGWorker` wraps Phase 8's retriever) -> verifier
      (`verify_reasoning` reuses Phase 5's grader, `verify_knowledge_
      grounded` checks the expected fact substring is present) loop.
      Router and planner are deliberately rule-based, not LLM-driven --
      Phases 5/6 already showed this 10M model's free-form judgment is
      unreliable at this scale; a learned router/planner is natural
      future work once the base model scales up (Phase 10). 20 tests
      using the same scripted-model-double technique as Phase 7,
      including end-to-end scripted scenarios (single reasoning task,
      single knowledge task, compound multi-subtask task, a deliberately
      wrong subtask to prove the verifier catches failure, verifier-count
      mismatch). 236/236 tests passing overall. **Real run**
      (`scripts/agent_demo.py`): two *different* real checkpoints as two
      specialized workers (Phase 7's tool-tuned model for reasoning,
      Phase 4's SFT model + Phase 8's retrieval index for knowledge) on
      6 scripted scenarios. Routing was **100% correct** on every
      subtask (8/8). Reasoning-via-tools: 3/4 correct (one interesting
      failure -- a degenerate token-repetition loop before ever
      completing a valid tool call, a known small-model failure mode).
      Knowledge-via-RAG: 0/4 correct -- not a retrieval failure (Phase 8
      already showed retrieval is reliable), but the same
      generation-fluency ceiling Phase 4/8 established for this
      checkpoint at 10M params. The agent *architecture* -- planning,
      routing, dispatching to the right specialized checkpoint, tool
      execution, verification -- worked correctly end-to-end throughout;
      what limits scenario pass rate is model scale/quality, consistent
      with (not contradicting) every earlier phase's findings.
- [x] **Phase 10 — Scale up (100M)**: `configs/100m.yaml` (13 layers, 12
      heads, 768 dim, 98.8M params, same tokenizer/block_size as the 10M
      config -- config-only scaling, no architecture changes). Added
      gradient accumulation to `training/pretrain.py` (needed: batch_size
      16 at this size thrashed into slow Windows "shared GPU memory"
      spillover on the RTX 4060's 8GB; batch_size=8 x grad_accum_steps=4
      = effective batch 32 fits cleanly at ~6GB). 3 new tests including a
      rigorous check that accumulated gradients exactly match a single
      large-batch gradient. Corpus expanded from 15 to 35 public-domain
      books (8.5MB -> 26.3MB, 2.38M -> 7.96M tokens), reusing the *same*
      trained tokenizer (deliberate -- avoids re-running the slow
      pure-Python BPE trainer at larger scale). **Real run**: pretrain
      (4000 steps, 76.3 min) val_loss 8.94 -> best **3.44** at step 3000
      (train loss kept falling after that while val loss crept back up --
      overfitting, honestly visible and correctly handled by best-checkpoint
      saving, not hidden). Then SFT (Dolly, 18 min) val_loss 8.20 -> 3.49,
      reasoning fine-tune (11 min) val_loss 2.77 -> 0.14, tool fine-tune
      (11 min) val_loss -> ~0.002 -- all four stages reused the *exact*
      same datasets/tokenizers as the 10M runs, config-only scaling
      confirmed working end-to-end.
- [x] **Phase 10.5 — Baseline evaluation**: `scripts/baseline_eval.py`,
      a permanent reusable report covering perplexity, reasoning (Phase 5),
      tool-calling (Phase 7), and RAG retrieval (Phase 8) -- consolidating
      those phases' eval code rather than reimplementing it, with results
      written to `reports/*.json` (model version, step, per-category
      accuracy + raw counts). Honestly lists what's *not* yet measured
      (general knowledge QA, coding, hallucination calibration) instead of
      skipping the topic. **10M vs 100M, side by side**:

      | metric | 10M | 100M |
      |---|---|---|
      | perplexity | 45.8 | **33.5** |
      | reasoning, no tools | 40.0% | 36.7% |
      | tool calling | 98.3% | 98.3% |
      | RAG precision@1 | 100% | 100% |

      The headline finding: **perplexity improved substantially with
      scale (a real, measured win), but neither raw reasoning accuracy
      nor tool-calling reliability did** -- reasoning was flat-to-slightly-down
      (noise-level; both models are still bad at raw arithmetic, ~0-11%),
      and tool-calling was identical at 98.3% both sizes, because tool
      use only requires learning a templated call format, which a 10M
      model already does as reliably as a 100M one -- the actual
      computation is offloaded to the deterministic sandbox either way.
      Sample chat generations at 100M are still largely incoherent (one
      new observation: a "rainbow colors" association surfaced, misapplied
      to the wrong question -- a hint of more specific factual encoding
      than 10M showed, still not coherent). **Conclusion, stated plainly**:
      ~8M pretraining tokens is nowhere near enough for a 98.8M-parameter
      model to reach fluent generation (consistent with Chinchilla-style
      scaling laws -- tokens must scale with parameters, not just one or
      the other), and reasoning capability doesn't come from parameter
      count alone at this scale -- tool-calling remains the load-bearing
      fix for exact computation, not more layers.
- [ ] **Phase 11 — Teacher data factory**: use stronger models to generate
      targeted synthetic training data (reasoning, coding, tool-use
      trajectories), prioritized by the Phase 10.5 baseline's weakest
      categories rather than generated blindly. **Needs a decision before
      starting**: which teacher model(s) are actually available (API keys/
      budget) -- this session can act as one teacher directly (no external
      API needed) but a genuine multi-teacher ensemble (Gemini/Qwen/Llama/
      etc.) requires credentials this environment doesn't have configured.
- [ ] **Phase 12 — Verifiable RL**: extend Phase 5/6's verifier-driven
      approach with an actual RL loop (not just DPO) for math/logic/coding,
      using a code-execution sandbox (a real one, with process isolation/
      timeouts -- distinct from Phase 5's arithmetic-only AST sandbox) for
      coding rewards. Every experiment reports both the training metric
      *and* the downstream task metric, since Phase 6 already showed those
      can diverge.
- [ ] **Phase 13 — Adaptive learning**: a stable core model + fast-changing
      memory + specialized adapters, gated by a "Learning Gate" (novelty /
      reliability / usefulness / verifier score) that decides whether an
      experience gets ignored, stored in memory, or used to train an
      adapter -- and adapters are kept only if they measurably improve the
      Phase 10.5 baseline, discarded otherwise. This is the project's most
      distinctive research direction, not a small feature.
- [ ] **Phase 14 — Memory + consolidation**: long-term memory separate from
      web/RAG (Phase 8, current) and core weights (stable), with periodic
      consolidation of repeatedly-useful verified memory into trained
      adapters -- gated the same way as Phase 13, improvement-or-discard.
- [ ] **Phase 15 — Advanced agents**: extend Phase 9's planner/router/
      worker/verifier loop toward multi-step goal -> plan -> execute ->
      observe -> correct -> verify workflows with specialized agents
      (research/coding/analysis), not just single-subtask dispatch.
- [ ] **Phase 16 — Real-time knowledge**: query-time web search + source
      cross-checking + citation, separate from the static Phase 8 RAG
      index -- freshness without continuously retraining the base model.
- [ ] **Phase 17 — Multimodal**: a vision encoder + projection into
      NeuraQuill's embedding space, extending the text-only pipeline.
- [ ] **Phase 18 — Scale further**: 100M -> 300M -> 1B -> 3B+, same
      tokenizer/pipeline, config-driven, as compute allows.
- [ ] **Phase 19 — Advanced training/efficiency**: attention variants
      (GQA/MQA/RoPE), curriculum learning, mixed precision/quantization,
      distributed training, and inference-side optimization (KV-cache,
      speculative decoding) once scale makes these worth the complexity.

Each phase's tests live in `tests/test_phase<N>_<name>.py` and must pass
(`pytest tests/ -v`) before the corresponding checkbox is marked done.
Every training experiment from Phase 10 onward records both its training
metric and its real downstream task metric -- Phase 6's DPO result (the
preference objective improved while task accuracy got worse) is the
reason this discipline is non-negotiable going forward.
