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
- [x] **Phase 11 — Teacher data factory**: a `Teacher` interface
      (provider/model/capabilities) over Ollama (local, no API key) and
      Anthropic (cloud), a roster built from `.env` with capability-based
      routing (`teachers/roster.py` -- not every teacher answers every
      question category), and rich per-(problem, teacher) `Trajectory`
      records (question, teacher, prompt_version, answer, verifier_result,
      extracted_answer, quality_score, disagreement_score) rather than
      flat Q&A pairs. Design choice, stated plainly: teachers are never
      trusted to invent new problems+answers from scratch (unverifiable);
      they write explanations for problems Phase 5's generators already
      know the correct answer to, and only explanations whose stated
      final answer verifies correct become SFT data -- diversity/quality
      from stronger models, without trusting anything unverifiable.
      Ollama models cycle pull -> generate -> delete -> next (never
      deletes a model that was already present before the run started).
      48 tests. 289/289 tests passing overall.

      **Real infrastructure bugs found and fixed during the actual run**
      (not just theoretical -- each one broke a real run before the fix):
      a Windows cp1252-vs-unicode crash in the `ollama pull` subprocess
      reader threads; multi-hundred-KB output-capture bloat from the
      redrawing progress bar (fixed by discarding stdout, not capturing
      it); an uncaught `subprocess.TimeoutExpired` on a stalled pull that
      crashed the *entire* run and nearly lost three teachers' already-
      collected results (fixed: caught + retried, and the script now
      checkpoints all results to disk after every single teacher, not
      once at the end, specifically so one teacher's failure can never
      erase another's success).

      **Real run**: roster = gemma3:4b, qwen3.5:9b, llama3.2:latest,
      mistral:latest, deepseek-r1:8b (all local via Ollama) + claude-opus-5
      (API). 25 verifiable problems (arithmetic + linear_equation, the
      Phase 10.5 baseline's weak spot) x 6 teachers. Local teachers:
      **109/124 accepted (87.9%)** -- gemma3:4b 92%, qwen3.5:9b 100%,
      llama3.2 84%, mistral 64%, deepseek-r1:8b 100% (24/24, one
      generation timeout skipped gracefully, not a crash) -- several
      clearing NeuraQuill's own raw reasoning accuracy (35-40%) by a wide
      margin, validating the core premise. **Claude Opus 5 contributed
      zero examples**: every call returned HTTP 400 "credit balance too
      low" -- a real account/billing gap, not a code or quality problem;
      the pipeline is ready the moment credits are added.

      **Then a second honest negative result, sharper than Phase 6's**:
      fine-tuned the Phase 10.5 100M reasoning checkpoint (36.7% baseline)
      on the 109 verified teacher examples (99 train / 10 val, 400 steps)
      and re-evaluated on the *same* held-out seed. Result: **20.0%** --
      a real regression, not noise. Train loss collapsed to 0.009 while
      val loss climbed back up after step 100 (textbook overfitting), and
      generations show it: garbled, confabulated arithmetic and
      degenerate repetition where the pre-finetune checkpoint was clean.
      99 examples spanning 5 very different teacher writing styles, at a
      peak LR tuned for a much larger/more uniform dataset, was enough to
      corrupt an already-converged checkpoint's output format rather than
      extend it. **This checkpoint is not adopted** -- the Phase 5/10.5
      reasoning checkpoints remain the best available. Lesson for next
      time this is revisited: either far more teacher volume, blending
      teacher data into the *original* template dataset instead of a
      pure continuation fine-tune, or a much gentler LR/step count with
      early stopping at the first val-loss uptick (step ~50, not 100).
- [ ] **Phase 12 — Verifiable reasoning + reward learning**, split into
      four sub-phases rather than jumping straight to RL, because Phase
      11's negative result showed the failure was training strategy
      (too little data, too little stylistic consistency, LR tuned for
      the wrong data scale), not data quality:
  - [x] **12A — Build a large, balanced, verified dataset.** Added 2 new
        local teachers to the roster (`glm4:9b`, `command-r7b:latest` --
        picked over larger 24B/30B alternatives an outside consultant
        suggested, rejected as too large for the RTX 4060's proven-safe
        ~8GB VRAM zone), and a persistent, hard-enforced Anthropic spend
        tracker (`teachers/spend_tracker.py`) since real API credits were
        wired in mid-phase -- checked *before* every call, refuses rather
        than overspends. Real run: 150 problems x 7 local teachers (no
        Claude -- account credits weren't actually active despite being
        enabled, a real billing-propagation issue, not a code problem) ->
        **925/1044 verified (88.6%)**, both new teachers strong (glm4:9b
        91%, command-r7b 95%) -- 8.5x the Phase 11 volume.
  - [x] **12B — Controlled SFT experiments**, same starting checkpoint
        (`neuraquill-100m-sft.pt`, pre-reasoning-tuning), same fixed
        recipe (600 steps, lr 3e-5, gentler than Phase 11's 8e-5) across
        conditions, varied only in data mixture. **Results, and a finding
        sharper than Phase 11's**:

        | condition | data | reasoning accuracy |
        |---|---|---|
        | A — no reasoning tuning | none | 1.7% |
        | B — original only | 2850 template examples | 25-26.7% |
        | C — teacher only | 833 verified teacher examples | **0.0%** |
        | D — mixed | 3729 (original + teacher) | 10-16.7% |
        | reference (Phase 5/10.5, original recipe) | 2850, 2000 steps @ lr 8e-5 | **36.7%** |

        All rejected by `scripts/evaluate_and_gate.py` against the 36.7%
        reference; none adopted. Two real findings, not one:

        1. **More teacher data made it *worse*, not better** (833 vs
           Phase 11's 109 -> 0.0% vs 20.0%), ruling out "just needed more
           volume" as the fix. Transcripts show genuine incoherence --
           hallucinated numbers unrelated to the question, degenerate
           repetition, procedures from the wrong problem type -- not an
           answer-extraction artifact (checked directly against raw
           generations, not just the accuracy number). C's val_loss curve
           looked perfectly healthy (monotonic 4.15 -> 0.76, no
           overfitting spike), yet downstream accuracy collapsed anyway
           -- **a clean validation-loss curve does not certify that a
           small model actually retained the skill being measured**,
           only that it fit the token distribution of that specific
           validation split. The likely mechanism: 833 examples spanning
           7 stylistically incompatible teacher "voices" (markdown vs.
           plain text, LaTeX vs. arithmetic notation, verbose vs. terse)
           gives a 100M model conflicting output-format signal it doesn't
           have the capacity to reconcile, corrupting coherent generation
           rather than extending it.
        2. **Condition B, same data as the reference, same starting
           checkpoint, undertrained relative to the reference recipe**
           (25-26.7% vs 36.7%) -- the deliberately gentler LR/step count
           chosen specifically to avoid repeating Phase 11's overfitting
           undertrained on this *uniform, single-format* dataset, where
           the original, more aggressive recipe (2000 steps @ 8e-5) was
           actually fine. The lesson isn't "always use a gentler
           recipe" -- it's that recipe and data consistency interact:
           aggressive hyperparameters are safe on stylistically uniform
           data and dangerous on heterogeneous data, and Phase 11
           conflated the two.

        **Next time teacher data is revisited**: normalize/rewrite
        teacher explanations into one consistent presentation format
        before training on them (keep the teachers' correct *content*,
        discard their raw stylistic diversity), or use teacher data as a
        much smaller proportion of the mix (closer to 1:10 than 1:3.4),
        rather than assuming either "more data" or "gentler LR" alone
        fixes what is actually a format-consistency problem. The Phase
        5/10.5 reasoning checkpoint (36.7%) remains the best available
        for this category, confirmed twice now.
  - [x] **12C — Diagnose.** `scripts/diagnose_failures.py`: classifies
        every held-out miss (no-answer-extracted / degenerate-repetition
        / close-miss <=5 off / wild-miss >5 off), broken down by problem
        kind. Two independent 100-problem runs against the reference
        checkpoint (38% and 44% accuracy -- consistent with the
        established range) gave the **same clean, decisive taxonomy both
        times**: 100% of failures fall in arithmetic/linear_equation
        (comparison/logic stayed at ~100% correct, as in every prior
        phase); **zero** were unparseable-output or degenerate-repetition
        failures -- the model always produces a syntactically valid
        number, every single time; failures split roughly 57%/43% between
        close misses (within 5) and wild misses. Reading the actual
        transcripts: the model's *procedure* is consistently correct
        (correctly restates the equation, correctly sequences "subtract,
        then divide") -- what's unreliable is the arithmetic executed at
        each step, sometimes so much so that the stated final answer
        doesn't even follow from the model's own stated intermediate
        value. **Conclusion: 100% of this checkpoint's remaining gap is
        one thing -- unreliable exact arithmetic execution, not
        comprehension, not format, not logic.** This is exactly what
        Phase 7 already fixed (98.3% via tool-calling, both model sizes)
        -- the diagnostic confirms *why* with hard evidence rather than
        inference, and says clearly: more SFT data (teacher or
        otherwise) is not the lever to pull for this gap; 12D-12F below
        remain worth doing for whatever residual isn't arithmetic (there
        wasn't any in this sample, but a coding/general-QA category would
        likely show a different taxonomy), while 12G (route arithmetic to
        the tool, as already built) is the actually-proven fix for what
        this diagnostic found.
  - [x] **12D — Teacher normalization.** `teachers/normalize.py`: strips
        markdown (`**bold**`, `## headers`) and LaTeX (`$$...$$`,
        `\(...\)`) syntax that doesn't match this project's tokenizer
        training distribution (Gutenberg prose + plain instructions, no
        markdown/LaTeX corpus); guarantees an explicit "The answer is X."
        statement rather than depending on the extractor's last-number
        fallback. Also caught a genuine, separate data-quality bug while
        building this: **~3% of Phase 12A's accepted examples were
        verified "correct" only because a truncated (`max_tokens`-cut-off)
        response happened to contain the right number somewhere** -- 19%
        of `command-r7b`'s examples specifically. `normalize_explanation`
        drops these outright rather than patching a bolted-on answer onto
        an interrupted reasoning chain. Applied to the 925 examples:
        **901 kept (97.4%)**, 0 failed re-verification after cleanup (24
        dropped, all truncated `command-r7b` cases). 21 tests.
  - [x] **12E — Controlled distillation sweep**: 0% / 10% / 20% / 30%
        *normalized* teacher-data blended into the original 3000-example
        dataset, same checkpoint/recipe as 12B, same held-out set.
        **Result: monotonic degradation at every single ratio tested**,
        even after 12D's real cleanup:

        | teacher % | reasoning accuracy |
        |---|---|
        | 0% (pure original) | 25-27% |
        | 10% (270 orig + 300 normalized teacher) | 20.0% |
        | 20% (2400 orig + 600 normalized teacher) | 15.0% |
        | 30% (2100 orig + 901 normalized teacher) | 15.0% |
        | 100% (raw, Phase 12B) | 0.0% |
        | reference (original recipe, 2000 steps) | **36.7%** |

        No sweet spot exists in this sweep -- every teacher fraction,
        however small, made the checkpoint *worse* than 0%, and 12D's
        genuine data-quality fixes (markdown/LaTeX stripped, truncated
        examples dropped) softened the collapse (15-20% vs. 12B's 0%
        at high ratios) but did not reverse the trend. **This shifts the
        most likely explanation from "fixable data-engineering problem"
        toward "100M is close to its capacity ceiling for absorbing
        stylistically diverse reasoning data at this training budget"** --
        which is exactly what 12H (capacity scaling) exists to test
        directly instead of guessing. Recommendation: stop iterating on
        teacher-blend ratio/cleanup for *this* model size -- diminishing
        and negative returns across four honestly-measured data points is
        a result, not a reason to try a fifth ratio. The proven, already-
        built fix for the arithmetic gap 12C actually diagnosed remains
        12G (tool-calling, 99% confirmed on the identical problem set).
  - [ ] **12F — Failure-driven training.** Once there's a working
        distillation recipe: run the current-best checkpoint on fresh
        unseen problems, classify the failures (reusing 12C's taxonomy),
        generate targeted teacher data for the specific failure modes
        found, verify, retrain, re-diagnose. A loop, not a one-shot.
  - [x] **12G — External intelligence for reasoning, confirmed directly
        against 12C's diagnostic.** Re-ran the *exact same* 100-problem
        set (same seed) that produced 12C's taxonomy through the Phase 7
        tool-tuned checkpoint + real orchestrator instead of the raw
        reasoning checkpoint: **99/100 (99%)** -- linear_equation 100%,
        comparison 100%, arithmetic 97.3% (1 miss). Not an inference from
        Phase 7's earlier number on a different set; this is the identical
        test 12C used, closing the loop with a like-for-like comparison.
        Confirms the diagnosis and the fix in one measurement: lean on
        Phase 7's tool-calling for arithmetic and Phase 8's RAG for
        knowledge questions rather than expecting a 100M model's raw
        weights to contain everything -- reserve the weights for
        classification/routing/concise reasoning, which is what they're
        actually positioned to do well.
  - [ ] **12H — Capacity scaling experiment**: the exact same dataset and
        recipe from 12E/12F run across 100M -> 300M -> 500M -> 1B (config
        change only, per Phase 10's proof that the pipeline scales this
        way) to find out empirically whether remaining headroom is a
        data problem or a capacity problem, rather than guessing. Target
        framed incrementally (36.7% -> 50% -> 65% -> ...), not jumping
        straight to a headline number -- a benchmark chased directly is a
        benchmark optimized, not a model that generalizes.
  - [ ] **12I — A real verifier/reward environment**: records
        (question, candidate, tool_calls, verification, reward) per
        attempt, not just question->answer; a real process-isolated
        code-execution sandbox for coding rewards (distinct from Phase
        5's arithmetic-only AST sandbox, which stays scoped to
        arithmetic); partial-credit rewards where a binary correct/wrong
        loses information (e.g. tool-call reward = correct tool +
        correct arguments + correct final result, each worth partial
        credit).
  - [ ] **12J — RL / reward optimization** on top of 12I's environment,
        only once it's built and produces sane rewards on known cases.
        Every experiment reports the training metric *and* the
        downstream task metric on fresh held-out problems, since Phase
        6 already showed those can diverge -- that discipline is
        non-negotiable from here on.

      **Held-out discipline going forward** (12H's whole premise depends
      on this): keep train / dev / a never-trained-on secret test set
      permanently separate. The 60-problem eval set used throughout
      Phases 5-12 has been dev-like (reused across many experiments to
      pick winners) -- a truly held-out secret set should be introduced
      before any headline accuracy number is reported as final.

      **Checkpoint naming/adoption discipline going forward**: every
      variant gets an explicit name (`neuraquill-100m-v1`,
      `-v1-teacher`, `-v1-mixed`, ...), and a new checkpoint is only
      *adopted* (becomes the default for the next stage to build on) if
      it demonstrably beats the current best on the Phase 10.5 baseline
      categories -- an evaluation gate, not a vibe. See
      `scripts/evaluate_and_gate.py`.

      **Teacher specialization** (once there's enough per-category data
      to measure it): track each teacher's verified accuracy *per
      category*, not just overall, so the Phase 11 router can eventually
      route by "who's actually best at this category" instead of a
      static capability list.
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
