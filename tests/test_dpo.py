"""
Phase 6 test suite: preference-pair generation (verifiers/preference.py)
and DPO training (training/dpo.py, dpo_dataset.py).

Run with: pytest tests/test_dpo.py -v
"""
import copy
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.config import GPTConfig  # noqa: E402
from model.lm import GPT  # noqa: E402
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from training.chat_template import CHAT_SPECIAL_TOKENS  # noqa: E402
from training.dpo import DPOTrainConfig, dpo_loss, sequence_logprob, train_dpo  # noqa: E402
from training.dpo_dataset import build_dpo_dataset, get_dpo_batch  # noqa: E402
from verifiers.generators import generate_dataset  # noqa: E402
from verifiers.preference import generate_preference_dataset, make_preference_pair  # noqa: E402
from verifiers.reward import grade  # noqa: E402

torch.manual_seed(0)


def tiny_config(**overrides) -> GPTConfig:
    defaults = dict(
        vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=16, dropout=0.0, bias=True
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


@pytest.fixture(scope="module")
def chat_tokenizer():
    corpus = (
        "Hello there, how are you today? I am doing well, thank you for asking. "
        "The weather is nice and sunny. What is two plus two? It equals four. "
    ) * 20
    tok = ByteLevelBPETokenizer.train(
        [corpus], vocab_size=500, special_tokens=("<|endoftext|>", "<|pad|>")
    )
    tok.add_special_tokens(CHAT_SPECIAL_TOKENS)
    return tok


# --------------------------------------------------------------------------
# preference pair generation (built on Phase 5's verifier)
# --------------------------------------------------------------------------


def test_make_preference_pair_chosen_verifies_correct_rejected_does_not():
    import random

    rng = random.Random(0)
    problems = generate_dataset(100, seed=0)
    for problem in problems:
        pair = make_preference_pair(problem, rng)
        assert grade(problem, pair.chosen).correct
        assert not grade(problem, pair.rejected).correct


def test_generate_preference_dataset_self_consistency():
    """Every emitted pair, re-graded independently against a freshly
    regenerated problem set of the same seed, must have chosen=correct,
    rejected=incorrect -- the same discipline as Phase 5's dataset check."""
    pairs = generate_preference_dataset(300, seed=1)
    assert len(pairs) > 0
    problems_by_question = {p.question: p for p in generate_dataset(300, seed=1)}
    for pair in pairs:
        problem = problems_by_question[pair.question]
        assert grade(problem, pair.chosen).correct
        assert not grade(problem, pair.rejected).correct


def test_generate_preference_dataset_covers_all_kinds():
    pairs = generate_preference_dataset(300, seed=2)
    kinds = {p.kind for p in pairs}
    assert kinds == {"arithmetic", "linear_equation", "comparison"}


def test_chosen_and_rejected_differ():
    pairs = generate_preference_dataset(50, seed=3)
    for pair in pairs:
        assert pair.chosen != pair.rejected


# --------------------------------------------------------------------------
# sequence_logprob
# --------------------------------------------------------------------------


def test_sequence_logprob_matches_manual_computation_when_fully_unmasked():
    config = tiny_config()
    model = GPT(config)
    model.eval()
    x = torch.randint(0, config.vocab_size, (3, config.block_size))
    y = torch.randint(0, config.vocab_size, (3, config.block_size))  # no -1s: every position counts

    with torch.no_grad():
        logits, _ = model(x)
        log_probs = F.log_softmax(logits, dim=-1)
        manual = torch.gather(log_probs, 2, y.unsqueeze(-1)).squeeze(-1).sum(dim=1)

    result = sequence_logprob(model, x, y)
    assert torch.allclose(result, manual, atol=1e-4)


def test_sequence_logprob_excludes_masked_positions():
    config = tiny_config()
    model = GPT(config)
    model.eval()
    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    y = torch.randint(0, config.vocab_size, (2, config.block_size))
    half = config.block_size // 2
    y_masked = y.clone()
    y_masked[:, half:] = -1

    with torch.no_grad():
        logits, _ = model(x)
        log_probs = F.log_softmax(logits, dim=-1)
        manual_first_half = (
            torch.gather(log_probs[:, :half], 2, y[:, :half].unsqueeze(-1)).squeeze(-1).sum(dim=1)
        )

    result = sequence_logprob(model, x, y_masked)
    assert torch.allclose(result, manual_first_half, atol=1e-4)


def test_sequence_logprob_all_masked_is_zero():
    config = tiny_config()
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    y = torch.full((2, config.block_size), -1, dtype=torch.long)
    result = sequence_logprob(model, x, y)
    assert torch.allclose(result, torch.zeros(2), atol=1e-6)


# --------------------------------------------------------------------------
# dpo_loss
# --------------------------------------------------------------------------


def test_dpo_loss_equals_ln2_when_policy_equals_reference():
    """The defining invariant of DPO: if the policy hasn't diverged from
    the reference at all, policy_logratios == ref_logratios exactly, so
    beta * 0 = 0 and loss = -log(sigmoid(0)) = ln(2), regardless of the
    actual data."""
    config = tiny_config()
    policy = GPT(config)
    ref = copy.deepcopy(policy)
    x_c = torch.randint(0, config.vocab_size, (4, config.block_size))
    y_c = torch.randint(0, config.vocab_size, (4, config.block_size))
    x_r = torch.randint(0, config.vocab_size, (4, config.block_size))
    y_r = torch.randint(0, config.vocab_size, (4, config.block_size))

    loss, metrics = dpo_loss(policy, ref, x_c, y_c, x_r, y_r, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-4)


def test_dpo_loss_metrics_keys():
    config = tiny_config()
    policy = GPT(config)
    ref = copy.deepcopy(policy)
    x_c = torch.randint(0, config.vocab_size, (2, config.block_size))
    y_c = torch.randint(0, config.vocab_size, (2, config.block_size))
    x_r = torch.randint(0, config.vocab_size, (2, config.block_size))
    y_r = torch.randint(0, config.vocab_size, (2, config.block_size))
    _, metrics = dpo_loss(policy, ref, x_c, y_c, x_r, y_r)
    assert set(metrics.keys()) == {"accuracy", "margin"}


def test_dpo_loss_rewards_higher_chosen_logprob():
    """Hand-construct a policy that's diverged from ref such that chosen is
    strictly more likely than rejected (relative to ref) and confirm the
    loss is below ln(2) and accuracy is 1.0."""
    config = tiny_config()
    policy = GPT(config)
    ref = copy.deepcopy(policy)

    x_c = torch.randint(0, config.vocab_size, (4, config.block_size))
    y_c = torch.randint(0, config.vocab_size, (4, config.block_size))
    x_r = torch.randint(0, config.vocab_size, (4, config.block_size))
    y_r = torch.randint(0, config.vocab_size, (4, config.block_size))

    # take a few gradient steps directly maximizing chosen logprob and
    # minimizing rejected logprob, bypassing dpo_loss, to force a policy
    # that's diverged favorably -- then check dpo_loss reflects that
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    for _ in range(20):
        optimizer.zero_grad()
        toy_loss = -sequence_logprob(policy, x_c, y_c).mean() + sequence_logprob(policy, x_r, y_r).mean()
        toy_loss.backward()
        optimizer.step()

    loss, metrics = dpo_loss(policy, ref, x_c, y_c, x_r, y_r, beta=0.1)
    assert loss.item() < math.log(2)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["margin"] > 0


# --------------------------------------------------------------------------
# dpo_dataset
# --------------------------------------------------------------------------


def test_build_dpo_dataset_shapes(chat_tokenizer):
    pairs = generate_preference_dataset(10, seed=5, kinds=["arithmetic"])
    # swap in text the tiny tokenizer can actually represent reasonably --
    # shape/plumbing is what's under test here, not text fidelity
    block_size = 32
    Xc, Yc, Xr, Yr = build_dpo_dataset(chat_tokenizer, pairs, block_size)
    n = len(pairs)
    assert Xc.shape == (n, block_size)
    assert Yc.shape == (n, block_size)
    assert Xr.shape == (n, block_size)
    assert Yr.shape == (n, block_size)


def test_get_dpo_batch_shapes():
    import numpy as np

    n, block_size = 20, 16
    Xc = np.random.randint(0, 100, size=(n, block_size)).astype(np.int64)
    Yc = np.random.randint(-1, 100, size=(n, block_size)).astype(np.int64)
    Xr = np.random.randint(0, 100, size=(n, block_size)).astype(np.int64)
    Yr = np.random.randint(-1, 100, size=(n, block_size)).astype(np.int64)
    x_c, y_c, x_r, y_r = get_dpo_batch(Xc, Yc, Xr, Yr, batch_size=6, device="cpu")
    assert x_c.shape == (6, block_size)
    assert y_r.shape == (6, block_size)


# --------------------------------------------------------------------------
# end-to-end DPO training: the policy must learn to prefer chosen over
# rejected relative to the (frozen) reference
# --------------------------------------------------------------------------


def test_train_dpo_improves_accuracy_and_margin(chat_tokenizer):
    torch.manual_seed(0)
    conversations = [
        ("Hello there, how are you today?", "I am doing well, thank you for asking.", "The weather is nice."),
        ("What is two plus two?", "It equals four.", "Hello there, how are you today?"),
        ("How are you today?", "I am doing well, thank you for asking.", "It equals four."),
    ]

    class Pair:
        def __init__(self, question, chosen, rejected):
            self.question, self.chosen, self.rejected = question, chosen, rejected

    pairs = [Pair(q, c, r) for q, c, r in conversations]
    # block_size must comfortably fit system prompt + user + assistant turns
    # -- too small and truncation cuts off before any assistant tokens
    # survive, leaving an all-masked target (sequence_logprob always 0,
    # dpo_loss permanently stuck at ln(2) regardless of training)
    block_size = 128
    Xc, Yc, Xr, Yr = build_dpo_dataset(chat_tokenizer, pairs, block_size)

    config = tiny_config(vocab_size=chat_tokenizer.vocab_size, block_size=block_size, n_embd=48, n_layer=3, n_head=4)
    policy = GPT(config)

    train_cfg = DPOTrainConfig(
        max_steps=150, warmup_steps=10, batch_size=3, max_lr=1e-3, min_lr=1e-4,
        beta=0.1, eval_interval=30, eval_batches=1, checkpoint_path=None, device="cpu",
    )
    history = train_dpo(policy, config, Xc, Yc, Xr, Yr, Xc, Yc, Xr, Yr, train_cfg, log_fn=lambda s: None)

    assert history.val_losses[0] == pytest.approx(math.log(2), abs=0.05)  # starts at the policy==ref baseline
    assert history.val_losses[-1] < history.val_losses[0]
    # accuracy is bounded at 1.0 and can saturate immediately on a tiny eval
    # set -- non-decreasing is the correct assertion, not strictly-increasing
    assert history.val_accuracies[-1] >= history.val_accuracies[0]
    assert history.val_accuracies[-1] == pytest.approx(1.0)
    # margin (unbounded) is the more informative signal that training kept
    # pushing chosen further ahead of rejected even after accuracy maxed out
    assert history.val_margins[-1] > history.val_margins[0]
