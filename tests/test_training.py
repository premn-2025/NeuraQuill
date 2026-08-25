"""
Phase 3 test suite: the pretraining loop (dataloader, LR schedule,
optimizer, checkpointing, end-to-end training, perplexity).

Run with: pytest tests/test_training.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.perplexity import estimate_loss, perplexity_from_loss  # noqa: E402
from model.config import GPTConfig  # noqa: E402
from model.lm import GPT  # noqa: E402
from training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from training.dataset import get_batch, split_train_val, tokenize_and_save  # noqa: E402
from training.lr_schedule import get_lr  # noqa: E402
from training.optim import configure_optimizer  # noqa: E402
from training.pretrain import TrainConfig, train  # noqa: E402

torch.manual_seed(0)


def tiny_config(**overrides) -> GPTConfig:
    defaults = dict(
        vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0, bias=True
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


# --------------------------------------------------------------------------
# dataset / batching
# --------------------------------------------------------------------------


def test_get_batch_shapes_and_dtype():
    data = np.arange(1000, dtype=np.uint16) % 64
    x, y = get_batch(data, block_size=16, batch_size=8, device="cpu")
    assert x.shape == (8, 16)
    assert y.shape == (8, 16)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_get_batch_targets_are_inputs_shifted_by_one():
    """y[i] must equal x shifted right by one token -- the fundamental
    next-token-prediction training pair."""
    data = np.arange(200, dtype=np.uint16)
    np.random.seed(0)
    x, y = get_batch(data, block_size=10, batch_size=4, device="cpu")
    for row in range(4):
        # since data is a strictly increasing arange, x[row]+1 == y[row] elementwise
        assert torch.equal(x[row] + 1, y[row])


def test_get_batch_raises_when_data_too_short():
    data = np.arange(5, dtype=np.uint16)
    with pytest.raises(ValueError):
        get_batch(data, block_size=16, batch_size=4, device="cpu")


def test_tokenize_and_save_roundtrips_through_get_batch(tmp_path):
    from tokenizer.bpe import ByteLevelBPETokenizer

    corpus = "the quick brown fox jumps over the lazy dog " * 50
    tok = ByteLevelBPETokenizer.train([corpus], vocab_size=300)
    out_path = tmp_path / "tokens.bin"
    n = tokenize_and_save(tok, corpus, str(out_path))
    assert n > 0
    assert out_path.stat().st_size == n * 2  # uint16 = 2 bytes/token

    from training.dataset import load_token_memmap

    data = load_token_memmap(str(out_path))
    assert len(data) == n
    x, y = get_batch(data, block_size=8, batch_size=4, device="cpu")
    assert x.shape == (4, 8)


def test_split_train_val_sizes_and_no_overlap_content():
    data = np.arange(1000, dtype=np.uint16)
    train_data, val_data = split_train_val(data, val_fraction=0.1)
    assert len(train_data) == 900
    assert len(val_data) == 100
    assert train_data[-1] == 899
    assert val_data[0] == 900


# --------------------------------------------------------------------------
# LR schedule
# --------------------------------------------------------------------------


def test_lr_schedule_warmup_is_linear_and_increasing():
    lrs = [get_lr(s, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4) for s in range(100)]
    assert lrs == sorted(lrs)  # strictly non-decreasing through warmup
    assert lrs[0] < lrs[-1]
    # with the (step + 1) / warmup_steps convention, the last warmup step
    # (index 99, i.e. the 100th step) already reaches max_lr exactly
    assert lrs[-1] == pytest.approx(1e-3, rel=1e-6)


def test_lr_schedule_peaks_at_end_of_warmup():
    lr_at_warmup_end = get_lr(100, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4)
    assert lr_at_warmup_end == pytest.approx(1e-3, rel=1e-6)


def test_lr_schedule_decays_after_warmup():
    lrs = [get_lr(s, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4) for s in range(100, 1000)]
    assert lrs == sorted(lrs, reverse=True)  # non-increasing (cosine decay)


def test_lr_schedule_reaches_min_lr_at_max_steps():
    lr_end = get_lr(1000, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4)
    assert lr_end == pytest.approx(1e-4, rel=1e-6)
    # and stays at min_lr beyond max_steps
    lr_past_end = get_lr(2000, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4)
    assert lr_past_end == pytest.approx(1e-4, rel=1e-6)


def test_lr_schedule_no_warmup():
    lr0 = get_lr(0, warmup_steps=0, max_steps=100, max_lr=1e-3, min_lr=0.0)
    assert lr0 == pytest.approx(1e-3, rel=1e-6)


# --------------------------------------------------------------------------
# optimizer param groups
# --------------------------------------------------------------------------


def test_optimizer_splits_decay_and_no_decay_params():
    config = tiny_config()
    model = GPT(config)
    opt = configure_optimizer(model, weight_decay=0.1, lr=1e-3)
    assert len(opt.param_groups) == 2
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    # every 1D param (biases, LayerNorm weight/bias) must be in the no-decay group
    for p in no_decay_group["params"]:
        assert p.dim() == 1
    for p in decay_group["params"]:
        assert p.dim() >= 2


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------


def test_checkpoint_save_and_load_restores_identical_weights(tmp_path):
    config = tiny_config()
    model = GPT(config)
    optimizer = configure_optimizer(model, 0.1, 1e-3)

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(str(ckpt_path), model, optimizer, step=42, best_val_loss=1.23, config=config)

    loaded_model, loaded_opt, step, best_val_loss, loaded_config = load_checkpoint(
        str(ckpt_path), device="cpu", optimizer=configure_optimizer(GPT(config), 0.1, 1e-3)
    )

    assert step == 42
    assert best_val_loss == pytest.approx(1.23)
    assert loaded_config == config

    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_checkpoint_resume_continues_training_without_loss_spike(tmp_path):
    """The classic checkpointing bug: resuming re-initializes something and
    the loss jumps back up. Train a few steps, save, resume into a fresh
    model/optimizer, and confirm the resumed loss is consistent with (not
    wildly worse than) where training left off."""
    torch.manual_seed(0)
    config = tiny_config(n_embd=32, n_layer=2, n_head=2, block_size=16)
    model = GPT(config)
    data = np.random.randint(0, config.vocab_size, size=5000).astype(np.uint16)
    train_data, val_data = split_train_val(data, val_fraction=0.1)

    ckpt_path = tmp_path / "resume_ckpt.pt"
    train_cfg = TrainConfig(
        max_steps=30, warmup_steps=5, batch_size=8, block_size=config.block_size,
        max_lr=1e-3, min_lr=1e-4, eval_interval=10, checkpoint_path=str(ckpt_path), device="cpu",
    )
    history1 = train(model, config, train_data, val_data, train_cfg, log_fn=lambda s: None)

    loaded_model, loaded_opt, resumed_step, best_val, _ = load_checkpoint(
        str(ckpt_path), device="cpu", optimizer=configure_optimizer(GPT(config), 0.1, 1e-3)
    )

    # continue training the resumed model for a few more steps
    train_cfg2 = TrainConfig(
        max_steps=resumed_step + 20, warmup_steps=5, batch_size=8, block_size=config.block_size,
        max_lr=1e-3, min_lr=1e-4, eval_interval=10, checkpoint_path=None, device="cpu",
    )
    history2 = train(
        loaded_model, config, train_data, val_data, train_cfg2,
        log_fn=lambda s: None, start_step=resumed_step, optimizer=loaded_opt,
    )

    # loss right after resuming should be in the same ballpark as the
    # checkpointed best_val_loss, not a random-init-level loss
    resumed_first_loss = history2.train_losses[0]
    random_guess_loss = np.log(config.vocab_size)
    assert resumed_first_loss < random_guess_loss  # better than random guessing
    assert resumed_first_loss < best_val * 3        # not a catastrophic regression


# --------------------------------------------------------------------------
# end-to-end training: loss must actually go down
# --------------------------------------------------------------------------


def test_training_loss_decreases_on_learnable_synthetic_pattern():
    """A synthetic but genuinely learnable sequence (periodic token pattern)
    -- if the training loop is wired correctly, loss should drop well
    below the random-guess baseline of ln(vocab_size)."""
    torch.manual_seed(0)
    vocab_size = 32
    period = 6
    n_tokens = 20000
    pattern = np.tile(np.arange(period, dtype=np.uint16), n_tokens // period + 1)[:n_tokens]

    train_data, val_data = split_train_val(pattern, val_fraction=0.05)

    config = tiny_config(vocab_size=vocab_size, n_embd=32, n_layer=2, n_head=2, block_size=16)
    model = GPT(config)

    train_cfg = TrainConfig(
        max_steps=300, warmup_steps=20, batch_size=16, block_size=16,
        max_lr=5e-3, min_lr=5e-4, eval_interval=50, eval_batches=5,
        checkpoint_path=None, device="cpu",
    )
    history = train(model, config, train_data, val_data, train_cfg, log_fn=lambda s: None)

    random_guess_loss = np.log(vocab_size)
    first_val = history.val_losses[0]
    last_val = history.val_losses[-1]

    assert first_val > 1.0  # starts out uncertain
    assert last_val < 0.3   # learns the near-deterministic periodic pattern
    assert last_val < first_val / 3


def test_estimate_loss_and_perplexity_consistency():
    config = tiny_config()
    model = GPT(config)
    data = np.random.randint(0, config.vocab_size, size=2000).astype(np.uint16)

    loss = estimate_loss(model, data, block_size=16, batch_size=8, num_batches=5, device="cpu")
    ppl = perplexity_from_loss(loss)

    assert loss > 0
    assert ppl == pytest.approx(np.exp(loss), rel=1e-6)
    # perplexity of an untrained model on random data should be roughly
    # in the neighborhood of vocab_size (i.e. "as confused as random guessing")
    assert ppl < config.vocab_size * 3


def test_estimate_loss_does_not_leave_model_in_eval_mode():
    """estimate_loss briefly switches to eval() for dropout/etc, but must
    restore train() so the training loop's next step behaves correctly."""
    config = tiny_config(dropout=0.1)
    model = GPT(config)
    model.train()
    data = np.random.randint(0, config.vocab_size, size=500).astype(np.uint16)
    estimate_loss(model, data, block_size=16, batch_size=4, num_batches=2, device="cpu")
    assert model.training is True
