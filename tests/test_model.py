"""
Phase 2 test suite: the transformer architecture itself.

Run with: pytest tests/test_model.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.attention import CausalSelfAttention  # noqa: E402
from model.config import GPTConfig  # noqa: E402
from model.lm import GPT  # noqa: E402
from model.normalization import LayerNorm  # noqa: E402

torch.manual_seed(0)


def tiny_config(**overrides) -> GPTConfig:
    defaults = dict(
        vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=16, dropout=0.0, bias=True
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


# --------------------------------------------------------------------------
# LayerNorm
# --------------------------------------------------------------------------


def test_layernorm_matches_pytorch_builtin():
    ln = LayerNorm(32, bias=True)
    ref = torch.nn.LayerNorm(32)
    with torch.no_grad():
        ref.weight.copy_(ln.weight)
        ref.bias.copy_(ln.bias)
    x = torch.randn(4, 10, 32)
    out = ln(x)
    ref_out = ref(x)
    assert torch.allclose(out, ref_out, atol=1e-5)


def test_layernorm_output_has_zero_mean_unit_var_before_affine():
    ln = LayerNorm(32, bias=False)
    with torch.no_grad():
        ln.weight.fill_(1.0)
    x = torch.randn(4, 10, 32) * 5 + 3
    out = ln(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(4, 10), atol=1e-5)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(4, 10), atol=1e-3)


def test_layernorm_no_bias_has_no_bias_param():
    ln = LayerNorm(32, bias=False)
    assert ln.bias is None


# --------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------


def test_config_rejects_n_embd_not_divisible_by_n_head():
    with pytest.raises(ValueError):
        GPTConfig(vocab_size=64, n_embd=17, n_head=2)


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


def test_forward_output_shape():
    config = tiny_config()
    model = GPT(config)
    B, T = 3, 12
    idx = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, config.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss():
    config = tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(idx, targets=targets)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_forward_rejects_sequence_longer_than_block_size():
    config = tiny_config(block_size=8)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 9))
    with pytest.raises(ValueError):
        model(idx)


def test_attention_module_output_shape():
    config = tiny_config()
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 10, config.n_embd)
    out = attn(x)
    assert out.shape == x.shape


# --------------------------------------------------------------------------
# causal masking: the single most important correctness property of a
# decoder-only transformer. Position t's output must be identical no matter
# what tokens appear *after* position t.
# --------------------------------------------------------------------------


def test_attention_causal_mask_blocks_future_leakage():
    config = tiny_config()
    attn = CausalSelfAttention(config)
    attn.eval()

    x = torch.randn(1, 10, config.n_embd)
    out1 = attn(x)

    x2 = x.clone()
    x2[:, 5:, :] = torch.randn(1, 5, config.n_embd)  # change everything after position 4
    out2 = attn(x2)

    # positions 0..4 must be unaffected by changes to positions 5..9
    assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-6)
    # sanity: positions 5..9 (which changed) generally *do* differ
    assert not torch.allclose(out1[:, 5:, :], out2[:, 5:, :], atol=1e-6)


def test_full_model_causal_mask_blocks_future_leakage():
    config = tiny_config()
    model = GPT(config)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (1, 10))
    logits1, _ = model(idx)

    idx2 = idx.clone()
    idx2[:, 5:] = torch.randint(0, config.vocab_size, (1, 5))
    logits2, _ = model(idx2)

    assert torch.allclose(logits1[:, :5, :], logits2[:, :5, :], atol=1e-5)


def test_causal_mask_is_lower_triangular():
    config = tiny_config()
    attn = CausalSelfAttention(config)
    mask = attn.causal_mask[0, 0]
    assert torch.equal(mask, torch.tril(torch.ones_like(mask)))


# --------------------------------------------------------------------------
# parameter count -- an independently-derived formula must match the real
# model exactly, to catch architecture bugs (missing/duplicated params).
# --------------------------------------------------------------------------


def _expected_param_count(config: GPTConfig) -> int:
    e, v, b, L = config.n_embd, config.vocab_size, config.block_size, config.n_layer
    wte = v * e
    wpe = b * e
    # per block: 2 LayerNorms (2e each) + attention (qkv 3e^2+3e, proj e^2+e)
    # + mlp (fc 4e^2+4e, proj 4e^2+e)
    per_block = (2 * e) + (3 * e * e + 3 * e + e * e + e) + (2 * e) + (4 * e * e + 4 * e + 4 * e * e + e)
    ln_f = 2 * e
    # lm_head is weight-tied to wte -> no additional params
    return wte + wpe + L * per_block + ln_f


@pytest.mark.parametrize(
    "config",
    [
        tiny_config(),
        tiny_config(n_layer=4, n_head=4, n_embd=32),
        GPTConfig(vocab_size=8192, block_size=512, n_layer=6, n_head=5, n_embd=320),
        GPTConfig(vocab_size=8192, block_size=512, n_layer=13, n_head=12, n_embd=768),
    ],
)
def test_param_count_matches_independent_formula(config):
    model = GPT(config)
    actual = model.num_parameters()
    expected = _expected_param_count(config)
    assert actual == expected


def test_10m_config_is_roughly_10m_params():
    config = GPTConfig(vocab_size=8192, block_size=512, n_layer=6, n_head=5, n_embd=320)
    model = GPT(config)
    n = model.num_parameters()
    assert 8_000_000 <= n <= 13_000_000


def test_100m_config_is_roughly_100m_params():
    config = GPTConfig(vocab_size=8192, block_size=512, n_layer=13, n_head=12, n_embd=768)
    model = GPT(config)
    n = model.num_parameters()
    assert 90_000_000 <= n <= 110_000_000


def test_weight_tying_shares_storage():
    config = tiny_config()
    model = GPT(config)
    assert model.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()


# --------------------------------------------------------------------------
# gradient flow
# --------------------------------------------------------------------------


def test_backward_produces_finite_gradients_for_all_parameters():
    config = tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 10))
    targets = torch.randint(0, config.vocab_size, (2, 10))

    logits, loss = model(idx, targets=targets)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradients"

    total_grad_norm = sum(p.grad.norm().item() for p in model.parameters())
    assert total_grad_norm > 0


def test_gradient_checkpoint_free_backward_is_deterministic_with_seed():
    config = tiny_config()
    torch.manual_seed(123)
    model_a = GPT(config)
    torch.manual_seed(123)
    model_b = GPT(config)

    idx = torch.randint(0, config.vocab_size, (2, 10))
    targets = torch.randint(0, config.vocab_size, (2, 10))

    _, loss_a = model_a(idx, targets=targets)
    _, loss_b = model_b(idx, targets=targets)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


# --------------------------------------------------------------------------
# overfitting sanity check: the ultimate test that the whole architecture +
# loss + backward pass actually work end-to-end. A correctly-wired model
# should be able to memorize one tiny fixed batch almost perfectly.
# --------------------------------------------------------------------------


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    config = tiny_config(n_layer=2, n_head=2, n_embd=32, block_size=16)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    idx = torch.randint(0, config.vocab_size, (4, 8))
    targets = torch.randint(0, config.vocab_size, (4, 8))

    losses = []
    for _ in range(150):
        optimizer.zero_grad()
        _, loss = model(idx, targets=targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # loss should drop sharply -- from near ln(vocab_size) (random-guess
    # loss) down close to zero once the model has memorized this one batch
    random_guess_loss = torch.log(torch.tensor(float(config.vocab_size))).item()
    assert losses[0] > random_guess_loss * 0.5  # starts near random-guess territory
    assert losses[-1] < 0.5                     # ends up having memorized the batch
    assert losses[-1] < losses[0] / 5            # dropped substantially


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def test_generate_produces_valid_token_ids_and_correct_length():
    config = tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 4))
    out = model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=5)
    assert out.shape == (2, 14)
    assert torch.all(out >= 0)
    assert torch.all(out < config.vocab_size)
    # the original context must be an exact prefix of the output
    assert torch.equal(out[:, :4], idx)


def test_generate_respects_block_size_without_erroring():
    config = tiny_config(block_size=16)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    # generate well past block_size -- must internally crop context, not crash
    out = model.generate(idx, max_new_tokens=30)
    assert out.shape == (1, 34)


def test_generate_temperature_near_zero_is_near_greedy():
    """Low temperature should make generation close to deterministic
    (overwhelmingly pick the argmax token)."""
    config = tiny_config()
    torch.manual_seed(1)
    model = GPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 4))

    torch.manual_seed(1)
    out1 = model.generate(idx.clone(), max_new_tokens=5, temperature=1e-4)
    torch.manual_seed(2)
    out2 = model.generate(idx.clone(), max_new_tokens=5, temperature=1e-4)
    assert torch.equal(out1, out2)
