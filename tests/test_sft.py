"""
Phase 4 test suite: chat template, tokenizer/model vocab extension,
masked SFT loss, and the SFT training loop.

Run with: pytest tests/test_sft.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.config import GPTConfig  # noqa: E402
from model.lm import GPT  # noqa: E402
from model.resize import resize_token_embeddings  # noqa: E402
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from training.chat_template import (  # noqa: E402
    CHAT_SPECIAL_TOKENS,
    build_training_example,
    encode_conversation,
)
from training.optim import configure_optimizer  # noqa: E402
from training.pretrain import TrainConfig  # noqa: E402
from training.sft import train_sft  # noqa: E402
from training.sft_dataset import build_sft_dataset, get_sft_batch  # noqa: E402

torch.manual_seed(0)


@pytest.fixture(scope="module")
def chat_tokenizer():
    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning is fun. Paris is the capital of France. "
        "Two plus two equals four. The sky is blue on a clear day. "
    ) * 20
    tok = ByteLevelBPETokenizer.train(
        [corpus], vocab_size=500, special_tokens=("<|endoftext|>", "<|pad|>")
    )
    tok.add_special_tokens(CHAT_SPECIAL_TOKENS)
    return tok


def tiny_config(vocab_size, **overrides) -> GPTConfig:
    defaults = dict(
        vocab_size=vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


# --------------------------------------------------------------------------
# tokenizer.add_special_tokens
# --------------------------------------------------------------------------


def test_add_special_tokens_assigns_new_ids(chat_tokenizer):
    for tok in CHAT_SPECIAL_TOKENS:
        assert tok in chat_tokenizer.vocab
        assert tok in chat_tokenizer.special_tokens


def test_add_special_tokens_is_idempotent():
    tok = ByteLevelBPETokenizer.train(["hello world"], vocab_size=280, special_tokens=("<|endoftext|>",))
    ids1 = tok.add_special_tokens(["<|user|>", "<|assistant|>"])
    vocab_size_after_first = tok.vocab_size
    ids2 = tok.add_special_tokens(["<|user|>", "<|assistant|>"])
    assert ids1 == ids2
    assert tok.vocab_size == vocab_size_after_first  # no duplicate entries added


def test_special_tokens_remain_atomic_after_extension(chat_tokenizer):
    text = "<|user|>Hello there<|assistant|>Hi!<|endoftext|>"
    ids = chat_tokenizer.encode(text)
    assert chat_tokenizer.decode(ids) == text


# --------------------------------------------------------------------------
# chat template: encode_conversation masking
# --------------------------------------------------------------------------


def test_encode_conversation_masks_only_assistant_tokens(chat_tokenizer):
    turns = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital of France."},
    ]
    ids, mask = encode_conversation(chat_tokenizer, turns)
    assert len(ids) == len(mask)
    assert set(mask) <= {0, 1}
    assert any(m == 1 for m in mask)  # some assistant tokens exist
    assert any(m == 0 for m in mask)  # some non-assistant tokens exist

    # reconstruct which text spans got mask==1 and confirm it's exactly the
    # assistant turn (content + its closing EOT), nothing from system/user
    decoded_masked = chat_tokenizer.decode([i for i, m in zip(ids, mask) if m == 1])
    assert decoded_masked == "Paris is the capital of France.<|endoftext|>"


def test_encode_conversation_role_tag_itself_is_never_masked_in(chat_tokenizer):
    turns = [{"role": "assistant", "content": "Hi!"}]
    ids, mask = encode_conversation(chat_tokenizer, turns)
    role_id = chat_tokenizer.vocab["<|assistant|>"]
    assert ids[0] == role_id
    assert mask[0] == 0  # the <|assistant|> tag predicts "who's speaking", not trained as a target


def test_encode_conversation_rejects_unknown_role(chat_tokenizer):
    with pytest.raises(ValueError):
        encode_conversation(chat_tokenizer, [{"role": "narrator", "content": "..."}])


def test_encode_conversation_multi_turn_masks_every_assistant_turn(chat_tokenizer):
    turns = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I am fine."},
    ]
    ids, mask = encode_conversation(chat_tokenizer, turns)
    decoded_masked = chat_tokenizer.decode([i for i, m in zip(ids, mask) if m == 1])
    assert decoded_masked == "Hello!<|endoftext|>I am fine.<|endoftext|>"


# --------------------------------------------------------------------------
# build_training_example
# --------------------------------------------------------------------------


def test_build_training_example_shapes_and_padding(chat_tokenizer):
    ids, mask = encode_conversation(
        chat_tokenizer, [{"role": "assistant", "content": "Hi"}]
    )
    block_size = 32
    pad_id = chat_tokenizer.vocab["<|pad|>"]
    x, y = build_training_example(ids, mask, block_size, pad_id)
    assert len(x) == block_size
    assert len(y) == block_size
    # tail should be padded
    assert x[-1] == pad_id
    # padded target positions must be ignore_index, not the pad token id
    assert y[-1] == -1


def test_build_training_example_ignore_index_matches_mask(chat_tokenizer):
    turns = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    ids, mask = encode_conversation(chat_tokenizer, turns)
    pad_id = chat_tokenizer.vocab["<|pad|>"]
    x, y = build_training_example(ids, mask, block_size=32, pad_id=pad_id)
    # y[j] should be ignore_index (-1) exactly where mask[j+1] == 0
    for j in range(len(ids) - 1):
        if mask[j + 1] == 0:
            assert y[j] == -1
        else:
            assert y[j] == ids[j + 1]


def test_build_training_example_truncates_long_conversation(chat_tokenizer):
    long_content = "the quick brown fox jumps over the lazy dog " * 20
    ids, mask = encode_conversation(chat_tokenizer, [{"role": "assistant", "content": long_content}])
    pad_id = chat_tokenizer.vocab["<|pad|>"]
    x, y = build_training_example(ids, mask, block_size=16, pad_id=pad_id)
    assert len(x) == 16
    assert len(y) == 16


# --------------------------------------------------------------------------
# model vocab resizing
# --------------------------------------------------------------------------


def test_resize_preserves_old_vocab_logits_exactly():
    config = tiny_config(vocab_size=50)
    model = GPT(config)
    model.eval()
    idx = torch.randint(0, 50, (1, 8))
    logits_before, _ = model(idx)

    resize_token_embeddings(model, 53)
    logits_after, _ = model(idx)

    assert model.config.vocab_size == 53
    assert logits_after.shape == (1, 8, 53)
    assert torch.allclose(logits_before, logits_after[:, :, :50], atol=1e-5)


def test_resize_new_rows_are_finite_and_distinct_from_zero_init():
    config = tiny_config(vocab_size=50)
    model = GPT(config)
    resize_token_embeddings(model, 55)
    new_rows = model.wte.weight[50:55]
    assert torch.isfinite(new_rows).all()
    assert new_rows.abs().sum() > 0  # not left as all-zeros


def test_resize_noop_when_same_size():
    config = tiny_config(vocab_size=50)
    model = GPT(config)
    w_before = model.wte.weight.clone()
    resize_token_embeddings(model, 50)
    assert torch.equal(model.wte.weight, w_before)


def test_resize_rejects_shrinking():
    config = tiny_config(vocab_size=50)
    model = GPT(config)
    with pytest.raises(ValueError):
        resize_token_embeddings(model, 40)


def test_resize_keeps_weight_tying():
    config = tiny_config(vocab_size=50)
    model = GPT(config)
    resize_token_embeddings(model, 60)
    assert model.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()


# --------------------------------------------------------------------------
# generation stopping at a stop token (chat inference needs this)
# --------------------------------------------------------------------------


def test_generate_stops_at_stop_token_for_batch_size_one():
    """Make every possible token id a stop token -- whatever the model
    predicts first, generation must halt immediately after it, regardless
    of the model's actual output distribution."""
    config = tiny_config(vocab_size=20, block_size=32)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    torch.manual_seed(0)
    out = model.generate(idx, max_new_tokens=50, temperature=1.0, stop_token_ids=set(range(config.vocab_size)))
    assert out.shape == (1, 5)  # 4 context tokens + exactly 1 generated


def test_generate_without_stop_tokens_runs_full_length():
    config = tiny_config(vocab_size=20, block_size=32)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 14)


# --------------------------------------------------------------------------
# end-to-end SFT: masked loss must actually teach assistant responses
# --------------------------------------------------------------------------


def test_sft_overfits_tiny_instruction_set(chat_tokenizer):
    """Train on a small fixed set of (prompt -> fixed assistant answer)
    conversations and confirm the masked SFT loss drops sharply -- proof
    the whole encode -> mask -> train pipeline is wired correctly."""
    torch.manual_seed(0)
    conversations = [
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris is the capital of France."},
        ],
        [
            {"role": "user", "content": "What is two plus two?"},
            {"role": "assistant", "content": "Two plus two equals four."},
        ],
        [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "The sky is blue on a clear day."},
        ],
    ]
    block_size = 32
    X, Y = build_sft_dataset(chat_tokenizer, conversations, block_size)
    # tiny dataset -- use the same examples for train and val, since the
    # point here is purely to verify the training mechanics overfit correctly
    X_train, Y_train = X, Y
    X_val, Y_val = X, Y

    config = tiny_config(vocab_size=chat_tokenizer.vocab_size, block_size=block_size, n_embd=48, n_layer=3, n_head=4)
    model = GPT(config)

    train_cfg = TrainConfig(
        max_steps=200, warmup_steps=20, batch_size=3, block_size=block_size,
        max_lr=5e-3, min_lr=5e-4, eval_interval=50, eval_batches=3,
        checkpoint_path=None, device="cpu",
    )
    history = train_sft(model, config, X_train, Y_train, X_val, Y_val, train_cfg, log_fn=lambda s: None)

    assert history.val_losses[0] > history.val_losses[-1]
    assert history.val_losses[-1] < 0.5  # near-memorized on this tiny fixed set


def test_get_sft_batch_shapes():
    X = np.random.randint(0, 100, size=(10, 16)).astype(np.int64)
    Y = np.random.randint(-1, 100, size=(10, 16)).astype(np.int64)
    x, y = get_sft_batch(X, Y, batch_size=4, device="cpu")
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
