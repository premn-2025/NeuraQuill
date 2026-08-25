"""
Phase 1 test suite: byte-level BPE tokenizer.

Run with: pytest tests/test_tokenizer.py -v
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer.bpe import BYTE_TO_UNICODE, UNICODE_TO_BYTE, ByteLevelBPETokenizer  # noqa: E402

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_corpus.txt"


@pytest.fixture(scope="module")
def corpus_text():
    return CORPUS_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def small_tok(corpus_text):
    """A small tokenizer, fast to train, reused across most tests."""
    return ByteLevelBPETokenizer.train([corpus_text], vocab_size=600)


# --------------------------------------------------------------------------
# byte<->unicode mapping
# --------------------------------------------------------------------------


def test_byte_unicode_mapping_is_bijective():
    assert len(BYTE_TO_UNICODE) == 256
    assert len(set(BYTE_TO_UNICODE.values())) == 256  # no collisions
    for b, ch in BYTE_TO_UNICODE.items():
        assert UNICODE_TO_BYTE[ch] == b


def test_byte_unicode_roundtrip_all_bytes():
    for b in range(256):
        ch = BYTE_TO_UNICODE[b]
        assert UNICODE_TO_BYTE[ch] == b


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def test_train_respects_vocab_size_budget(corpus_text):
    tok = ByteLevelBPETokenizer.train([corpus_text], vocab_size=500)
    assert tok.vocab_size <= 500
    # base bytes + special tokens must always be present
    assert tok.vocab_size >= 256 + len(tok.special_tokens)


def test_train_vocab_size_too_small_raises(corpus_text):
    with pytest.raises(ValueError):
        ByteLevelBPETokenizer.train([corpus_text], vocab_size=10)


def test_train_produces_more_merges_with_bigger_budget(corpus_text):
    small = ByteLevelBPETokenizer.train([corpus_text], vocab_size=400)
    big = ByteLevelBPETokenizer.train([corpus_text], vocab_size=900)
    assert len(big.merges) > len(small.merges)


def test_training_is_deterministic(corpus_text):
    tok1 = ByteLevelBPETokenizer.train([corpus_text], vocab_size=600)
    tok2 = ByteLevelBPETokenizer.train([corpus_text], vocab_size=600)
    assert tok1.vocab == tok2.vocab
    assert tok1.merges == tok2.merges


def test_common_word_gets_merged_into_fewer_tokens(corpus_text):
    """'the' appears very often in the corpus -> should end up as few tokens
    (ideally a single merged token), while a rare/unseen word stays closer
    to one-token-per-byte."""
    tok = ByteLevelBPETokenizer.train([corpus_text], vocab_size=800)
    common = tok.encode(" the")
    assert len(common) <= 2  # heavily merged vs. its 4 raw bytes


# --------------------------------------------------------------------------
# round-trip correctness (the most important property)
# --------------------------------------------------------------------------


ROUNDTRIP_CASES = [
    "",
    " ",
    "a",
    "The quick brown fox jumps over the lazy dog.",
    "def foo(x, y):\n    return x + y\n",
    "SELECT * FROM users WHERE id = 1;",
    "Wait, what?! Really... Yes, absolutely!",
    "He said, \"I'll be there by 5 PM,\" and then he left.",
    "    leading and trailing whitespace   ",
    "line1\nline2\n\nline4",
    "tabs\tand\tmore\ttabs",
    "1234567890 -3.14159 1e10",
    "こんにちは世界",  # Japanese, unseen in training corpus except partially
    "你好，世界。",
    "Bonjour, ça va? Très bien, merci!",
    "I love debugging 🐛 and shipping 🚀 features.",
    "​‌zero-width chars",
    "emoji sequence: 👨‍👩‍👧‍👦 family",
    "a" * 500,  # long repeated run
    "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
]


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_roundtrip_exact(small_tok, text):
    ids = small_tok.encode(text)
    decoded = small_tok.decode(ids)
    assert decoded == text


def test_roundtrip_fuzz_random_unicode(small_tok):
    rng = random.Random(42)
    for _ in range(200):
        length = rng.randint(0, 40)
        chars = [chr(rng.randint(0x20, 0x2FFFF)) for _ in range(length)]
        # skip lone surrogates, which are not valid utf-8 on their own
        chars = [c for c in chars if not (0xD800 <= ord(c) <= 0xDFFF)]
        text = "".join(chars)
        ids = small_tok.encode(text)
        assert small_tok.decode(ids) == text


def test_encode_never_raises_on_arbitrary_bytes_as_text(small_tok):
    """Every possible unicode string must be encodable -- that's the whole
    point of operating at the byte level."""
    rng = random.Random(7)
    for _ in range(50):
        raw = bytes(rng.randint(0, 255) for _ in range(20))
        text = raw.decode("utf-8", errors="replace")
        ids = small_tok.encode(text)  # must not raise
        assert small_tok.decode(ids) == text


def test_empty_string_encodes_to_empty_list(small_tok):
    assert small_tok.encode("") == []
    assert small_tok.decode([]) == ""


# --------------------------------------------------------------------------
# special tokens
# --------------------------------------------------------------------------


def test_special_token_is_atomic(corpus_text):
    tok = ByteLevelBPETokenizer.train(
        [corpus_text], vocab_size=600, special_tokens=("<|endoftext|>", "<|pad|>")
    )
    text = "Hello<|endoftext|>World"
    ids = tok.encode(text)
    eot_id = tok.vocab["<|endoftext|>"]
    assert eot_id in ids
    assert tok.decode(ids) == text


def test_special_token_survives_roundtrip_alone(corpus_text):
    tok = ByteLevelBPETokenizer.train(
        [corpus_text], vocab_size=600, special_tokens=("<|endoftext|>",)
    )
    ids = tok.encode("<|endoftext|>")
    assert ids == [tok.vocab["<|endoftext|>"]]
    assert tok.decode(ids) == "<|endoftext|>"


def test_multiple_special_tokens_in_sequence(corpus_text):
    tok = ByteLevelBPETokenizer.train(
        [corpus_text],
        vocab_size=600,
        special_tokens=("<|endoftext|>", "<|pad|>", "<|user|>", "<|assistant|>"),
    )
    text = "<|user|>Hi there<|assistant|>Hello!<|endoftext|>"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_decode_unknown_id_raises(small_tok):
    bad_id = small_tok.vocab_size + 9999
    with pytest.raises(KeyError):
        small_tok.decode([bad_id])


# --------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------


def test_save_load_preserves_vocab_and_merges(small_tok, tmp_path):
    save_dir = tmp_path / "tok_artifacts"
    small_tok.save(str(save_dir))
    loaded = ByteLevelBPETokenizer.load(str(save_dir))

    assert loaded.vocab == small_tok.vocab
    assert loaded.merges == small_tok.merges
    assert loaded.special_tokens == small_tok.special_tokens


def test_save_load_preserves_encode_behavior(small_tok, tmp_path):
    save_dir = tmp_path / "tok_artifacts2"
    small_tok.save(str(save_dir))
    loaded = ByteLevelBPETokenizer.load(str(save_dir))

    for text in ROUNDTRIP_CASES:
        assert loaded.encode(text) == small_tok.encode(text)
        assert loaded.decode(loaded.encode(text)) == text


def test_encode_before_training_raises():
    tok = ByteLevelBPETokenizer()
    with pytest.raises(RuntimeError):
        tok.encode("hello")
