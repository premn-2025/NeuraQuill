"""
Byte-level BPE tokenizer, implemented from scratch (no HuggingFace
`tokenizers`, no `sentencepiece`, no `tiktoken`).

Approach (same family as GPT-2's tokenizer):
  1. Every input string is first viewed as a sequence of UTF-8 bytes, so
     there is never an "unknown token" / OOV problem — literally any
     string round-trips exactly, in any language, including emoji.
  2. Bytes (0-255) are remapped to printable unicode characters purely so
     that BPE (which operates on strings) has printable symbols to work
     with internally.
  3. Text is first split into "pre-tokens" (word-ish chunks) with a
     regex, then BPE merge rules are learned/applied within each
     pre-token. Pre-tokenization keeps merges from ever crossing a
     word/whitespace boundary, which is what GPT-2/GPT-4-style
     tokenizers do.
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --- byte <-> printable unicode mapping (GPT-2 style) -----------------------


def _byte_to_unicode() -> Dict[int, str]:
    """Reversible mapping from every byte value (0-255) to a printable
    unicode char, so BPE can operate on strings instead of raw bytes."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_TO_UNICODE: Dict[int, str] = _byte_to_unicode()
UNICODE_TO_BYTE: Dict[str, int] = {v: k for k, v in BYTE_TO_UNICODE.items()}

# Pre-tokenizer pattern: contractions, then runs of word chars, runs of
# non-word/non-space chars, then whitespace. Close cousin of GPT-2's regex,
# using Python's built-in unicode-aware \w / \s instead of the external
# `regex` package's \p{L}/\p{N} (keeps this dependency-free).
_PRETOKEN_RE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _pretokenize(text: str) -> List[str]:
    return _PRETOKEN_RE.findall(text)


def _to_byte_symbols(pretoken: str) -> Tuple[str, ...]:
    """Map a pre-token's raw UTF-8 bytes to the printable symbol alphabet."""
    raw = pretoken.encode("utf-8")
    return tuple(BYTE_TO_UNICODE[b] for b in raw)


def _get_pair_counts(word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
    counts: Counter = Counter()
    for word, freq in word_freqs.items():
        for a, b in zip(word, word[1:]):
            counts[(a, b)] += freq
    return counts


def _merge_word(word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    a, b = pair
    merged = a + b
    out: List[str] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class ByteLevelBPETokenizer:
    """A from-scratch byte-level BPE tokenizer.

    Because it operates on raw UTF-8 bytes, `decode(encode(text)) == text`
    for *any* string — there is no unknown-token failure mode.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Sequence[str] = (),
    ):
        self.special_tokens: List[str] = list(special_tokens)
        self.vocab: Dict[str, int] = vocab or {}
        self.merges: List[Tuple[str, str]] = merges or []
        self._merge_rank: Dict[Tuple[str, str], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self._id_to_token: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        self._special_re = None
        if self.special_tokens:
            alt = "|".join(re.escape(t) for t in self.special_tokens)
            self._special_re = re.compile(f"({alt})")

    def add_special_tokens(self, tokens: Sequence[str]) -> List[int]:
        """Append new atomic special tokens to an already-trained tokenizer
        (e.g. adding chat-template tokens like <|user|>/<|assistant|> to a
        tokenizer that was only trained for plain-text pretraining).
        Returns the ids assigned to `tokens` (existing ones keep their id).
        The corresponding model's embedding table must be resized to match
        the new vocab_size -- see model/resize.py."""
        assigned = []
        for tok in tokens:
            if tok in self.vocab:
                assigned.append(self.vocab[tok])
                continue
            new_id = len(self.vocab)
            self.vocab[tok] = new_id
            self._id_to_token[new_id] = tok
            self.special_tokens.append(tok)
            assigned.append(new_id)
        alt = "|".join(re.escape(t) for t in self.special_tokens)
        self._special_re = re.compile(f"({alt})")
        return assigned

    # -- training -------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        vocab_size: int,
        special_tokens: Sequence[str] = ("<|endoftext|>", "<|pad|>"),
        min_pair_freq: int = 1,
        verbose: bool = False,
    ) -> "ByteLevelBPETokenizer":
        if vocab_size < 256 + len(special_tokens):
            raise ValueError(
                f"vocab_size must be >= 256 base bytes + "
                f"{len(special_tokens)} special tokens"
            )

        # 1. count pre-token frequencies across the corpus, as byte-symbol tuples
        word_freqs: Counter = Counter()
        for text in texts:
            for pretoken in _pretokenize(text):
                word_freqs[_to_byte_symbols(pretoken)] += 1

        # 2. base vocab = the 256 byte symbols
        base_symbols = [BYTE_TO_UNICODE[b] for b in range(256)]
        vocab: Dict[str, int] = {s: i for i, s in enumerate(base_symbols)}

        merges: List[Tuple[str, str]] = []
        n_target_merges = vocab_size - len(base_symbols) - len(special_tokens)

        words: Dict[Tuple[str, ...], int] = dict(word_freqs)
        for step in range(n_target_merges):
            pair_counts = _get_pair_counts(words)
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_pair_freq:
                break
            new_words: Dict[Tuple[str, ...], int] = Counter()
            for word, freq in words.items():
                new_words[_merge_word(word, best_pair)] += freq
            words = new_words
            merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = len(vocab)
            if verbose and step % 100 == 0:
                print(f"merge {step}: {best_pair} -> {new_token!r} (freq={best_count})")

        for tok in special_tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)

        return cls(vocab=vocab, merges=merges, special_tokens=list(special_tokens))

    # -- encode / decode --------------------------------------------------------

    def _bpe_encode_pretoken(self, pretoken: str) -> List[str]:
        symbols = list(_to_byte_symbols(pretoken))
        if len(symbols) <= 1:
            return symbols
        while True:
            pairs = list(zip(symbols, symbols[1:]))
            ranked = [
                (self._merge_rank[p], p) for p in pairs if p in self._merge_rank
            ]
            if not ranked:
                break
            _, best_pair = min(ranked, key=lambda x: x[0])
            symbols = list(_merge_word(tuple(symbols), best_pair))
        return symbols

    def encode(self, text: str) -> List[int]:
        if not self.vocab:
            raise RuntimeError("Tokenizer has no vocab — train() or load() it first.")
        if text == "":
            return []

        if self._special_re is not None:
            segments = [s for s in self._special_re.split(text) if s != ""]
        else:
            segments = [text]

        ids: List[int] = []
        for seg in segments:
            if seg in self.special_tokens:
                ids.append(self.vocab[seg])
                continue
            for pretoken in _pretokenize(seg):
                for tok in self._bpe_encode_pretoken(pretoken):
                    ids.append(self.vocab[tok])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        tokens = []
        for i in ids:
            tok = self._id_to_token.get(i)
            if tok is None:
                raise KeyError(f"Unknown token id: {i}")
            tokens.append(tok)

        text_parts: List[str] = []
        out_bytes = bytearray()
        for tok in tokens:
            if tok in self.special_tokens:
                if out_bytes:
                    text_parts.append(bytes(out_bytes).decode("utf-8", errors="replace"))
                    out_bytes = bytearray()
                text_parts.append(tok)
            else:
                out_bytes.extend(UNICODE_TO_BYTE[c] for c in tok)
        if out_bytes:
            text_parts.append(bytes(out_bytes).decode("utf-8", errors="replace"))
        return "".join(text_parts)

    # -- persistence --------------------------------------------------------------

    def save(self, path: str) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "vocab.json").write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "merges.txt").write_text(
            "\n".join(f"{a}\t{b}" for a, b in self.merges), encoding="utf-8"
        )
        (out_dir / "special_tokens.json").write_text(
            json.dumps(self.special_tokens, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPETokenizer":
        in_dir = Path(path)
        vocab = json.loads((in_dir / "vocab.json").read_text(encoding="utf-8"))
        merges_text = (in_dir / "merges.txt").read_text(encoding="utf-8")
        merges = []
        for line in merges_text.splitlines():
            if not line:
                continue
            a, b = line.split("\t")
            merges.append((a, b))
        special_tokens = json.loads(
            (in_dir / "special_tokens.json").read_text(encoding="utf-8")
        )
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
