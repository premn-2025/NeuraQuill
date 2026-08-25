"""
A from-scratch TF-IDF vectorizer + cosine similarity -- no sklearn, no
external embedding model. This is a deliberate choice over using
NeuraQuill's own token embeddings as a retrieval signal: a 10M-parameter
decoder trained mostly for next-token prediction makes a noisy, unreliable
similarity signal at this scale, whereas TF-IDF is deterministic,
interpretable, and reliable for the keyword-driven retrieval this project
tests against.
"""
import math
import re
from collections import Counter
from typing import Dict, List

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


class TfidfVectorizer:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self._fitted = False

    def fit(self, documents: List[str]) -> "TfidfVectorizer":
        n_docs = len(documents)
        doc_freq: Counter = Counter()
        for doc in documents:
            for term in set(tokenize(doc)):
                doc_freq[term] += 1

        self.vocab = {term: i for i, term in enumerate(sorted(doc_freq))}
        self.idf = [0.0] * len(self.vocab)
        for term, i in self.vocab.items():
            # smoothed idf: ln((1+N)/(1+df)) + 1 -- always positive, and a
            # term in every document still gets a small positive weight
            # rather than collapsing to zero
            self.idf[i] = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
        self._fitted = True
        return self

    def transform(self, text: str) -> Dict[int, float]:
        if not self._fitted:
            raise RuntimeError("call fit() before transform()")
        terms = tokenize(text)
        if not terms:
            return {}
        tf = Counter(terms)
        max_tf = max(tf.values())
        vec: Dict[int, float] = {}
        for term, count in tf.items():
            idx = self.vocab.get(term)
            if idx is None:
                continue  # out-of-vocabulary term at query time -- ignored, not an error
            tf_weight = 0.5 + 0.5 * (count / max_tf)  # sublinear TF scaling
            vec[idx] = tf_weight * self.idf[idx]
        return vec

    def fit_transform(self, documents: List[str]) -> List[Dict[int, float]]:
        self.fit(documents)
        return [self.transform(doc) for doc in documents]


def cosine_similarity(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[i] * b[i] for i in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
