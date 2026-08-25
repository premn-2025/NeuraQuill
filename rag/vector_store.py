"""A simple local (in-memory) vector store: cosine-similarity search over
sparse TF-IDF vectors. Same interface shape a pgvector/FAISS-backed store
would have (add / search), so swapping the backend later doesn't change
callers."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import cosine_similarity


@dataclass
class IndexedChunk:
    id: int
    text: str
    vector: Dict[int, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def __init__(self):
        self._items: List[IndexedChunk] = []

    def add(self, text: str, vector: Dict[int, float], metadata: Optional[Dict[str, Any]] = None) -> int:
        idx = len(self._items)
        self._items.append(IndexedChunk(id=idx, text=text, vector=vector, metadata=metadata or {}))
        return idx

    def __len__(self) -> int:
        return len(self._items)

    def search(self, query_vector: Dict[int, float], top_k: int = 5) -> List[Tuple[IndexedChunk, float]]:
        scored = [(item, cosine_similarity(query_vector, item.vector)) for item in self._items]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
