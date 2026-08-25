"""Ties chunking + embeddings + vector store together: build an index over
a set of documents, then retrieve the top-k most relevant chunks for a
query."""
from typing import Dict, List, Tuple

from .chunking import Chunk, chunk_documents
from .embeddings import TfidfVectorizer
from .vector_store import IndexedChunk, VectorStore


def build_index(
    documents: Dict[str, str], chunk_size: int = 50, overlap: int = 10
) -> Tuple[VectorStore, TfidfVectorizer, List[Chunk]]:
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([c.text for c in chunks])

    store = VectorStore()
    for chunk, vector in zip(chunks, vectors):
        store.add(chunk.text, vector, metadata={"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id})
    return store, vectorizer, chunks


def retrieve(
    query: str, store: VectorStore, vectorizer: TfidfVectorizer, top_k: int = 3
) -> List[Tuple[IndexedChunk, float]]:
    query_vector = vectorizer.transform(query)
    return store.search(query_vector, top_k=top_k)
