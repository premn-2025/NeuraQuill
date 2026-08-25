"""Word-based chunking with overlap. Word-based (not character-based)
keeps chunk boundaries at whole-word edges, which is what makes retrieved
chunks readable as standalone context rather than fragments cut mid-word."""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    start_word: int
    end_word: int  # exclusive


def chunk_text(doc_id: str, text: str, chunk_size: int = 50, overlap: int = 10) -> List[Chunk]:
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    words = text.split()
    n = len(words)
    if n == 0:
        return []

    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=" ".join(words[start:end]),
                start_word=start,
                end_word=end,
            )
        )
        chunk_id += 1
        if end == n:
            break
        start = end - overlap
    return chunks


def chunk_documents(documents: Dict[str, str], chunk_size: int = 50, overlap: int = 10) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc_id, text in documents.items():
        all_chunks.extend(chunk_text(doc_id, text, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
