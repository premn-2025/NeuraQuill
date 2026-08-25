"""
Phase 8 test suite: chunking, TF-IDF embeddings, the local vector store,
and end-to-end retrieval precision on a known small corpus.

Run with: pytest tests/test_rag.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunking import chunk_documents, chunk_text  # noqa: E402
from rag.embeddings import TfidfVectorizer, cosine_similarity, tokenize  # noqa: E402
from rag.retriever import build_index, retrieve  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402

# --------------------------------------------------------------------------
# chunking: boundary correctness
# --------------------------------------------------------------------------


def test_chunk_text_covers_entire_document_via_words():
    text = " ".join(f"word{i}" for i in range(120))
    chunks = chunk_text("doc1", text, chunk_size=50, overlap=10)
    # every word index in [0, 120) must be covered by at least one chunk
    covered = set()
    for c in chunks:
        covered.update(range(c.start_word, c.end_word))
    assert covered == set(range(120))


def test_chunk_text_boundaries_match_actual_word_split():
    text = " ".join(f"word{i}" for i in range(120))
    words = text.split()
    chunks = chunk_text("doc1", text, chunk_size=50, overlap=10)
    for c in chunks:
        assert c.text == " ".join(words[c.start_word : c.end_word])


def test_chunk_text_overlap_is_correct():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text("doc1", text, chunk_size=30, overlap=5)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.start_word == prev.end_word - 5


def test_chunk_text_short_document_yields_one_chunk():
    text = "just a few words here"
    chunks = chunk_text("doc1", text, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == len(text.split())


def test_chunk_text_empty_document_yields_no_chunks():
    assert chunk_text("doc1", "", chunk_size=50, overlap=10) == []
    assert chunk_text("doc1", "   ", chunk_size=50, overlap=10) == []


def test_chunk_text_rejects_overlap_ge_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("doc1", "a b c", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("doc1", "a b c", chunk_size=10, overlap=15)


def test_chunk_text_no_duplicate_chunk_ever_repeats_the_whole_remainder():
    """Regression guard: a naive off-by-one in the overlap step can loop
    forever or re-emit the same final chunk twice."""
    text = " ".join(f"word{i}" for i in range(53))  # not an exact multiple of chunk_size
    chunks = chunk_text("doc1", text, chunk_size=20, overlap=5)
    assert chunks[-1].end_word == 53
    # no two chunks should be identical in span
    spans = [(c.start_word, c.end_word) for c in chunks]
    assert len(spans) == len(set(spans))


def test_chunk_documents_tags_each_chunk_with_its_source_doc():
    docs = {"a": " ".join(["x"] * 60), "b": " ".join(["y"] * 30)}
    chunks = chunk_documents(docs, chunk_size=25, overlap=5)
    doc_ids = {c.doc_id for c in chunks}
    assert doc_ids == {"a", "b"}
    assert all(c.doc_id == "a" for c in chunks if c.text.startswith("x"))


# --------------------------------------------------------------------------
# TF-IDF embeddings
# --------------------------------------------------------------------------


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_tfidf_identical_documents_have_similarity_one():
    vec = TfidfVectorizer()
    v1, v2 = vec.fit_transform(["the cat sat on the mat", "the cat sat on the mat"])
    assert cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-6)


def test_tfidf_unrelated_documents_have_lower_similarity_than_related_ones():
    vec = TfidfVectorizer()
    docs = [
        "the cat sat on the mat",
        "a cat is sitting on a mat right now",
        "quantum physics describes subatomic particles",
    ]
    v1, v2, v3 = vec.fit_transform(docs)
    sim_related = cosine_similarity(v1, v2)
    sim_unrelated = cosine_similarity(v1, v3)
    assert sim_related > sim_unrelated


def test_tfidf_common_terms_get_lower_idf_than_rare_terms():
    vec = TfidfVectorizer()
    vec.fit(["the fox ran", "the dog ran", "the cat ran", "a rare zorblatt appeared"])
    idf_the = vec.idf[vec.vocab["the"]]  # appears in 3/4 docs
    idf_zorblatt = vec.idf[vec.vocab["zorblatt"]]  # appears in 1/4 docs
    assert idf_zorblatt > idf_the


def test_tfidf_transform_before_fit_raises():
    vec = TfidfVectorizer()
    with pytest.raises(RuntimeError):
        vec.transform("hello")


def test_tfidf_out_of_vocabulary_query_terms_are_ignored_not_error():
    vec = TfidfVectorizer()
    vec.fit(["the cat sat on the mat"])
    result = vec.transform("a completely unrelated xyzzy plugh query")
    assert isinstance(result, dict)  # doesn't raise; OOV terms just contribute nothing


def test_cosine_similarity_empty_vector_is_zero():
    assert cosine_similarity({}, {0: 1.0}) == 0.0
    assert cosine_similarity({}, {}) == 0.0


# --------------------------------------------------------------------------
# VectorStore
# --------------------------------------------------------------------------


def test_vector_store_search_ranks_by_similarity():
    store = VectorStore()
    store.add("a", {0: 1.0})
    store.add("b", {0: 0.5, 1: 0.5})
    store.add("c", {1: 1.0})
    results = store.search({0: 1.0}, top_k=3)
    assert [item.text for item, _ in results] == ["a", "b", "c"]


def test_vector_store_respects_top_k():
    store = VectorStore()
    for i in range(10):
        store.add(f"doc{i}", {0: 1.0})
    results = store.search({0: 1.0}, top_k=3)
    assert len(results) == 3


def test_vector_store_empty_search_returns_empty():
    store = VectorStore()
    assert store.search({0: 1.0}, top_k=5) == []


def test_vector_store_len():
    store = VectorStore()
    assert len(store) == 0
    store.add("a", {0: 1.0})
    assert len(store) == 1


# --------------------------------------------------------------------------
# end-to-end retrieval precision on a known small corpus -- the explicit
# PLAN.md requirement for this phase
# --------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "zorblatt": "The Zorblatt is a fictional creature that lives in deep caves and eats crystals for energy.",
    "x200": "The X200 gadget has a battery life of 14 hours and weighs exactly 300 grams.",
    "glimmerfern": "The Glimmerfern is a rare plant found only on the slopes of Mount Kavara, blooming at midnight.",
    "captain_orlo": "Captain Orlo led the expedition across the Thundersand Desert in the year 1842.",
    "blueglass_lake": "Blueglass Lake is famous for its perfectly still, mirror-like water and glowing blue algae.",
    "recipe_stew": "The traditional Kelvorian stew recipe requires wild mushrooms, root vegetables, and smoked paprika.",
    "starforge_engine": "The Starforge engine converts solidified starlight into propulsion energy for deep space travel.",
    "whistling_bridge": "The Whistling Bridge makes a musical humming sound whenever strong winds pass through its cables.",
    "moonpetal_tea": "Moonpetal tea is brewed from dried moonpetal flowers and is said to help with restful sleep.",
    "iron_falcon_guild": "The Iron Falcon Guild was a merchant organization that controlled trade routes through the Karvos mountains.",
}

QUERIES = [
    ("What does a Zorblatt eat?", "zorblatt"),
    ("How much does the X200 weigh?", "x200"),
    ("Where does the Glimmerfern plant grow?", "glimmerfern"),
    ("Who led the expedition across the Thundersand Desert?", "captain_orlo"),
    ("What is special about the water in Blueglass Lake?", "blueglass_lake"),
    ("What ingredients go into Kelvorian stew?", "recipe_stew"),
    ("What powers the Starforge engine?", "starforge_engine"),
    ("Why does the Whistling Bridge make sound?", "whistling_bridge"),
    ("What is Moonpetal tea used for?", "moonpetal_tea"),
    ("What did the Iron Falcon Guild control?", "iron_falcon_guild"),
]


@pytest.fixture(scope="module")
def kb_index():
    return build_index(KNOWLEDGE_BASE, chunk_size=50, overlap=10)


def test_retrieval_precision_at_1_on_known_corpus(kb_index):
    store, vectorizer, chunks = kb_index
    correct = 0
    for query, expected_doc_id in QUERIES:
        results = retrieve(query, store, vectorizer, top_k=1)
        assert len(results) == 1
        top_chunk, score = results[0]
        if top_chunk.metadata["doc_id"] == expected_doc_id:
            correct += 1
    assert correct == len(QUERIES), f"precision@1 = {correct}/{len(QUERIES)}"


def test_retrieval_recall_at_3_on_known_corpus(kb_index):
    store, vectorizer, chunks = kb_index
    for query, expected_doc_id in QUERIES:
        results = retrieve(query, store, vectorizer, top_k=3)
        retrieved_doc_ids = {chunk.metadata["doc_id"] for chunk, _ in results}
        assert expected_doc_id in retrieved_doc_ids


def test_retrieval_returns_relevant_chunk_from_a_multi_chunk_document():
    """When a document spans multiple chunks, a query about content only
    in a later chunk must retrieve *that* chunk, not just chunk 0 of the
    right document -- validates chunk-level (not document-level) granularity."""
    long_doc = (
        "This document opens with unrelated filler about gardening and weather patterns "
        "in temperate climates, going on for quite a while about soil composition and rainfall. "
    ) * 3 + "Deep in the middle of this document, the secret launch code is Falcon-7-Zebra. " + (
        "More unrelated filler follows about cooking recipes and kitchen equipment maintenance. "
    ) * 3
    store, vectorizer, chunks = build_index({"long": long_doc}, chunk_size=30, overlap=5)
    results = retrieve("What is the secret launch code?", store, vectorizer, top_k=1)
    assert "Falcon-7-Zebra" in results[0][0].text


def test_retrieval_on_empty_index_returns_empty():
    store, vectorizer, chunks = build_index({}, chunk_size=50, overlap=10)
    results = retrieve("anything", store, vectorizer, top_k=5)
    assert results == []
