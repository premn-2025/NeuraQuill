"""
Real-world Phase 8 check: index a small set of invented facts (things
NeuraQuill could not possibly know from its training corpus) mixed into
thousands of chunks of real book text (the actual Phase 3 pretraining
corpus, as realistic-scale "noise"), then measure retrieval precision on
the invented-fact queries within that much bigger haystack.

Usage:
    python scripts/eval_rag.py
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.retriever import build_index, retrieve  # noqa: E402

INVENTED_FACTS = {
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


def main():
    corpus_path = Path("data/pretrain_corpus_raw.txt")
    documents = dict(INVENTED_FACTS)

    if corpus_path.exists():
        text = corpus_path.read_text(encoding="utf-8")
        # split the real corpus into pseudo-"documents" of ~2000 words each,
        # purely so build_index's per-document chunking has bite-sized units
        words = text.split()
        doc_size = 2000
        for i in range(0, len(words), doc_size):
            documents[f"gutenberg_{i // doc_size}"] = " ".join(words[i : i + doc_size])
        print(f"Mixed in {len(words) // doc_size:,} real book pseudo-documents from {corpus_path}")
    else:
        print(f"({corpus_path} not found -- running precision test on invented facts only)")

    print(f"Building index over {len(documents):,} documents ...")
    store, vectorizer, chunks = build_index(documents, chunk_size=60, overlap=10)
    print(f"Index built: {len(chunks):,} chunks total, vocab size {len(vectorizer.vocab):,}")

    correct_at_1 = 0
    correct_at_3 = 0
    for query, expected_doc_id in QUERIES:
        results = retrieve(query, store, vectorizer, top_k=3)
        top_doc_id = results[0][0].metadata["doc_id"] if results else None
        retrieved_ids = {c.metadata["doc_id"] for c, _ in results}
        hit1 = top_doc_id == expected_doc_id
        hit3 = expected_doc_id in retrieved_ids
        correct_at_1 += int(hit1)
        correct_at_3 += int(hit3)
        print(f"  [{'OK ' if hit1 else 'MISS'}] {query!r} -> top doc: {top_doc_id} (expected {expected_doc_id})")

    n = len(QUERIES)
    print(f"\nPrecision@1: {correct_at_1}/{n} ({correct_at_1 / n * 100:.1f}%)")
    print(f"Recall@3:    {correct_at_3}/{n} ({correct_at_3 / n * 100:.1f}%)")
    print(f"(haystack size: {len(chunks):,} chunks, {len(documents):,} documents)")


if __name__ == "__main__":
    main()
