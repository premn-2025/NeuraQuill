"""
End-to-end RAG demo: retrieve relevant chunks for a question, splice them
into the chat prompt as context, and generate an answer with a real
NeuraQuill checkpoint -- showing the model can be handed knowledge it
never saw during training.

Usage:
    python scripts/rag_chat.py --question "How much does the X200 weigh?"
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rag.retriever import build_index, retrieve  # noqa: E402
from tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/neuraquill-10m-sft.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/artifacts_chat")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    print("Indexing knowledge base ...")
    store, vectorizer, chunks = build_index(KNOWLEDGE_BASE, chunk_size=50, overlap=10)
    results = retrieve(args.question, store, vectorizer, top_k=args.top_k)

    print(f"\nRetrieved context for: {args.question!r}")
    context_parts = []
    for chunk, score in results:
        print(f"  [score={score:.3f}, doc={chunk.metadata['doc_id']}] {chunk.text}")
        context_parts.append(chunk.text)
    context = " ".join(context_parts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, step, best_val_loss, config = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    tok = ByteLevelBPETokenizer.load(args.tokenizer)
    eot_id = tok.vocab["<|endoftext|>"]

    system_prompt = "You are a helpful assistant. Use the provided context to answer the question."
    prompt_text = (
        f"<|system|>{system_prompt}<|endoftext|>"
        f"<|user|>Context: {context}\n\nQuestion: {args.question}<|endoftext|>"
        "<|assistant|>"
    )
    ids = tok.encode(prompt_text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=0.5, top_k=40, stop_token_ids={eot_id})
    reply_ids = out[0, len(ids):].tolist()
    if reply_ids and reply_ids[-1] == eot_id:
        reply_ids = reply_ids[:-1]
    reply = tok.decode(reply_ids)

    print(f"\nModel answer: {reply}")


if __name__ == "__main__":
    main()
