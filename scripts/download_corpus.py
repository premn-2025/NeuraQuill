"""
Phase 3 data acquisition: download a curated set of public-domain
(Project Gutenberg, pre-1928 US-published, copyright-expired) books,
strip Gutenberg's license/header/footer boilerplate, and concatenate
them into one pretraining corpus.

Usage:
    python scripts/download_corpus.py --out data/pretrain_corpus_raw.txt
"""
import argparse
import http.client
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# (Gutenberg ebook id, title) -- all public domain, plain-text mirrors
BOOKS = [
    (1342, "Pride and Prejudice"),
    (84, "Frankenstein"),
    (11, "Alice's Adventures in Wonderland"),
    (1661, "The Adventures of Sherlock Holmes"),
    (98, "A Tale of Two Cities"),
    (2701, "Moby Dick"),
    (1228, "On the Origin of Species"),
    (2680, "Meditations"),
    (132, "The Art of War"),
    (76, "Adventures of Huckleberry Finn"),
    (5200, "Metamorphosis"),
    (345, "Dracula"),
    (2554, "Crime and Punishment"),
    (1952, "The Yellow Wallpaper"),
    (55, "The Wonderful Wizard of Oz"),
    # Phase 10 additions -- scaling up the pretraining corpus for
    # NeuraQuill-100M. IDs below are less exhaustively pre-verified than
    # the Phase 3 set above; main() skips (rather than crashes on) any
    # that fail to download, so an occasional wrong/dead ID is fine.
    (46, "A Christmas Carol"),
    (1400, "Great Expectations"),
    (730, "Oliver Twist"),
    (174, "The Picture of Dorian Gray"),
    (844, "The Importance of Being Earnest"),
    (43, "Strange Case of Dr Jekyll and Mr Hyde"),
    (35, "The Time Machine"),
    (36, "The War of the Worlds"),
    (120, "Treasure Island"),
    (209, "The Turn of the Screw"),
    (74, "The Adventures of Tom Sawyer"),
    (2591, "Grimm's Fairy Tales"),
    (100, "The Complete Works of William Shakespeare"),
    (1080, "A Modest Proposal"),
    (1497, "The Republic"),
    (2814, "Dubliners"),
    (6130, "The Iliad"),
    (1727, "The Odyssey"),
    (205, "Walden"),
    (2600, "War and Peace"),
]

URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)


def strip_boilerplate(text: str) -> str:
    """Remove Gutenberg's license header/footer, keeping just the book body."""
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def download_book(book_id: int, timeout: int = 30, retries: int = 3) -> str:
    url = URL_TEMPLATE.format(id=book_id)
    req = urllib.request.Request(url, headers={"User-Agent": "NeuraQuill-corpus-builder/0.1"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            return raw.decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, ConnectionError, OSError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download book {book_id} after {retries} attempts: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/pretrain_corpus_raw.txt")
    parser.add_argument("--raw-dir", type=str, default="data/gutenberg_raw")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_texts = []
    total_chars = 0
    n_ok = 0
    n_failed = 0
    for book_id, title in BOOKS:
        cache_path = raw_dir / f"{book_id}.txt"
        try:
            if cache_path.exists():
                body = cache_path.read_text(encoding="utf-8")
                print(f"[cache] {title} ({book_id}) -- {len(body):,} chars")
            else:
                print(f"[download] {title} ({book_id}) ...", end=" ", flush=True)
                raw = download_book(book_id)
                body = strip_boilerplate(raw)
                if len(body) < 1000:
                    raise RuntimeError(f"suspiciously short body ({len(body)} chars) -- likely a bad id")
                cache_path.write_text(body, encoding="utf-8")
                print(f"{len(body):,} chars")
        except Exception as e:
            print(f"[skip] {title} ({book_id}) -- {e}")
            n_failed += 1
            continue
        all_texts.append(f"\n\n<|endoftext|>\n\n{body}")
        total_chars += len(body)
        n_ok += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(all_texts), encoding="utf-8")

    print(f"\nWrote {out_path} -- {total_chars:,} chars from {n_ok} books "
          f"(~{total_chars / 1e6:.1f} MB); skipped {n_failed} failed downloads")


if __name__ == "__main__":
    main()
