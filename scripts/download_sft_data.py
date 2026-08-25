"""
Download the Databricks Dolly-15k instruction dataset (CC BY-SA 3.0,
permits commercial use) -- ~15k human-written instruction/response pairs.

Usage:
    python scripts/download_sft_data.py --out data/dolly15k.jsonl
"""
import argparse
import http.client
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/dolly15k.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"[cache] {out_path} already exists ({out_path.stat().st_size:,} bytes)")
        return

    req = urllib.request.Request(URL, headers={"User-Agent": "NeuraQuill-sft-data/0.1"})
    last_err = None
    for attempt in range(4):
        try:
            print(f"Downloading {URL} ...")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            print(f"Wrote {out_path} ({len(data):,} bytes)")
            return
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, ConnectionError, OSError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download dolly15k after retries: {last_err}")


if __name__ == "__main__":
    main()
