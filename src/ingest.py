"""Build a tiny keyword index over the zip files in knowledge/.

The index only stores word-frequency counts per zip (a few KB of JSON) —
the zip contents themselves are read into memory once, indexed, and
discarded. Nothing is left unzipped on disk. Re-run this whenever you
add or change a .zip in knowledge/.
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from common import tokenize, read_zip_text  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..")
KNOWLEDGE_DIR = os.path.join(BASE, "knowledge")
INDEX_PATH = os.path.join(BASE, "index.json")


def build_index():
    if not os.path.isdir(KNOWLEDGE_DIR):
        print(f"No knowledge/ folder found at {KNOWLEDGE_DIR}")
        return

    zip_files = sorted(f for f in os.listdir(KNOWLEDGE_DIR) if f.lower().endswith(".zip"))
    if not zip_files:
        print(f"No .zip files found in {KNOWLEDGE_DIR} — add some and re-run.")
        return

    doc_freqs = {}
    doc_lengths = {}
    df = Counter()

    for fname in zip_files:
        path = os.path.join(KNOWLEDGE_DIR, fname)
        text = read_zip_text(path)
        tokens = tokenize(text)
        counts = Counter(tokens)
        doc_freqs[fname] = counts
        doc_lengths[fname] = len(tokens)
        for tok in counts:
            df[tok] += 1
        print(f"  indexed {fname}  ({len(tokens)} tokens)")

    index = {
        "num_docs": len(zip_files),
        "doc_freqs": doc_freqs,
        "doc_lengths": doc_lengths,
        "df": df,
    }
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f)
    print(f"\nWrote index for {len(zip_files)} zip(s) -> {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
