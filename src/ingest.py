"""Build a tiny keyword index over the zip files in knowledge/.

Indexes each ARTICLE individually (not each zip as one blob), so retrieval
can find and return the one matching article inside a zip that holds many —
critical once zips hold hundreds or thousands of articles each. The zip
contents themselves are read into memory once per article, indexed, and
discarded. Nothing is left unzipped on disk. Re-run this whenever you add
or change a .zip in knowledge/.

Note on index size: because every article gets its own entry, index.json
grows with your knowledge base — it's no longer just "a few KB." For a
knowledge base in the hundreds-of-MB range, expect an index in the tens of
MB, not KB. Still tiny next to the knowledge/ zips themselves, and still
loads comfortably into memory on 8GB RAM.
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from common import tokenize, iter_zip_members, make_doc_key  # noqa: E402

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
    total_articles = 0

    for fname in zip_files:
        path = os.path.join(KNOWLEDGE_DIR, fname)
        count_in_zip = 0
        for member_name, text in iter_zip_members(path):
            tokens = tokenize(text)
            if not tokens:
                continue
            key = make_doc_key(fname, member_name)
            counts = Counter(tokens)
            doc_freqs[key] = counts
            doc_lengths[key] = len(tokens)
            for tok in counts:
                df[tok] += 1
            count_in_zip += 1
        total_articles += count_in_zip
        print(f"  indexed {fname}  ({count_in_zip} article(s))")

    index = {
        "num_docs": len(doc_freqs),
        "doc_freqs": doc_freqs,
        "doc_lengths": doc_lengths,
        "df": df,
    }
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f)
    size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    print(f"\nWrote index for {total_articles} article(s) across {len(zip_files)} zip(s) "
          f"-> {INDEX_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build_index()
