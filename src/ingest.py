"""Build a SQLite keyword index over the zip files in knowledge/.

Earlier versions built the whole index in memory as one big dict of
per-article word-frequency counters, then wrote it out as a single JSON
file at the very end. That worked fine at thousands of articles, but at
~1.5 million full-length articles it needs tens of GB of RAM to hold
everything before it can write anything — and gets killed by the OS on an
8GB machine before it ever finishes.

This version writes straight to a SQLite database (index.db) as it scans,
one article at a time. Nothing about the whole knowledge base is ever held
in memory at once — RAM use stays roughly constant regardless of how many
articles you have; only disk space grows. Vocabulary (unique words seen so
far) is cached in memory for speed, which does grow with the knowledge
base, but far more slowly than per-article word counts would.

Re-run this whenever you add or change a .zip in knowledge/. It fully
rebuilds index.db from scratch each time.

Note on index size: a full per-article inverted index over ~1.5 million
real articles is a genuinely large structure — index.db can end up
several GB, possibly bigger than the knowledge/ zips themselves. That's
expected and is not counted against --max-gb (which only caps knowledge/).
Trading disk space for RAM is the right trade here: disk is cheap and
plentiful, RAM on an 8GB machine is not.
"""

import os
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from common import tokenize, iter_zip_members, make_doc_key  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..")
KNOWLEDGE_DIR = os.path.join(BASE, "knowledge")
INDEX_PATH = os.path.join(BASE, "index.db")

COMMIT_EVERY = 2000  # articles between commits — keeps a crash from losing much


def build_index():
    if not os.path.isdir(KNOWLEDGE_DIR):
        print(f"No knowledge/ folder found at {KNOWLEDGE_DIR}")
        return

    zip_files = sorted(f for f in os.listdir(KNOWLEDGE_DIR) if f.lower().endswith(".zip"))
    if not zip_files:
        print(f"No .zip files found in {KNOWLEDGE_DIR} — add some and re-run.")
        return

    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    for ext in ("-wal", "-shm"):
        stale = INDEX_PATH + ext
        if os.path.exists(stale):
            os.remove(stale)

    conn = sqlite3.connect(INDEX_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("""CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        doc_key TEXT UNIQUE,
        length INTEGER
    )""")
    conn.execute("""CREATE TABLE terms (
        id INTEGER PRIMARY KEY,
        term TEXT UNIQUE
    )""")
    conn.execute("""CREATE TABLE postings (
        term_id INTEGER,
        doc_id INTEGER,
        tf INTEGER
    )""")

    # In-memory term -> id cache. This grows with vocabulary size (unique
    # words), not with corpus size (total word occurrences) — a much
    # smaller number, and the only thing besides SQLite's own small
    # buffers kept in memory during the scan.
    term_ids = {}
    next_term_id = 1

    total_articles = 0
    since_commit = 0
    for fname in zip_files:
        path = os.path.join(KNOWLEDGE_DIR, fname)
        count_in_zip = 0
        for member_name, text in iter_zip_members(path):
            tokens = tokenize(text)
            if not tokens:
                continue
            counts = Counter(tokens)

            key = make_doc_key(fname, member_name)
            cur = conn.execute(
                "INSERT INTO docs (doc_key, length) VALUES (?, ?)",
                (key, len(tokens)),
            )
            doc_id = cur.lastrowid

            posting_rows = []
            for term, tf in counts.items():
                tid = term_ids.get(term)
                if tid is None:
                    tid = next_term_id
                    next_term_id += 1
                    term_ids[term] = tid
                    conn.execute("INSERT INTO terms (id, term) VALUES (?, ?)", (tid, term))
                posting_rows.append((tid, doc_id, tf))
            conn.executemany(
                "INSERT INTO postings (term_id, doc_id, tf) VALUES (?, ?, ?)",
                posting_rows,
            )

            count_in_zip += 1
            total_articles += 1
            since_commit += 1
            if since_commit >= COMMIT_EVERY:
                conn.commit()
                since_commit = 0
        print(f"  indexed {fname}  ({count_in_zip} article(s), "
              f"{total_articles:,} total, {len(term_ids):,} unique terms so far)")
    conn.commit()

    print("\nBuilding document-frequency table (one pass over postings)...")
    conn.execute("""CREATE TABLE doc_freq AS
        SELECT term_id, COUNT(*) AS df FROM postings GROUP BY term_id""")
    conn.commit()

    print("Building indexes for fast lookup (this can take a while on a large knowledge base)...")
    conn.execute("CREATE UNIQUE INDEX idx_doc_freq_term ON doc_freq(term_id)")
    conn.execute("CREATE INDEX idx_postings_term ON postings(term_id)")
    conn.commit()

    conn.execute("PRAGMA journal_mode=DELETE")  # merge WAL back into the main file when done
    conn.commit()
    conn.close()

    size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    print(f"\nWrote index for {total_articles:,} article(s) across {len(zip_files)} zip(s), "
          f"{len(term_ids):,} unique terms -> {INDEX_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build_index()
