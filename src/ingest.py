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
rebuilds index.db from scratch each time — there is no partial-recovery
promise here, so the build is tuned for raw write speed rather than crash
safety (see the PRAGMAs below).

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

COMMIT_EVERY = 20000  # articles between commits — purely for progress visibility;
                       # see the PRAGMAs in build_index() for why this can be large

# Words common enough that they're useless for retrieval — ask.py already
# skips any term appearing in more than half of all documents
# (MAX_DOC_FRACTION), so a stopword's postings would never be read at query
# time anyway. Filtering them out here means we never write those rows in
# the first place: fewer rows to insert, a smaller index.db, and a faster
# build, with no change in what questions can be answered. Document
# `length` (used to normalize term frequency at query time) still counts
# every token, stopwords included, so scoring math is unaffected.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some
such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under
until up very was wasn't we we'd we'll we're we've were weren't what what's
when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself
yourselves
""".split())


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
    for ext in ("-wal", "-shm", "-journal"):
        stale = INDEX_PATH + ext
        if os.path.exists(stale):
            os.remove(stale)

    conn = sqlite3.connect(INDEX_PATH)
    # This build is always a from-scratch rebuild (index.db was just
    # deleted above), and a crash mid-build just means re-running — there's
    # nothing to roll back to. So we skip SQLite's crash-safety machinery
    # entirely rather than pay for it: no rollback journal, no fsyncs, a
    # big page cache, and temp structures (used by the GROUP BY / CREATE
    # INDEX steps at the end) kept in RAM instead of on disk. This is
    # several times faster for a one-shot bulk load like this than the
    # default durability-first settings.
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")  # ~200MB page cache
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
    duplicate_articles = 0
    since_commit = 0
    for fname in zip_files:
        path = os.path.join(KNOWLEDGE_DIR, fname)
        count_in_zip = 0
        for member_name, text in iter_zip_members(path):
            tokens = tokenize(text)
            if not tokens:
                continue
            counts = Counter(t for t in tokens if t not in STOPWORDS)
            if not counts:
                continue  # an article that was nothing but stopwords/noise

            key = make_doc_key(fname, member_name)
            # INSERT OR IGNORE rather than a plain INSERT: with ~1.5M articles
            # collected across more than one fetch_bulk.py run, the same
            # article can legitimately show up twice (e.g. runs whose page
            # ranges overlapped slightly). That's not corruption — just skip
            # the repeat rather than crashing the whole build.
            cur = conn.execute(
                "INSERT OR IGNORE INTO docs (doc_key, length) VALUES (?, ?)",
                (key, len(tokens)),
            )
            if cur.rowcount == 0:
                duplicate_articles += 1
                continue
            doc_id = cur.lastrowid

            posting_rows = []
            new_term_rows = []
            for term, tf in counts.items():
                tid = term_ids.get(term)
                if tid is None:
                    tid = next_term_id
                    next_term_id += 1
                    term_ids[term] = tid
                    new_term_rows.append((tid, term))
                posting_rows.append((tid, doc_id, tf))
            if new_term_rows:
                conn.executemany(
                    "INSERT INTO terms (id, term) VALUES (?, ?)", new_term_rows
                )
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
    conn.close()

    size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    dup_note = f", skipped {duplicate_articles:,} duplicate article(s)" if duplicate_articles else ""
    print(f"\nWrote index for {total_articles:,} article(s) across {len(zip_files)} zip(s){dup_note}, "
          f"{len(term_ids):,} unique terms -> {INDEX_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build_index()
