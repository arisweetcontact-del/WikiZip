"""Ask a question. Scores every ARTICLE in knowledge/ against your question
using the SQLite keyword index (built by ingest.py), then reads ONLY the
best-matching article(s) out of their zip in memory (not the whole zip),
and hands that text to a small local model (served by Ollama) as context.

Usage:
    python3 src/ask.py "what does the onboarding doc say about week one?"
"""

import json
import math
import os
import sqlite3
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import tokenize, read_zip_member_text, split_doc_key  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..")
KNOWLEDGE_DIR = os.path.join(BASE, "knowledge")
INDEX_PATH = os.path.join(BASE, "index.db")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"

TOP_K = 2
MAX_CONTEXT_CHARS = 12000

# Query-time safety valve: a term appearing in more than this fraction of
# all documents is treated as a stopword and skipped, rather than pulling
# what could be well over a million postings rows for it. Set high enough
# (0.5) that it only catches near-universal words ("the", "of", "and"),
# not legitimately common topical words, which still carry real tf-idf
# signal even at moderate document frequency.
MAX_DOC_FRACTION = 0.5


def open_index():
    """Open the index read-only. Safe to share across threads (server.py
    caches one connection and reuses it for every request)."""
    if not os.path.exists(INDEX_PATH):
        return None
    return sqlite3.connect(f"file:{INDEX_PATH}?mode=ro", uri=True, check_same_thread=False)


def rank(question, conn):
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return []

    num_docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    if num_docs == 0:
        return []

    scores = {}
    for t in q_tokens:
        term_row = conn.execute("SELECT id FROM terms WHERE term = ?", (t,)).fetchone()
        if term_row is None:
            continue
        term_id = term_row[0]

        df_row = conn.execute("SELECT df FROM doc_freq WHERE term_id = ?", (term_id,)).fetchone()
        if df_row is None:
            continue
        df = df_row[0]
        if df / num_docs > MAX_DOC_FRACTION:
            continue  # too common to be useful — skip scanning its postings

        idf = math.log((num_docs + 1) / (df + 1)) + 1
        for doc_id, tf, length in conn.execute(
            """SELECT p.doc_id, p.tf, d.length
               FROM postings p JOIN docs d ON p.doc_id = d.id
               WHERE p.term_id = ?""",
            (term_id,),
        ):
            scores[doc_id] = scores.get(doc_id, 0.0) + (tf / max(length, 1)) * idf

    ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:50]
    results = []
    for doc_id, score in ranked_ids:
        row = conn.execute("SELECT doc_key FROM docs WHERE id = ?", (doc_id,)).fetchone()
        if row:
            results.append((score, row[0]))
    return results


def build_context(top_docs):
    """top_docs: list of doc keys, each "zipfile::member". Reads each
    matched article's full text directly out of its zip (never the whole
    zip), so a full-length article is actually usable as context."""
    parts = []
    per_doc_budget = MAX_CONTEXT_CHARS // max(len(top_docs), 1)
    for doc_key in top_docs:
        zip_filename, member_name = split_doc_key(doc_key)
        zip_path = os.path.join(KNOWLEDGE_DIR, zip_filename)
        text = read_zip_member_text(zip_path, member_name, max_chars=per_doc_budget)
        label = os.path.splitext(os.path.basename(member_name))[0].replace("_", " ")
        parts.append(f"[Source: {label} ({zip_filename})]\n{text}")
    return "\n\n".join(parts)


def ask_ollama(prompt):
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        return (
            f"Couldn't reach Ollama at {OLLAMA_URL} ({e}).\n"
            f"Make sure Ollama is running and you've pulled the model:\n"
            f"  ollama pull {OLLAMA_MODEL}"
        )


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 src/ask.py "your question"')
        return
    question = " ".join(sys.argv[1:])

    conn = open_index()
    if conn is None:
        print("No index found — run `python3 src/ingest.py` first.")
        return

    ranked = [(s, f) for s, f in rank(question, conn) if s > 0][:TOP_K]

    if not ranked:
        print("No relevant article matched that question.")
        top_docs = []
    else:
        top_docs = [f for _, f in ranked]
        print("Using article(s):", ", ".join(top_docs))

    context = build_context(top_docs) if top_docs else ""

    system = (
        "You are a helpful assistant. Answer using only the provided context. "
        "If the answer isn't in the context, say you don't know."
    )
    prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    print("\n" + ask_ollama(prompt))


if __name__ == "__main__":
    main()
