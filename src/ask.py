"""Ask a question. Scores every ARTICLE in knowledge/ against your question
using the tiny index, then reads ONLY the best-matching article(s) out of
their zip in memory (not the whole zip), and hands that text to a small
local model (served by Ollama) as context.

Usage:
    python3 src/ask.py "what does the onboarding doc say about week one?"
"""

import json
import math
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import tokenize, read_zip_member_text, split_doc_key  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..")
KNOWLEDGE_DIR = os.path.join(BASE, "knowledge")
INDEX_PATH = os.path.join(BASE, "index.json")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"

TOP_K = 2
# Total characters of article text handed to the model as context, split
# across the top matches. Doubled from the original 6000 now that retrieval
# is per-article (full articles, not just whatever came first in a zip) —
# still comfortably within qwen2.5:1.5b's context window on 8GB RAM/CPU.
MAX_CONTEXT_CHARS = 12000


def load_index():
    with open(INDEX_PATH) as f:
        return json.load(f)


def score(query_tokens, doc_counts, doc_len, df, num_docs):
    s = 0.0
    for t in query_tokens:
        if t not in doc_counts:
            continue
        tf = doc_counts[t] / max(doc_len, 1)
        idf = math.log((num_docs + 1) / (df.get(t, 0) + 1)) + 1
        s += tf * idf
    return s


def rank(question, index):
    q_tokens = tokenize(question)
    scores = []
    for doc_key, counts in index["doc_freqs"].items():
        s = score(q_tokens, counts, index["doc_lengths"][doc_key], index["df"], index["num_docs"])
        scores.append((s, doc_key))
    scores.sort(reverse=True)
    return scores


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

    if not os.path.exists(INDEX_PATH):
        print("No index found — run `python3 src/ingest.py` first.")
        return

    index = load_index()
    ranked = [(s, f) for s, f in rank(question, index) if s > 0][:TOP_K]

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
