#WikiZip ***may be inaccurate due to data holes***

A tiny local question-answering setup that runs comfortably on 8GB of RAM.

**The idea:**  reference material from Wikipedia sits zipped, article by article. A small keyword index tells the system which article is relevant to a
question. nothing is unzipped permanently. When you ask something, only
the matching article(s) get read into memory straight out of whichever
zip holds them their text is handed to a small local language model as
context, and the model answers from that. The model itself is small and
the "knowledge" stays compressed on disk until it's actually needed.

The database contains roughly 1,700,000 articles of varying topics standing at about 4GB and is compiled through data dump API's via the wiki. 

## How it stays tiny

 The LLM is a small model...Roughly 1GB on disk


  The knowledge base stays compressed. `ingest.py` reads every article
  inside every zip once, in memory, to build a lightweight per-article
  word-frequency index, then discards the extracted text. Nothing is left
  unzipped on disk. The index grows with the knowledge base — a few KB for
  a small collection, tens of MB once you're indexing hundreds of
  thousands of articles — but it's still a small fraction of the size of
  the knowledge zips themselves, and loads fine into memory on 8GB RAM.

At query time, only the top 1–2 matching articles are opened and read
  directly out of whichever zip holds them, still in memory via Python's
  `zipfile`, never extracted to disk  and their text becomes the model's
  context for that one answer.



## Setup

1. **Install Ollama** (runs the local model — no Python packaging involved):
   - https://ollama.com/download, or `brew install ollama`
2. **Pull a small model:**
   ```
   ollama pull qwen2.5:1.5b
   ```
3. **Add material** to `knowledge/` as `.zip` files (each containing `.txt`/`.md` files), then build the index:
   ```
   python3 src/ingest.py
   ```

No `pip install` needed — everything in `src/` uses only Python's standard library.

## Usage

**Command line:**
```
python3 src/ask.py "your question"
```

**Web UI** (a small local browser interface instead of the CLI):
```
python3 src/server.py
```
Then open http://localhost:8765.

## Growing the knowledge base

Two different scripts for two different scales:

- **`src/fetch_history.py`** — pulls articles from a specific Wikipedia category tree (e.g. `Category:History`) via the live API, one request per article. Good for a few hundred to a couple thousand articles on a focused topic. Slow at large scale (one HTTP round-trip per article).

- **`src/fetch_bulk.py`** — streams Wikipedia's real bulk export dumps (`enwiki-latest-pages-articles-multistream*.bz2`) and pulls full article text, cleaned down from wikitext, instead of per-article API calls. This is the one for "as much as possible, fast." Articles are taken in order (not randomly sampled), stored close to full length, and written with LZMA compression (smaller than standard zip deflate on text like this). By default there's no article-count limit — it just runs until it hits a size cap, the configured dump files run out, or you stop it. Total output is capped by `--max-gb` (default 7.5 GB) regardless of article count; the script estimates its own compressed size as it goes, stops itself once it's near the cap, and reports the real on-disk size of `knowledge/` when done. Saves a checkpoint every 50,000 articles collected, so a dropped connection or Ctrl+C doesn't lose progress — whatever's been collected so far is already written out as real, usable zips.
  ```
  python3 src/fetch_bulk.py
  ```
  Add `--target N` to stop after a fixed article count instead, or `--max-gb N` to change the size cap.

Either way, re-run `python3 src/ingest.py` after adding new zips.
