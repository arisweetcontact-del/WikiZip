#WikiZip (NOT USEABLE YET! WILL FAIL IF RUN)

A tiny local question-answering setup that runs on 8GB of RAM, no GPU.

**The idea:**  reference material sits zipped. A small
keyword index (a few KB) tells the system which zip is relevant to a
question. nothing is unzipped permanently. When you ask something, only
the matching zip(s) get read into memory, their text is handed to a small
local language model as context, and the model answers from that. The
model itself is small and the "knowledge" stays compressed on disk
until it's actually needed.


## How it stays tiny

 **The model** is a small quantized instruct model ~350MB–1GB on disk
  
  
  **The knowledge base** stays compressed. `ingest.py` reads each zip
  once, in memory, to build a lightweight word-frequency index (a JSON
  file, typically a few KB even for a large knowledge base, then
  discards the extracted text. Nothing is left unzipped on disk.

**At query time**, only the top 1–2 matching zips are opened (still in
  memory, via Python's `zipfile`, never extracted to disk) and their text
  becomes the model's context for that one answer.



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

- **`src/fetch_bulk.py`** — streams Wikipedia's real bulk export dumps (`enwiki-latest-pages-articles-multistream*.bz2`) and pulls full article text, cleaned down from wikitext, instead of per-article API calls. This is the one for "as much as possible, fast." Articles are taken in order (not randomly sampled) until the target is hit, and each one is capped at ~3000 characters to keep total size manageable. Saves a checkpoint to `knowledge/` every 50,000 articles collected, so a dropped connection or interrupted run doesn't lose progress — whatever's been collected so far is already written out as real, usable zips.
  ```
  python3 src/fetch_bulk.py --target 250000
  ```

Either way, re-run `python3 src/ingest.py` after adding new zips.
