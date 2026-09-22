#WikiZip ***may be inaccurate due to data holes***

A tiny local question-answering setup that runs comfortably on 8GB of RAM.

**The idea:**  reference material from Wikipedia sits zipped, article by article. A small keyword index tells the system which article is relevant to a
question. nothing is unzipped permanently. When you ask something, only
the matching article(s) get read into memory straight out of whichever
zip holds them their text is handed to a small local language model as
context, and the model answers from that. The model itself is small and
the "knowledge" stays compressed on disk until it's actually needed.

The database contains 1,500,000 articles of varying topics standing at about 4GB and is compiled through data dump API's via the wiki. 




## Setup

1. **Install Ollama** (runs the local model — no Python packaging involved):
   - https://ollama.com/download, or `brew install ollama`
2. **Pull a small model:**
   ```
   ollama pull qwen2.5:1.5b
   ```

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


