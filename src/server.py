"""A tiny local web UI for WikiZip. Zero external dependencies — Python's
built-in http.server on the backend, plain HTML/CSS/JS on the frontend.

Reuses the same retrieval + Ollama logic as ask.py (imported directly, not
duplicated), so both the CLI and the web UI stay in sync.

Usage:
    python3 src/server.py
    (then open http://localhost:8765 in a browser)
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(__file__))
import ask  # noqa: E402 — reuse rank(), build_context(), ask_ollama(), load_index()

BASE = os.path.join(os.path.dirname(__file__), "..")
WEB_DIR = os.path.join(BASE, "web")
PORT = 8765


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet; comment this out if you want request logs

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(os.path.join(WEB_DIR, "index.html"), "text/html")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/ask":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Bad request body."}, status=400)
            return

        question = (body.get("question") or "").strip()
        if not question:
            self._send_json({"error": "Ask something first."}, status=400)
            return

        index = self.server.get_index()
        if index is None:
            self._send_json({
                "error": "No index found — run `python3 src/ingest.py` first."
            })
            return

        try:
            ranked = [(s, f) for s, f in ask.rank(question, index) if s > 0][:ask.TOP_K]
            top_files = [f for _, f in ranked]
            context = ask.build_context(top_files) if top_files else ""

            system = (
                "You are a helpful assistant. Answer using only the provided "
                "context. If the answer isn't in the context, say you don't know."
            )
            prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
            answer = ask.ask_ollama(prompt)

            self._send_json({"sources": top_files, "answer": answer})
        except Exception as e:
            self._send_json({"error": f"Something went wrong: {e}"}, status=500)

    def _serve_file(self, path, content_type):
        if not os.path.exists(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class WikiZipServer(ThreadingHTTPServer):
    """Loads the index once at startup and caches it in memory, instead of
    re-reading (potentially a large) index.json off disk on every request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = None
        self._index_mtime = None

    def get_index(self):
        if not os.path.exists(ask.INDEX_PATH):
            return None
        mtime = os.path.getmtime(ask.INDEX_PATH)
        if self._index is None or mtime != self._index_mtime:
            print("Loading index.json into memory...")
            self._index = ask.load_index()
            self._index_mtime = mtime
            print("Index loaded.")
        return self._index


def main():
    server = WikiZipServer(("localhost", PORT), Handler)
    print(f"WikiZip running at http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
