"""Shared helpers: tokenizing and reading text out of zip files in memory
(nothing gets permanently extracted to disk)."""

import os
import re
import zipfile

TOKEN_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


def read_zip_text(zip_path, max_chars=None):
    """Read all .txt/.md content out of a zip archive, entirely in memory."""
    parts = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".txt", ".md")):
                with zf.open(name) as f:
                    try:
                        text = f.read().decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    parts.append(f"## {os.path.basename(name)}\n{text}")
    full = "\n\n".join(parts)
    if max_chars:
        full = full[:max_chars]
    return full
