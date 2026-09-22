"""Shared helpers: tokenizing and reading text out of zip files in memory
(nothing gets permanently extracted to disk)."""

import os
import re
import zipfile

TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")

# Separator used to build a document key that identifies one article inside
# one zip: "<zip filename>::<member filename>". "::" never appears in either
# half (safe_filename() in fetch scripts strips colons), so splitting on it
# is unambiguous.
DOC_KEY_SEP = "::"


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


def iter_zip_members(zip_path):
    """Yield (member_name, text) for every .txt/.md file in a zip, entirely
    in memory. Used to index or read one article at a time instead of an
    entire zip's contents at once."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".txt", ".md")):
                with zf.open(name) as f:
                    try:
                        text = f.read().decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    yield name, text


def make_doc_key(zip_filename, member_name):
    return f"{zip_filename}{DOC_KEY_SEP}{member_name}"


def split_doc_key(doc_key):
    zip_filename, member_name = doc_key.split(DOC_KEY_SEP, 1)
    return zip_filename, member_name


def read_zip_member_text(zip_path, member_name, max_chars=None):
    """Read the full text of ONE specific file out of a zip, in memory."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as f:
            text = f.read().decode("utf-8", errors="ignore")
    if max_chars:
        text = text[:max_chars]
    return text
