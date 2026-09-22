"""Pull a large number of Wikipedia articles from the real bulk export dumps
(Wikipedia discontinued the old standalone "abstracts" dump — this now reads
the pages-articles-multistream XML dumps instead: full wikitext, which gets
stripped down to plain text here with lightweight regex cleanup, not a full
wikitext parser). One continuous streaming download+parse, not one API call
per article — that's what makes hundreds of thousands of articles feasible
overnight.

Uses only the Python standard library (urllib, bz2, xml.etree, zipfile, re).
Must be run somewhere with real internet access — this will NOT work from a
sandboxed/proxied environment that blocks dumps.wikimedia.org.

Usage:
    python3 src/fetch_bulk.py --target 250000
    python3 src/fetch_bulk.py --target 2000000 --shard-size 1000

Notes:
    - Articles are taken, in order, from the lower page-ID range of the
      dump (the first several multistream part files) until the target
      count is reached — not a random sample across all of Wikipedia.
      That's a real scope trade-off: downloading and scanning ALL 27 part
      files just to sample evenly would take far longer than "overnight."
      The part files below cover roughly the first ~2 million page IDs,
      comfortably more than enough to reach typical targets after
      redirects/stubs are skipped.
    - Wikitext-to-plaintext cleanup here is regex-based, not a full parser
      (no external dependencies). It strips templates, refs, tables, links,
      and formatting reasonably well, but won't be perfect on every article
      — some residual markup may slip through occasionally.
    - Full articles are kept (no fixed truncation) except for a generous
      50,000-character safety ceiling that only ever kicks in for a
      handful of unusually massive pages. Overall disk usage is bounded
      separately by --max-gb, not by per-article length.
"""

import argparse
import bz2
import os
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

BASE = os.path.join(os.path.dirname(__file__), "..")
KNOWLEDGE_DIR = os.path.join(BASE, "knowledge")

HEADERS = {"User-Agent": "tiny-rag/1.0 (personal offline knowledge base project)"}
DUMP_BASE = "https://dumps.wikimedia.org/enwiki/latest/"

# Lower page-ID range multistream part files, in order. Each covers a
# contiguous block of page IDs. More than enough real articles live in
# just the first few of these to reach typical targets.
PART_FILES = [
    "enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2",
    "enwiki-latest-pages-articles-multistream2.xml-p41243p151573.bz2",
    "enwiki-latest-pages-articles-multistream3.xml-p151574p311329.bz2",
    "enwiki-latest-pages-articles-multistream4.xml-p311330p558391.bz2",
    "enwiki-latest-pages-articles-multistream5.xml-p558392p958045.bz2",
    "enwiki-latest-pages-articles-multistream6.xml-p958046p1483661.bz2",
    "enwiki-latest-pages-articles-multistream7.xml-p1483662p2134111.bz2",
]

# Not a target length — a safety ceiling. Virtually all articles land far
# below this; it only guards against a handful of unusually massive pages
# (long list articles, major topics) eating a disproportionate share of the
# --max-gb budget in one shot.
MAX_EXTRACT_CHARS = 50000


def localname(tag):
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def strip_templates(text):
    """Remove {{...}} templates, handling nesting (regex alone can't)."""
    out = []
    depth = 0
    i, n = 0, len(text)
    while i < n:
        if text[i:i + 2] == "{{":
            depth += 1
            i += 2
            continue
        if text[i:i + 2] == "}}" and depth > 0:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


_RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_RE_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_RE_FILELINK = re.compile(r"\[\[(File|Image):[^\]]*\]\]", re.IGNORECASE)
_RE_PIPED_LINK = re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]")
_RE_PLAIN_LINK = re.compile(r"\[\[([^\]]*)\]\]")
_RE_EXTLINK = re.compile(r"\[https?://[^\s\]]*\s*([^\]]*)\]")
_RE_BOLD_ITALIC = re.compile(r"'{2,5}")
_RE_HEADER = re.compile(r"^=+\s*(.*?)\s*=+$", re.MULTILINE)
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_WS = re.compile(r"[ \t]+")
_RE_BLANKLINES = re.compile(r"\n{3,}")


def wikitext_to_plain(text):
    if not text:
        return ""
    text = _RE_COMMENT.sub("", text)
    text = _RE_REF.sub("", text)
    text = _RE_TABLE.sub("", text)
    text = strip_templates(text)
    text = _RE_FILELINK.sub("", text)
    text = _RE_PIPED_LINK.sub(r"\1", text)
    text = _RE_PLAIN_LINK.sub(r"\1", text)
    text = _RE_EXTLINK.sub(r"\1", text)
    text = _RE_BOLD_ITALIC.sub("", text)
    text = _RE_HEADER.sub(r"\1", text)
    text = _RE_HTML_TAG.sub("", text)
    text = _RE_WS.sub(" ", text)
    text = _RE_BLANKLINES.sub("\n\n", text)
    return text.strip()


def open_bz2_stream(url):
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=60)
    return bz2.BZ2File(resp)


def iter_articles():
    """Stream (title, plain_text) pairs out of the multistream dump files,
    in order, skipping redirects and disambiguation-style near-empty pages."""
    for fname in PART_FILES:
        url = DUMP_BASE + fname
        print(f"  reading {fname}")
        try:
            stream = open_bz2_stream(url)
        except urllib.error.HTTPError as e:
            print(f"  ! could not open {fname}: {e} — skipping")
            continue

        for event, elem in ET.iterparse(stream, events=("end",)):
            if localname(elem.tag) != "page":
                continue

            title = None
            raw_text = None
            ns = None
            for child in elem.iter():
                name = localname(child.tag)
                if name == "title" and title is None:
                    title = child.text
                elif name == "ns" and ns is None:
                    ns = child.text
                elif name == "text" and raw_text is None:
                    raw_text = child.text

            elem.clear()  # free memory — critical for a multi-GB stream

            if ns != "0" or not title or not raw_text:
                continue  # only main-namespace articles with real content
            if raw_text.lstrip().upper().startswith("#REDIRECT"):
                continue

            plain = wikitext_to_plain(raw_text)[:MAX_EXTRACT_CHARS]
            if len(plain) < 200:
                continue  # too short to be useful (stub/disambiguation-ish)

            yield title, plain


CHECKPOINT_EVERY = 50000  # articles scanned between safety saves

# Text generally shrinks to somewhere around 40-55% of its original size
# under zip's deflate compression. We use 0.55 (i.e. assume compression is
# LESS favorable than typical) so the raw-byte budget below stays a safe,
# conservative estimate of the real on-disk zip size rather than an
# optimistic one.
ASSUMED_COMPRESSION_RATIO = 0.55


def collect_articles(target, max_raw_bytes=None, checkpoint_cb=None):
    """Take articles in order until `target` is reached, the source runs
    out, or `max_raw_bytes` of raw text has been collected (whichever comes
    first). Returns whatever was collected even if interrupted — nothing is
    lost on a dropped connection or Ctrl+C."""
    collected = []
    scanned = 0
    raw_bytes = 0
    try:
        for title, plain in iter_articles():
            scanned += 1
            collected.append((title, plain))
            raw_bytes += len(plain.encode("utf-8"))
            if scanned % 2000 == 0:
                print(f"  collected {len(collected):,} articles "
                      f"(scanned {scanned:,} candidates, "
                      f"~{raw_bytes / (1024*1024):.0f} MB raw)…", end="\r")
            if checkpoint_cb and len(collected) % CHECKPOINT_EVERY == 0:
                print(f"\n  checkpoint at {len(collected):,} — saving progress to knowledge/...")
                checkpoint_cb(collected)
            if len(collected) >= target:
                break
            if max_raw_bytes and raw_bytes >= max_raw_bytes:
                print(f"\n  reached the size cap after {len(collected):,} articles "
                      f"(~{raw_bytes / (1024*1024*1024):.2f} GB raw) — stopping here.")
                break
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        print(f"\n  ! connection interrupted after collecting {len(collected):,} articles: {e}")
        print("  Saving what was collected so far — nothing is lost.")
    except KeyboardInterrupt:
        print(f"\n  Stopped by user after collecting {len(collected):,} articles.")
        print("  Saving what was collected so far.")
    print()
    return collected


def safe_filename(title):
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()[:80]


def write_shards(items, shard_size, shard_prefix="bulk"):
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    shard_num = 0
    for i in range(0, len(items), shard_size):
        shard_num += 1
        shard_items = items[i:i + shard_size]
        zip_path = os.path.join(KNOWLEDGE_DIR, f"{shard_prefix}-{shard_num:04d}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_LZMA) as zf:
            for title, text in shard_items:
                fname = safe_filename(title) + ".txt"
                zf.writestr(fname, text)
        if shard_num % 25 == 0 or i + shard_size >= len(items):
            print(f"  wrote {shard_num} shard(s) so far…", end="\r")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=None,
                     help="stop after this many articles. Omit to run with no article-count "
                          "limit — collection then stops only at --max-gb, when the dump runs "
                          "out, or when you Ctrl+C.")
    ap.add_argument("--shard-size", type=int, default=1000, help="articles per zip")
    ap.add_argument("--max-gb", type=float, default=7.5,
                     help="stop once the estimated compressed output would exceed this many GB "
                          "(safety cap independent of --target; default 7.5)")
    args = ap.parse_args()

    target = args.target if args.target is not None else float("inf")

    max_raw_bytes = None
    if args.max_gb:
        max_raw_bytes = int((args.max_gb * 1024 * 1024 * 1024) / ASSUMED_COMPRESSION_RATIO)

    if args.target is not None:
        print(f"Collecting up to {args.target:,} articles from Wikipedia's bulk export dump "
              f"(capped at ~{args.max_gb:.1f} GB of compressed output)...")
    else:
        print(f"Collecting articles from Wikipedia's bulk export dump with no article-count "
              f"limit (capped at ~{args.max_gb:.1f} GB of compressed output)...")
    print("(this streams and discards as it goes — no multi-GB file is kept on disk)")
    print(f"(progress is saved to knowledge/ every {CHECKPOINT_EVERY:,} articles collected, "
          f"so a dropped connection — or you hitting Ctrl+C — loses nothing)")

    def checkpoint(collected_so_far):
        write_shards(collected_so_far, args.shard_size)

    collected = collect_articles(target, max_raw_bytes=max_raw_bytes, checkpoint_cb=checkpoint)
    print(f"\nGot {len(collected):,} articles.\n")

    print("Writing final zip shards into knowledge/...")
    write_shards(collected, args.shard_size)

    total_bytes = sum(len(t.encode("utf-8")) for _, t in collected)
    print(f"\nRaw text: ~{total_bytes / (1024*1024):.1f} MB before compression.")

    actual_knowledge_bytes = sum(
        os.path.getsize(os.path.join(KNOWLEDGE_DIR, f))
        for f in os.listdir(KNOWLEDGE_DIR)
        if os.path.isfile(os.path.join(KNOWLEDGE_DIR, f))
    )
    actual_gb = actual_knowledge_bytes / (1024 * 1024 * 1024)
    print(f"knowledge/ is now ~{actual_gb:.2f} GB on disk (cap was {args.max_gb:.1f} GB).")
    if actual_gb > args.max_gb:
        print("  ! this is over the cap — compression on this batch came out less favorable "
              "than estimated. Consider removing the last bulk-*.zip shard if you need to be "
              "strictly under the limit.")

    if args.target is not None and len(collected) < args.target and (not max_raw_bytes or total_bytes < max_raw_bytes):
        print(f"(Got {len(collected):,} of the {args.target:,} requested — the configured "
              f"part files ran out. Add more filenames to PART_FILES for a larger run.)")
    elif args.target is None and (not max_raw_bytes or total_bytes < max_raw_bytes):
        print("(The configured part files ran out before hitting the size cap. Add more "
              "filenames to PART_FILES for a longer run.)")
    print("Done (or safely stopped). Run `python3 src/ingest.py` next to index the zips.")


if __name__ == "__main__":
    main()
