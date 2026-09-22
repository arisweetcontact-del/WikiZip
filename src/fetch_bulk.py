"""Pull a large number of Wikipedia articles (short lead-paragraph abstracts,
not full text) from Wikipedia's bulk abstracts dump — one big download and a
streaming parse, instead of one API call per article. This is what makes
hundreds of thousands of articles feasible overnight; fetch_history.py's
one-request-per-article approach would take days at that scale.

Uses only the Python standard library (urllib, gzip, xml.etree, zipfile).
Must be run somewhere with real internet access — this will NOT work from a
sandboxed/proxied environment that blocks dumps.wikimedia.org.

Usage:
    python3 src/fetch_bulk.py --target 250000
    python3 src/fetch_bulk.py --target 2000000 --shard-size 1000

Notes:
    - Sampling is uniform-random across the whole dump (reservoir sampling),
      not just the first N encountered — so you get broad topic coverage,
      not an alphabetical slice.
    - Abstracts are short (typically a sentence or two). At 200k-2M articles
      this stays well under 1GB even before zip compression — if you want
      denser per-article content instead of more articles, fetch_history.py
      (full extracts, but far slower per-article) is the better tool.
"""

import argparse
import gzip
import os
import random
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

BASE = os.path.join(os.path.dirname(__file__), "..")
KNOWLEDGE_DIR = os.path.join(BASE, "knowledge")

HEADERS = {"User-Agent": "tiny-rag/1.0 (personal offline knowledge base project)"}
DUMP_BASE = "https://dumps.wikimedia.org/enwiki/latest/"
COMBINED_NAME = "enwiki-latest-abstract.xml.gz"
SPLIT_NAME_FMT = "enwiki-latest-abstract{}.xml.gz"
MAX_SPLIT_FILES = 30


def open_gz_stream(url):
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=60)
    return gzip.GzipFile(fileobj=resp)


def dump_urls():
    """Yield the dump file URL(s) to read, trying the single combined file
    first, then falling back to the numbered split files if that 404s."""
    combined_url = DUMP_BASE + COMBINED_NAME
    try:
        req = urllib.request.Request(combined_url, method="HEAD", headers=HEADERS)
        urllib.request.urlopen(req, timeout=30)
        yield combined_url
        return
    except urllib.error.HTTPError:
        pass

    print("  (combined dump not found — using split files instead)")
    found_any = False
    for n in range(1, MAX_SPLIT_FILES + 1):
        url = DUMP_BASE + SPLIT_NAME_FMT.format(n)
        try:
            req = urllib.request.Request(url, method="HEAD", headers=HEADERS)
            urllib.request.urlopen(req, timeout=30)
            yield url
            found_any = True
        except urllib.error.HTTPError:
            if found_any:
                return  # ran past the last split file
            continue


def iter_docs():
    """Stream (title, abstract) pairs out of the dump, across however many
    files it's split into, without ever holding the whole thing in memory."""
    for url in dump_urls():
        print(f"  reading {url}")
        stream = open_gz_stream(url)
        # iterparse lets us process <doc> elements one at a time and discard
        # them, instead of parsing the whole (multi-GB uncompressed) tree.
        for event, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag == "doc":
                title_el = elem.find("title")
                abstract_el = elem.find("abstract")
                title = (title_el.text or "").strip() if title_el is not None else ""
                abstract = (abstract_el.text or "").strip() if abstract_el is not None else ""
                # dump titles are prefixed "Wikipedia: "
                if title.startswith("Wikipedia: "):
                    title = title[len("Wikipedia: "):]
                if title and abstract:
                    yield title, abstract
                elem.clear()  # free memory — critical for a multi-GB stream


CHECKPOINT_EVERY = 50000  # articles scanned between safety saves


def reservoir_sample(target, checkpoint_cb=None):
    """Single-pass uniform random sample of `target` (title, abstract) pairs
    from a stream of unknown length.

    If the connection drops or anything else goes wrong mid-stream, this
    returns whatever it has so far instead of raising — call sites should
    still write that out as real, usable zips rather than losing the run."""
    sample = []
    i = -1
    try:
        for i, item in enumerate(iter_docs()):
            if i < target:
                sample.append(item)
            else:
                j = random.randint(0, i)
                if j < target:
                    sample[j] = item
            if (i + 1) % 5000 == 0:
                print(f"  scanned {i + 1:,} articles, sample has {len(sample):,}…", end="\r")
            if checkpoint_cb and (i + 1) % CHECKPOINT_EVERY == 0:
                print(f"\n  checkpoint at {i + 1:,} scanned — saving progress to knowledge/...")
                checkpoint_cb(sample)
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        print(f"\n  ! connection interrupted after scanning {i + 1:,} articles: {e}")
        print(f"  Saving the {len(sample):,} articles collected so far — nothing is lost.")
        print("  Re-run this script later to start a fresh (larger/different) sample.")
    except KeyboardInterrupt:
        print(f"\n  Stopped by user after scanning {i + 1:,} articles.")
        print(f"  Saving the {len(sample):,} articles collected so far.")
    print()
    return sample


def safe_filename(title):
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()[:80]


def write_shards(items, shard_size, shard_prefix="bulk"):
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    shard_num = 0
    for i in range(0, len(items), shard_size):
        shard_num += 1
        shard_items = items[i:i + shard_size]
        zip_path = os.path.join(KNOWLEDGE_DIR, f"{shard_prefix}-{shard_num:04d}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for title, abstract in shard_items:
                fname = safe_filename(title) + ".txt"
                zf.writestr(fname, abstract)
        if shard_num % 25 == 0 or i + shard_size >= len(items):
            print(f"  wrote {shard_num} shard(s) so far…", end="\r")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=250000, help="number of articles to sample")
    ap.add_argument("--shard-size", type=int, default=1000, help="articles per zip")
    args = ap.parse_args()

    random.seed()

    print(f"Sampling {args.target:,} articles from Wikipedia's abstracts dump...")
    print("(this streams and discards as it goes — no multi-GB file is kept on disk)")
    print(f"(progress is saved to knowledge/ every {CHECKPOINT_EVERY:,} articles scanned, "
          f"so a dropped connection loses nothing)")

    def checkpoint(sample_so_far):
        write_shards(sample_so_far, args.shard_size)

    sample = reservoir_sample(args.target, checkpoint_cb=checkpoint)
    print(f"\nGot {len(sample):,} articles.\n")

    print("Writing final zip shards into knowledge/...")
    write_shards(sample, args.shard_size)

    total_bytes = sum(len(a.encode("utf-8")) for _, a in sample)
    print(f"\nRaw text: ~{total_bytes / (1024*1024):.1f} MB before compression.")
    if len(sample) < args.target:
        print(f"(Got {len(sample):,} of the {args.target:,} requested — re-run to try for "
              f"a fuller sample later.)")
    print("Done (or safely stopped). Run `python3 src/ingest.py` next to index the zips.")


if __name__ == "__main__":
    main()
