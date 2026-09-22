"""Pull History articles from Wikipedia's public API and package them into
zip shards in knowledge/ — a few hundred articles per zip, so each zip stays
small (~1MB) while the whole set can grow into the thousands.

Uses only the Python standard library (urllib). Must be run somewhere with
real internet access to en.wikipedia.org — this will NOT work from a
sandboxed/proxied environment that blocks that domain.

Usage:
    python3 src/fetch_history.py                  # ~500 articles, a good first batch
    python3 src/fetch_history.py --limit 2000      # pull more
    python3 src/fetch_history.py --shard-size 300  # articles per zip (default 250)
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

API_URL = "https://en.wikipedia.org/w/api.php"
REST_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
HEADERS = {"User-Agent": "tiny-rag/1.0 (personal offline knowledge base project)"}

BASE = os.path.join(os.path.dirname(__file__), "..")
KNOWLEDGE_DIR = os.path.join(BASE, "knowledge")

ROOT_CATEGORY = "Category:History"


def api_get(params, max_retries=6):
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                print(f"  ! HTTP {e.code}, backing off {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                backoff = min(backoff * 2, 30)
                continue
            raise
    raise RuntimeError(f"Gave up after {max_retries} retries on {url}")


def collect_titles(root_category, limit, max_depth=2):
    """Breadth-first walk of the category tree, collecting article titles."""
    titles = []
    seen_cats = set()
    queue = [(root_category, 0)]

    while queue and len(titles) < limit:
        cat, depth = queue.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        time.sleep(0.5)

        cmcontinue = None
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cat,
                "cmlimit": "500",
                "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            data = api_get(params)
            members = data.get("query", {}).get("categorymembers", [])
            for m in members:
                ns = m.get("ns")
                title = m["title"]
                if ns == 0:  # article
                    titles.append(title)
                elif ns == 14 and depth < max_depth:  # subcategory
                    queue.append((title, depth + 1))
            cont = data.get("continue", {}).get("cmcontinue")
            if not cont or len(titles) >= limit:
                break
            cmcontinue = cont
            time.sleep(0.5)

        print(f"  scanned {cat} (depth {depth}) — {len(titles)} article titles so far")

    # de-dupe, preserve order
    seen = set()
    out = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:limit]


def fetch_one_summary(title, max_retries=5):
    """Fetch one article's plain-text extract via the REST summary API —
    one request per title, since the legacy action=query&prop=extracts
    endpoint only reliably returns content for the first title when you
    batch several together (a known MediaWiki quirk)."""
    url = REST_SUMMARY_URL.format(urllib.parse.quote(title.replace(" ", "_")))
    req = urllib.request.Request(url, headers=HEADERS)
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
                return data.get("extract", "")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ""  # disambiguation/missing page — skip quietly
            if e.code == 429 or e.code >= 500:
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                time.sleep(wait)
                backoff = min(backoff * 2, 20)
                continue
            return ""
        except urllib.error.URLError:
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)
    return ""


def fetch_extracts(titles):
    """Fetch a plain-text extract for every title, one request at a time."""
    results = {}
    for i, title in enumerate(titles, 1):
        extract = fetch_one_summary(title)
        if extract:
            results[title] = extract
        if i % 10 == 0 or i == len(titles):
            print(f"  fetched {len(results)}/{i} (of {len(titles)}) extracts", end="\r")
        time.sleep(0.3)
    print()
    return results


def safe_filename(title):
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()[:80]


def write_shards(extracts, shard_size, shard_prefix="history"):
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    items = list(extracts.items())
    shard_num = 0
    for i in range(0, len(items), shard_size):
        shard_num += 1
        shard_items = items[i:i + shard_size]
        zip_path = os.path.join(KNOWLEDGE_DIR, f"{shard_prefix}-{shard_num:03d}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for title, text in shard_items:
                fname = safe_filename(title) + ".txt"
                zf.writestr(fname, text)
        size_kb = os.path.getsize(zip_path) / 1024
        print(f"  wrote {zip_path} ({len(shard_items)} articles, {size_kb:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="max articles to pull")
    ap.add_argument("--shard-size", type=int, default=250, help="articles per zip")
    ap.add_argument("--category", default=ROOT_CATEGORY, help="root category to walk")
    args = ap.parse_args()

    print(f"Collecting article titles under {args.category} (up to {args.limit})...")
    titles = collect_titles(args.category, args.limit)
    print(f"Found {len(titles)} article titles.\n")

    print("Fetching plain-text extracts...")
    extracts = fetch_extracts(titles)
    print(f"\nGot extracts for {len(extracts)} articles.\n")

    print("Writing zip shards into knowledge/...")
    write_shards(extracts, args.shard_size)

    print("\nDone. Run `python3 src/ingest.py` next to index the new zips.")


if __name__ == "__main__":
    main()
