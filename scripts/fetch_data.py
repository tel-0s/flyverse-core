"""Fetch the data this repository does not ship: the MaleCNS flat-connectome files and the third-party
expression / typing tables used by the receptor integration. Driven by flyverse/data/manifest.json.

    python scripts/fetch_data.py --list                      # what the manifest knows, and what is present
    python scripts/fetch_data.py --malecns                   # the four MaleCNS files flyverse reads (3.7 GB) -> $FLYVERSE_DATA
    python scripts/fetch_data.py --external all              # every external source -> data/external/<source>/
    python scripts/fetch_data.py --external ozel2021,davis2020
    python scripts/fetch_data.py --verify                    # re-hash everything present, report mismatches

MaleCNS files go to the directory flyverse.connectome.DATA_DIR reads (env FLYVERSE_DATA, default the project's
Windows path; pass --data-dir to override); after fetching them, `python -c "from flyverse import connectome;
connectome.load(rebuild=True)"` builds cache/. External files go under data/external/ (git-ignored: they are other
people's data, redistributed under their own licences -- the manifest records the citation and licence of each).

Downloads stream to a .part file and are renamed only after the SHA-256 in the manifest matches (when one is
recorded); files already present with a matching hash are skipped. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST = os.path.join(ROOT, "flyverse", "data", "manifest.json")
EXTERNAL = os.path.join(ROOT, "data", "external")


def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def fetch(url: str, dest: str, expect: str | None, force: bool = False) -> str:
    """Download url to dest; returns 'present' | 'fetched' | 'mismatch'."""
    if os.path.exists(dest) and not force:
        if not expect or sha256(dest) == expect:
            return "present"
        print(f"  {os.path.basename(dest)}: hash differs from the manifest, refetching")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "flyverse-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0); done = 0; last = -1
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b); done += len(b)
            pct = int(100 * done / total) if total else -1
            if pct != last and pct % 10 == 0:
                print(f"  {os.path.basename(dest)}: {done / 1e6:,.0f} MB" + (f" ({pct}%)" if total else ""), flush=True); last = pct
    if expect and sha256(part) != expect:
        os.replace(part, dest + ".bad")
        return "mismatch"
    os.replace(part, dest)
    return "fetched"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--malecns", action="store_true", help="fetch the MaleCNS flat-connectome files flyverse reads")
    ap.add_argument("--external", default="", help="'all' or comma-separated source keys from the manifest")
    ap.add_argument("--data-dir", default=None, help="where the MaleCNS files go (default: flyverse.connectome.DATA_DIR)")
    ap.add_argument("--verify", action="store_true", help="hash every present file against the manifest")
    ap.add_argument("--force", action="store_true", help="refetch even when present")
    args = ap.parse_args()
    m = json.load(open(MANIFEST, encoding="utf-8"))
    if args.data_dir:
        data_dir = args.data_dir
    else:
        sys.path.insert(0, ROOT)
        from flyverse import connectome
        data_dir = str(connectome.DATA_DIR)
    groups: list[tuple[str, str, list[dict]]] = [("malecns", data_dir, m["malecns"]["files"])]
    for key, src in m["external"].items():
        groups.append((key, os.path.join(EXTERNAL, key), src["files"]))

    if args.list or not (args.malecns or args.external or args.verify):
        for key, base, files in groups:
            info = m["external"].get(key) or m["malecns"]
            print(f"[{key}] {info.get('citation', '')}  licence: {info.get('licence', '?')}  -> {base}")
            for f in files:
                p = os.path.join(base, f["path"]); st = "present" if os.path.exists(p) else "missing"
                print(f"   {st:8s} {f['path']}  ({f.get('size_mb', '?')} MB)")
            if not files:
                print("   (no files recorded yet)")
        return 0

    wanted = set()
    if args.malecns:
        wanted.add("malecns")
    if args.external:
        wanted |= set(m["external"]) if args.external == "all" else set(args.external.split(","))
    bad = 0
    for key, base, files in groups:
        if args.verify:
            for f in files:
                p = os.path.join(base, f["path"])
                if os.path.exists(p):
                    ok = (not f.get("sha256")) or sha256(p) == f["sha256"]
                    print(f"{'ok ' if ok else 'BAD'} {key}/{f['path']}"); bad += (not ok)
            continue
        if key not in wanted:
            continue
        print(f"[{key}] -> {base}")
        for f in files:
            p = os.path.join(base, f["path"])
            try:
                st = fetch(f["url"], p, f.get("sha256"), force=args.force)
            except Exception as e:                                    # keep going; report at the end
                st = f"FAILED ({e})"; bad += 1
            if st == "mismatch":
                bad += 1
            print(f"  {st:8s} {f['path']}")
    if bad:
        print(f"{bad} file(s) failed or mismatched"); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
