#!/usr/bin/env python3
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""Parallel, chunk-resumable download of Zenodo record files, md5-verified.

Zenodo throttles each connection (~150 kB/s observed), so each file is fetched
as 16 MB byte ranges over several connections; finished chunk ids are appended
to <file>.chunks so a restart only fetches what is missing.

Usage: python zenodo_fetch.py RECORD_ID DEST [FILE ...]          (no FILE: every file of the record)
       python zenodo_fetch.py --size RECORD_ID [FILE ...]          (print total bytes)
"""
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import urllib.request

CHUNK = 16 << 20
WORKERS = 8


def record_files(rec, names):
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}", timeout=60) as r:
        files = {f["key"]: f for f in json.load(r)["files"]}
    return [files[n] for n in names] if names else list(files.values())


def get_range(url, out, lo, hi):
    for _ in range(30):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
                if r.status != 206 or len(data) != hi - lo + 1:
                    raise OSError(f"bad range reply: status {r.status}, {len(data)} bytes")
            with open(out, "r+b") as fh:
                fh.seek(lo)
                fh.write(data)
            return
        except Exception as e:
            print(f"{os.path.basename(out)} bytes {lo}-{hi}: {e!r}, retrying", flush=True)
            time.sleep(30)
    raise RuntimeError(f"giving up on {out} bytes {lo}-{hi}")


def fetch(f, dest):
    out, size, url = os.path.join(dest, f["key"]), f["size"], f["links"]["self"]
    state = out + ".chunks"
    done = set(map(int, open(state).read().split())) if os.path.exists(state) else set()
    n_chunks = -(-size // CHUNK)
    todo = [i for i in range(n_chunks) if i not in done]
    os.makedirs(dest, exist_ok=True)
    open(out, "ab").close()
    with cf.ThreadPoolExecutor(WORKERS) as ex, open(state, "a") as log:
        futs = {ex.submit(get_range, url, out, i * CHUNK, min(size, (i + 1) * CHUNK) - 1): i for i in todo}
        for fut in cf.as_completed(futs):
            fut.result()
            done.add(futs[fut])
            log.write(f"{futs[fut]}\n")
            log.flush()
            print(f"{f['key']}: {len(done)}/{n_chunks} chunks", flush=True)
    md5 = hashlib.md5()
    with open(out, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            md5.update(block)
    if os.path.getsize(out) != size or f"md5:{md5.hexdigest()}" != f["checksum"]:
        sys.exit(f"{f['key']}: size/md5 mismatch (expected {size} {f['checksum']})")
    print(f"{f['key']}: md5 ok", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--size":
        print(sum(f["size"] for f in record_files(sys.argv[2], sys.argv[3:])))
    else:
        for f in record_files(sys.argv[1], sys.argv[3:]):
            fetch(f, sys.argv[2])
