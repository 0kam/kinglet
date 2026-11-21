#!/usr/bin/env bash
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# Launch (or resume) a detached download of one BirdSet subset from Hugging Face into
# <downloads>/birdset_<SUBSET> (downloads is set in paper/config.yaml).
# Usage: bash download_birdset.sh <SUBSET>   e.g. POW, HSN, NBP, NES, PER, SNE, SSW, UHH, XCM, XCL
# Test sets (POW..UHH): test + test_5s shards and all metadata parquet.
# XCM / XCL: everything under the subset dir.
# Re-running is safe: skips if done or already running, otherwise resumes.
set -euo pipefail
S=${1:?usage: $0 SUBSET}
PAPER="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT=$(PYTHONPATH="$PAPER" python -c "from common import config; print(config.DOWNLOADS)")
PY=python
REPO=DBD-research-group/BirdSet REV=data
NAME=birdset_$S DEST=$ROOT/birdset_$S
LOG=$ROOT/logs/$NAME.log PIDF=$ROOT/logs/$NAME.pid
case $S in XC*) PATTERNS="$S/*" ;; *) PATTERNS="$S/*test* $S/*.parquet" ;; esac
export PATTERNS REPO REV DEST
# Keep the Hugging Face xet chunk cache on the download disk.
export HF_XET_CACHE=$ROOT/.hf_xet_cache

if [[ ${2:-} == _run ]]; then  # detached worker
  for try in $(seq 1 50); do
    echo "[$(date -Is)] attempt $try"
    "$PY" - <<'PY' && exit 0
import hashlib, json, os
from huggingface_hub import snapshot_download
d = os.environ["DEST"]
snapshot_download(os.environ["REPO"], repo_type="dataset", revision=os.environ["REV"],
                  allow_patterns=os.environ["PATTERNS"].split(), local_dir=d, max_workers=4)
expected = json.load(open(f"{d}/.expected_files.json"))
bad = [p for p, s in expected if os.path.getsize(f"{d}/{p}") != s]
assert not bad, f"size mismatch: {bad}"
listing = "".join(f"{p}\t{s}\n" for p, s in expected)
with open(f"{d}/.done", "w") as f:
    f.write(listing + f"sha256(file list)\t{hashlib.sha256(listing.encode()).hexdigest()}\n")
print("DONE", len(expected), "files")
PY
    sleep 120
  done
  exit 1
fi

mkdir -p "$ROOT/logs" "$DEST"
[[ -f $DEST/.done ]] && { echo "$NAME already done"; exit 0; }
if [[ -f $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "$NAME already running (pid $(cat "$PIDF"))"; exit 0
fi

# Expected size -> $DEST/.expected_bytes, .expected_files.json
"$PY" - <<'PY'
import json, os
from huggingface_hub import HfApi
from huggingface_hub.utils import filter_repo_objects
files = [f for f in HfApi().list_repo_tree(os.environ["REPO"], repo_type="dataset", revision=os.environ["REV"],
                                           path_in_repo=os.environ["PATTERNS"].split("/")[0], recursive=True)
         if hasattr(f, "size")]
files = list(filter_repo_objects(files, allow_patterns=os.environ["PATTERNS"].split(), key=lambda f: f.path))
d = os.environ["DEST"]
json.dump([(f.path, f.size) for f in files], open(f"{d}/.expected_files.json", "w"))
open(f"{d}/.expected_bytes", "w").write(str(sum(f.size for f in files)))
print(f"{len(files)} files, {sum(f.size for f in files)/1e9:.2f} GB expected")
PY

# Free space must be >= 1.3x the remaining bytes of every unfinished dataset.
need=0
for d in "$ROOT"/*/; do
  [[ -f $d/.expected_bytes && ! -f $d/.done ]] || continue
  left=$(( $(cat "$d/.expected_bytes") - $(du -sB1 "$d" | cut -f1) ))
  (( left > 0 )) && need=$(( need + left ))
done
free=$(( $(df -B1 --output=avail "$ROOT" | tail -1) ))
echo "remaining queued: $((need/10**9)) GB, free: $((free/10**9)) GB"
(( free * 10 >= need * 13 )) || { echo "NOT STARTED: free < 1.3x queued"; exit 1; }

setsid nohup bash "$0" "$S" _run >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "$NAME started, pid $!, log $LOG"
