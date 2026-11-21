#!/usr/bin/env bash
# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
# Download (or resume) a Zenodo record used by the paper with zenodo_fetch.py (chunked, md5-verified).
# Usage: bash download_zenodo.sh wabad      -> the directory `wabad_src` of paper/config.yaml
#        (WABAD, Zenodo record 17293588: Metadata.csv, Pooled annotations.csv and one zip per site;
#        prepare_wabad.py reads the zips directly, so nothing is unpacked here.)
# Re-running is safe: finished chunks are skipped.
set -euo pipefail
NAME=${1:?usage: $0 wabad}
PAPER="$(cd "$(dirname "$0")/../.." && pwd)"
case $NAME in
  wabad) REC=17293588
         DEST=$(PYTHONPATH="$PAPER" python -c "from common import config; print(config.WABAD_SRC)") ;;
  *) echo "unknown dataset: $NAME"; exit 1 ;;
esac
echo "Zenodo record $REC -> $DEST ($(( $(python "$PAPER/data_prep/download/zenodo_fetch.py" --size "$REC") / 10**9 )) GB)"
for try in $(seq 1 50); do
  python "$PAPER/data_prep/download/zenodo_fetch.py" "$REC" "$DEST" && exit 0
  echo "attempt $try failed, retrying in 120 s"; sleep 120
done
exit 1
