#!/usr/bin/env bash
# Concurrency ladder on the production serve: warm-cache 2-phase (coresident.py).
# Usage: conc-sweep.sh   (env: CTXS="46000 46000 46000 46000 2000" NS="4 8 12 16 16")
set -uo pipefail
cd /mnt/nvme0/bigmodels/dsv41-exl3-2.0bpw/recipe
set -a; . ../serve.env; set +a
export PORT="${PORT:-8000}"
NS=${NS:-"4 8 12 16 16"}
CTXS=${CTXS:-"46000 46000 46000 46000 2000"}
OUT=/mnt/nvme0/bigmodels/dsv41-engram-q4/conc-sweep-$(date +%m%d-%H%M).log
echo "### conc-sweep start $(date -Is)" | tee "$OUT"
i=1
for N in $NS; do
  CTX=$(echo $CTXS | cut -d" " -f$i)
  echo "--- N=$N CTX=$CTX $(date -Is)" | tee -a "$OUT"
  N=$N CTX=$CTX DECODE_TOKENS=400 timeout 2400 python3 /mnt/nvme0/bigmodels/dsv41-engram-q4/tools/coresident.py 2>&1 | tee -a "$OUT"
  i=$((i+1))
done
echo "### conc-sweep done $(date -Is)" | tee -a "$OUT"
