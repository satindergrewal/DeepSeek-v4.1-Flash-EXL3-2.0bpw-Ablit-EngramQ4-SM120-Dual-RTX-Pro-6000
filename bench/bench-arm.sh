#!/usr/bin/env bash
# Four-cell bench matrix, tagged, so every tuning arm is measured identically.
#   ./bench-arm.sh <tag>
set -uo pipefail
TAG="${1:?usage: bench-arm.sh <tag>}"
cd /mnt/nvme0/bigmodels/dsv41-exl3-2.0bpw/recipe
set -a; . ../serve.env; set +a
export API_KEY PORT="${PORT:-8000}" SERVED_MODEL="${SERVED_MODEL:-deepseek-v4.1-flash}"
OUT=/mnt/nvme0/bigmodels/dsv41-exl3-2.0bpw/bench-$TAG.log
echo "### bench $TAG start $(date -Is)" | tee -a "$OUT"
for ctx in 2000 46000; do
  for conc in 1 4; do
    echo "--- CTX=$ctx CONC=$conc" | tee -a "$OUT"
    CTX="$ctx" CONC="$conc" timeout 1800 python3 bench/bench-sbs.py 2>&1 | tee -a "$OUT"
  done
done
echo "### bench $TAG done $(date -Is)" | tee -a "$OUT"
