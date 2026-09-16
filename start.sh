#!/usr/bin/env bash
# Boot the DSV4.1-Flash ablit+EngramQ4 serve on :8000 and wait for ready.
set -uo pipefail
cd "$(dirname "$0")"
set -a; . serve/serve.env; set +a
export PACK="${PACK:-/mnt/nvme0/bigmodels/dsv41-ablit-engram-q4}"
export ENGRAM_DISK=1 EAGER=0 PORT="${PORT:-8000}"
export DSV41_ENGRAM_DEDUP=1 ENGRAM_THREADS=64 MAX_MODEL_LEN=1048576
export NCCL_P2P_DISABLE=1 VLLM_PREFIX_CACHE_RETENTION_INTERVAL=64
./serve/serve-engram-vision.sh &
for i in $(seq 1 240); do
  curl -sf -m 5 -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "READY after ~$((i*5))s"; exit 0; }
  sleep 5
done
echo "not ready after 20m - check the serve log"; exit 1
