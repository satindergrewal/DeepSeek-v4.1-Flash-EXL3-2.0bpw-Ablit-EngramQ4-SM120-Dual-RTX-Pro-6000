#!/usr/bin/env bash
# warm-engram.sh [PACK] -- preload the Engram table shards into the page cache.
#
# WHY: the loader SKIPS the Engram tensors (is_engram_embed_tensor), so their shards are never
# read at startup and never cached. Every gather then pays a cold random NVMe read, which
# measured as the entire remaining gap to the recipe's pinned-RAM numbers.
#
# SAFE: page cache is reclaimable. If the kernel needs the RAM it evicts these pages and we
# degrade to the disk path. This is NOT pin_memory -- pinning ~97 GiB is what wedged the box,
# because pinned pages cannot be reclaimed.
#
# Run AFTER the server reports ready (loading the weights first would evict everything).
# Cost: ~20 s for 97.6 GiB at ~4.9 GiB/s on the box's NVMe.
set -euo pipefail
PACK="${1:-/mnt/nvme0/bigmodels/dsv41-engram-q4}"
python3 - "$PACK" <<'PY'
import json, os, sys, time
pack = sys.argv[1]
idx = json.load(open(os.path.join(pack, "model.safetensors.index.json")))["weight_map"]
shards = sorted({v for k, v in idx.items() if ".engram.embed." in k})
if not shards:
    print("no engram shards found in %s -- nothing to warm" % pack); raise SystemExit(1)
print("engram shards: %s" % ", ".join(shards))
t0, tot = time.time(), 0
for s in shards:
    p = os.path.join(pack, s)
    fd = os.open(p, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
    except OSError:
        pass
    with open(p, "rb") as fh:               # plain read() populates the page cache
        while True:
            b = fh.read(1 << 22)
            if not b:
                break
            tot += len(b)
    os.close(fd)
    print("  %s  %.2f GiB  t=%.0fs" % (s, tot / 2**30, time.time() - t0))
dt = time.time() - t0
print("warmed %.2f GiB in %.0fs (%.2f GiB/s)" % (tot / 2**30, dt, tot / 2**30 / max(dt, 1e-9)))
PY
grep -E 'MemFree|^Cached|MemAvailable' /proc/meminfo | sed 's/^/  /'
