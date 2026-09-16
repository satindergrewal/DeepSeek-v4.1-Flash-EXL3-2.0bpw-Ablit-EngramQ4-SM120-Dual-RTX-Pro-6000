#!/usr/bin/env bash
# Refuse to start a large pinned-host-RAM allocation unless it will actually fit.
#
# WHY THIS EXISTS: on 2026-09-14 a serve that pinned ~97 GiB wedged this box. `free -g`
# read "available 119 GB" and I treated it as headroom, but `available` counts RECLAIMABLE
# page cache (the box was holding 112 GB of it). Pinning does not politely consume free
# pages - it forces synchronous reclaim/compaction of that cache, and the machine went into
# sustained memory pressure ("Under memory pressure, flushing caches" for 17 min) with
# userspace starved. Ping still worked and :22 still accepted TCP, while sshd could not
# finish a banner. Recovery was a hard reboot.
#
# Usage:  preflight-pin.sh <GiB-to-pin> [--reserve GiB] [--drop-caches] [--yes]
# Exit 0 = safe to proceed, 1 = refuse.
set -uo pipefail
PIN=${1:?usage: preflight-pin.sh <GiB-to-pin> [--reserve GiB] [--drop-caches]}
shift || true
RESERVE=16; DROPC=0; ASSUME_YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reserve) RESERVE=${2:?}; shift 2 ;;
    --drop-caches) DROPC=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

read -r TOTAL FREE CACHED <<<"$(awk '/^MemTotal:/{t=$2} /^MemFree:/{f=$2} /^Cached:/{c=$2} END{printf "%.1f %.1f %.1f", t/1048576, f/1048576, c/1048576}' /proc/meminfo)"
AVAIL=$(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo)

printf 'pinning   : %.1f GiB\n' "$PIN"
printf 'MemTotal  : %.1f GiB\n' "$TOTAL"
printf 'MemFree   : %.1f GiB   <- what pinning can take without reclaim\n' "$FREE"
printf 'Cached    : %.1f GiB   <- reclaimable, but forcing it is what wedged the box\n' "$CACHED"
printf 'MemAvail  : %.1f GiB   <- free + reclaimable; NOT a safe pin budget\n' "$AVAIL"
printf 'reserve   : %.1f GiB\n\n' "$RESERVE"

if (( DROPC )); then
  echo "dropping page cache ..."
  sync
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 \
    && echo "  dropped" || echo "  could not drop (needs root) - assuming worst case"
  read -r FREE <<<"$(awk '/^MemFree:/{printf "%.1f", $2/1048576}' /proc/meminfo)"
  printf '  MemFree now: %.1f GiB\n' "$FREE"
fi

# headroom that would REMAIN after the pin - this is what the reserve protects
ROOM=$(awk -v f="$FREE" -v pi="$PIN" 'BEGIN{printf "%.1f", f-pi}')
OK=$(awk -v f="$FREE" -v p="$PIN" -v r="$RESERVE" 'BEGIN{print (f - p >= r) ? 1 : 0}')
echo
if (( OK )); then
  echo "SAFE: pinning $PIN GiB leaves ~$ROOM GiB above the $RESERVE GiB reserve."
  exit 0
fi
echo "REFUSE: pinning $PIN GiB against MemFree $FREE GiB with a $RESERVE GiB reserve does"
echo "        not fit. MemAvailable ($AVAIL GiB) is NOT the number to size against."
echo
echo "        options: pin less, drop caches first (--drop-caches), or lower the table"
echo "        precision. Reboot first if the box has been doing heavy disk I/O."
exit 1
