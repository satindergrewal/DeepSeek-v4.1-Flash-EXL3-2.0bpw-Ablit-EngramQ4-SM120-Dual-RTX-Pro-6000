#!/usr/bin/env python3
"""DSpark acceptance length, measured the recipe's way.

AUTHORITATIVE METHOD: vLLM logs "Mean acceptance length: X.XX" once per request.
Average those lines. That is where the pack recipe's 3.66 came from.

Do NOT derive acceptance from /metrics counters -- that was tried and got it wrong:
spec_decode_num_draft_tokens_total is ROUNDS x num_speculative_tokens, not rounds,
and a delta taken across a window that mixes prefill and decode is not a decode
acceptance at all. The counters are kept below only as a rough cross-check.

Usage: LOG=<serve log> [PORT=8000] python3 acceptance.py
"""
import os, re

LOG = os.environ.get("LOG", "")
vals = []
if LOG and os.path.exists(LOG):
    vals = [float(m) for m in re.findall(r"Mean acceptance length: ([0-9.]+)", open(LOG, errors="ignore").read())]
if vals:
    print("  AUTHORITATIVE (per-request log lines from %s):" % os.path.basename(LOG))
    print("    mean acceptance length = %.2f over %d reports" % (sum(vals) / len(vals), len(vals)))
else:
    print("  no 'Mean acceptance length' lines found (LOG=%r) -- no authoritative number" % LOG)

try:
    import urllib.request
    PORT = os.environ.get("PORT", "8000")
    txt = urllib.request.urlopen("http://127.0.0.1:%s/metrics" % PORT, timeout=20).read().decode()

    def grab(name):
        for line in txt.splitlines():
            if line.startswith("vllm:" + name + " ") or line.startswith("vllm:" + name + "{"):
                return float(line.rsplit(" ", 1)[1])
        return None

    d = grab("spec_decode_num_drafts_total")
    t = grab("spec_decode_num_draft_tokens_total")
    a = grab("spec_decode_num_accepted_tokens_total")
    if d and a is not None:
        print("  cross-check (cumulative /metrics, may mix phases):")
        print("    drafts(rounds)=%.0f draft_tokens=%.0f accepted=%.0f" % (d, t, a))
        print("    1 + accepted/rounds = %.2f   <-- rounds, NOT draft_tokens" % (1 + a / d))
except Exception as e:
    print("  metrics cross-check unavailable: %s" % e)
