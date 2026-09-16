# DSV4.1-Flash box: ranked fix plan (2026-09-15)

Produced by a 17-agent diagnostic sweep + adversarial verification. Box: 2x RTX PRO 6000
Blackwell sm_120, TP2, pack /mnt/nvme0/bigmodels/dsv41-engram-q4.

## CORRECTIONS to earlier claims in BENCH-RESULTS.md

1. **FI_PATCH was NEVER off.** All 8 scripts in recipe/serve/ set `FI_PATCH=${FI_PATCH-1}`
   at line 37. The patch was active in every run. The prefill gap is NOT explained by it.
   Do not re-open this as a defect.
2. **Vision is template instantiation, not kernel engineering.** In
   `flashinfer/data/include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh`,
   TOPK appears in only four places and **none sizes a buffer**: `static constexpr int
   NI = TOPK / BI` (1152/64 = 18, integral), a `topk_len` clamp, and two loop bounds. The
   kernel is already a tiled online-softmax accumulator with a runtime trip count.
3. **`--limit-mm-per-prompt {"image": 4}` is a configured ceiling, not evidence 4 images
   work.** Both image tests 500'd.

## FIX-1 [BLOCKER] V4.1 vision: add topk=1152 to THREE dispatch sites

`combined_topk = round_up(top_k + window_size + max_image_tokens, 128)`, and the index
width that reaches the kernel is `prefill_index_width = window_size + max_image_tokens`
= 128 + 1024 = **1152** once the vision overrides are set.

**Consequence that makes a partial patch dangerous:** `sparse_swa.py:444-449` sets
`max_image_tokens = getattr(hf_config,'vision_max_n_token',0) if vision_n_layers>0 else 0`.
So with the overrides in place, **TEXT-ONLY prefills also carry topk=1152**. A patch that
fixes only the path in the stack trace will break text as well. The 391 smoke test is
therefore a regression test for the NEW kernel path, not the old one.

Edits to `recipe/patches/flashinfer/sparse_mla_sm120_prefill.cu`:

  A. dispatch_dsv4_single (~line 297-309), insert before the 2048 case:
       else if (topk == 1152) DISPATCH_BY_NH_CM(FP8, 1152);
  B. dispatch_dsv4_dual (~line 324, 396): it hard-requires `topk == 128` in BOTH the
     guard at 324 and the bail at 396. Must accept 1152 and instantiate the
     DISPATCH_DUAL_MG_FULLTILE / DISPATCH_FULLTILE_BY_NH_PBSX templates at TK=1152,
     for extra_page_block_size 64 (our config) and 32.
  C. decode side, `sparse_mla_sm120_decode_dsv4.cu`: `DSV4_DISPATCH(H,K)` covers
     K in {128,192,256,512,1024} - no 1152. Add DSV4_DISPATCH(32,1152). AND update the
     Python gate `_DECODE_DSV4_DISPATCH` frozenset in
     `flashinfer/mla/_sparse_mla_sm120.py:88-115` in lockstep - if the two disagree,
     the runtime silently picks a different path.

Then rebuild (JIT; the empty-aot mount already forces the build) and gate on:
  - 391 smoke (exercises the new TK path)
  - a real image, on a THROWAWAY container/port (see FIX-2)

FALLBACK if the dual instantiation misbehaves: padding the index width to an ALREADY
instantiated value (2048) needs no kernel change for the single path, at 33.5 MB instead
of 18.9 MB of index workspace. It does NOT fix the dual path.

## FIX-2 [GATE, no code] Never co-serve vision on the agent-facing engine until FIX-1 lands

The vision failure is a fatal `TVM_FFI_ICHECK` in the worker, surfacing as EngineDeadError.
It kills the WHOLE engine: every concurrent agent, including plain text, gets 500 until
restart. Validate any mm path on a separate container and port.

## FIX-3 [BLOCKER] Prefix caching: discriminating experiment BEFORE any code change

Measured: identical 53K prompt twice = 27.6s then 27.0s (full re-prefill both), 0.0% hit
rate, fresh KV allocation each time. Cause not yet attributed. Run, in order, with the
identical-prompt A/B:
  (a) SPEC=none            -> is it DSpark?
  (b) graphbreak patch off -> is it our engram patch?
  (c) stock serve.sh pack  -> is it upstream vLLM for this model?
If it survives all three it is upstream and belongs in a vLLM report. Until then, size
agents assuming FULL re-prefill per turn.

## FIX-4 [SILENT 2.95x LOSS] Capture sizes must track MAX_NUM_SEQS

vLLM returns CUDAGraphMode.NONE when num_tokens > max(cudagraph_capture_sizes), so the
step runs EAGER. CAPTURE_SIZES tops out at 48 = 8 seqs x (1+5). Setting MAX_NUM_SEQS=16
(96 tokens/step) silently loses the entire graph win and lands back near 30 t/s with no
error. Derive CAPTURE_SIZES from MAX_NUM_SEQS*(1+num_spec), or assert at startup.

## FIX-5 [QUALITY HONESTY] THINKING defaults ON; all quality numbers measured OFF

Every shipped serve script resolves THINKING=on at effort 50. All our evidence (GSM8K
98.0/97.0, HumanEval 92.1/91.5, PPL 3.2753/3.3362) was collected with thinking OFF. Pin
THINKING=off for the agent-facing serve, and make thinking-on a probed arm before it
becomes the default. Note the switch is process-global.

## FIX-6 [CAPABILITY] xmoe is built K=2 MCG only

TORCH_CHECKs any other variant at call time. The documented route past 2.0bpw divergence
needs K=3. Either build the K=3 instance in kernels/build.sh when a mixed pack is built,
or state in README that XMOE_EXT must be cleared above 2bpw.

## FIX-7 [MEASUREMENT] Stop trusting the log hit-rate line

`Prefix cache hit rate` is a 1000-request rolling metric rounded to one decimal, so real
but small reuse reads 0.0%. For an A/B use `vllm:prefix_cache_queries_total` /
`vllm:prefix_cache_hits_total` deltas around a single request pair.

## FIX-8 [OPS, cheap]

- NCCL_P2P_LEVEL / NCCL_IB_DISABLE are silently dropped by the PASS_ENV whitelist. Only
  NCCL_PROTO and NCCL_ALGO actually reach the engine. Extend the regex or document it.
- NO_CUSTOM_AR also injects PYTORCH_CUDA_ALLOC_CONF=expandable_segments. Separate them.
- Spec-decode Triton kernels JIT-compile on FIRST USE, after "Application startup
  complete". Warm one realistic prompt at boot before declaring ready.
- Zero weight headroom: budget closes at +0.00 GiB/card (72.20 budget vs 72.19 used).
  Vision (~1 GiB) and DENSE_HYBRID_M (+3.3 GiB/rank) are unbudgeted. Re-run the near-1M
  prefill walk after any such change.
- Engram n-gram hashes are correct-by-docstring; the reuse invariant is asserted only in
  prose. Add a debug-only id-at-slot assertion.
- Page-cache warm is RECLAIMABLE, so it is not sustained throughput. Keep
  DSV41_ENGRAM_DEDUP=1 permanently as the cold-case floor.

## PREFIX-CACHE ROOT CAUSE: fully diagnosed 2026-09-15 (instrumented, proven)

The blocker is NOT DSpark itself. It is the interaction of DSpark's eagle last-block drop
with the 22 SLIDING-WINDOW layers (groups 0-21, SlidingWindowMLASpec, window=128, bs=64).

Proof chain (all instrumented, on-box):
 1. With dspark: MLA group hits 59,712 tokens PERFECTLY. The SW group returns 0. The
    coordinator min-intersects -> 0 for the whole request.
 2. SW debug: need=3 eagle=True, last_cached_i=142, match=False. The eagle drop raises the
    contiguous requirement 2 -> 3, and the SW manager only retains ~1-2 blocks at the tail
    (it frees blocks as they leave the window), so 3 consecutive can never exist.
 3. Excluding SW groups from eagle_group_ids drops the requirement to need=2 eagle=False -
    and it STILL returns 0, because only ONE SW block is cached at the tail. A sliding-window
    manager cannot cache more than its window, so it can never match the MLA group's hit.
 4. vllm has a mechanism designed for exactly this: HybridKVCacheCoordinator computes
    num_uncached_common_prefix_tokens = longest_hit_length - hit_length ("a shared prefix
    that a sparse-retention group has not cached yet") and returns it - but NOTHING in
    vllm/v1/ consumes it. It is dead code.

## THE REAL FIX (scheduler feature, not a flag)

Wire up num_uncached_common_prefix_tokens:
 1. In the scheduler, when num_uncached_common_prefix_tokens > 0: take the cache hit for the
    cacheable groups at FULL length (59,712), and schedule the lagging group's missing span
    (hit..full) as a PREFILL for that group only. The SW group then computes its window from
    the freshly-prefilled tail instead of requiring it from cache.
 2. Correctness gate stays: cached vs uncached generation must be byte-identical at temp 0.
 3. Effort: a focused scheduler change (kv_cache_manager + scheduler input mapping), a day of
    careful work with the harness already written (tools + /tmp/run-cachefix.sh MOUNT_SPEC).

## INTERIM CONFIG (running now, and the right default until the above lands)

SPEC=none: decode 57.7 t/s, but WARM AGENT TURN = 6.1 s vs 31 s -> 5.6x faster end to end.
DSpark stays available as an opt-in for long-form generation on a fresh context.

## CORRECTION to the prefix-cache diagnosis (adversarial review, 2026-09-15 late)

The mechanism stated above ("the SW manager frees blocks as they leave the window, so only
~1-2 survive") is REFUTED by code reading:
  - BlockPool.free_blocks keeps hash entries until physical reuse.
  - SlidingWindowManager hits include null placeholders below the matched run, so the window
    does NOT cap hit length.
A full trace of STOCK dspark on this model yields hit = 52,736 (run 1648..1644, eagle pop,
realign) - not the measured 0. So the real difference is in the DRAFT swa_cache group's real
hash coverage during decode, which neither the trace nor the earlier comment captured.
Also corrected: the eagle-flagged SW group is the DRAFT's swa_cache (dspark draft layers are
DeepseekV4DecoderLayer swa_caches registered last -> rule-2 annotation), not the target's
22 SW layers as stated.

Additional defect found in the first fix (D1): excluding SW groups with a possibly-empty
remainder drops tail protection entirely for a pure-SWA + eagle config (lookahead-polluted
tail becomes reusable, and the multi-module-MTP alignment check is silently skipped).

## CORRECTED PATCH (apply-checked, compiles, NOT yet validated on the GPU)

  recipe/patches/vllm/dspark-prefix-cache/kv_cache_coordinator.patch
  recipe/patches/vllm/dspark-prefix-cache/single_type_kv_cache_manager.patch

Changes vs the first fix: never-empty fallback (fixes D1), eagle drop preserved on non-SW
groups so tail protection survives model-wide, hot-path INFO logging demoted to debug
(fixes D4), comments corrected. NOTE: one logger init in the STM overlay is redundant.

## REMAINING UNKNOWN (the thing that actually decides this)

Measured hits are 0 even with need=2 eagle=False. The reviewer's trace says stock dspark
should hit 52,736. So the remaining question is: why does the DRAFT swa_cache group have no
usable hash coverage at lookup time during decode? Next step: keep the CACHEDBG/SWDBG
instrumentation, add per-group coverage dumps (hash->block presence for the last 8 hashes),
one boot. Do NOT quote the free-blocks mechanism again - it is refuted.

## ROOT CAUSE FOUND (2026-09-16 ~02:00 NZST)
DSpark vs prefix caching: WRITE path, not read path, not scheduler.
- Draft SWA (SlidingWindowMLASpec, 22 groups, win=128 bs=64) commit calls RUN every step,
  but reachable_block_mask gates every insert. This build hard-defaults
  prefix_cache_retention_interval=0 (config/cache.py:61-67, VLLM_PREFIX_CACHE_RETENTION_INTERVAL
  unset => 0). At 0, the SW mask commits ONLY the newest boundary tail per prefill
  ("segment_tokens = None if retention_interval == 0 else retention_interval"),
  so decode-generated blocks are NEVER hash-committed during DSpark decode.
- Receipt: SWDBG last_cached_i=198 of max_nb=199 (exactly one lookupable boundary);
  PASS2 identical-prompt lookup: MLA hit=12736 but SW hit=0 -> coordinator min -> 0 -> full re-prefill.
- ONE-REQUEST LAG, not zero caching: PASS3 of the same prompt hit fully (SW hit=12736,
  match=True, 0.56s vs 5.9s cold) because two prior prefills each committed one tail block
  (need=2 then satisfiable). All earlier "0.0% hit rate / 2-pass" measurements were
  structurally blinded: the first re-send can never hit under retention=0.
- num_uncached_common_prefix_tokens is CONSUMED (shared_prefix_boundary) but both consumers
  are inert on DSV4.1 (Marconi mamba gating; retention-shaping path) - scheduler wiring is a
  red herring. Reviewer patch set adds nothing beyond deployed (audit 2026-09-16).
- DEPLOYMENT DEFECT FIXED: vision serve script never mounted the instrumented KV files
  (stock code ran all night). Added MOUNTS_EXTRA hook + explicit mounts to
  recipe/serve/serve-engram-vision.sh; also added -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL
  pass-through. Both edits backed up (.bak-mount).
- FIX UNDER TEST: boot with VLLM_PREFIX_CACHE_RETENTION_INTERVAL=64 (= bs => dense commits).
  Gate: PASS2 of identical prompt must hit (SW match=True on second pass) + no decode-speed
  regression + images still correct.
- NCCL: P2P-enabled init deadlocks on the Workstation+Max-Q pair (ncclCommInitRank spin,
  2/2 boots 2026-09-15 night). NCCL_P2P_DISABLE=1 is REQUIRED on every boot of any script
  (defaults to 0 when launcher env unset). Hang signature: 100% CPU both workers, 964MiB
  VRAM only, zero IO, silence after pynccl init line. Diagnose with host-side:
  sudo py-spy dump --pid <worker> (passwordless sudo works; container ptrace is blocked).

## FIX PROVEN (2026-09-16 ~02:30 NZST)
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=64 boot:
- GATE PASS: PASS1 5.85s -> PASS2 0.67s (8.7x), SWDBG match=True on second lookup,
  SW hit=12736 (was 0). Container env confirmed VLLM_PREFIX_CACHE_RETENTION_INTERVAL=64.
- Images: TEST1 red|blue correct, TEST2 two-images "Green, white." correct.
- Correctness under cached prefix: varied ~3K-token prefix, identical correct answers
  both passes ("a half-buried brass key"), 3.09s -> 1.15s.
- Speed bench matrix (2K/46K x conc 1/4): see bench-retention64.log (run in flight at
  time of writing; results append below when done).

## FINAL PRODUCTION STATE (2026-09-16 ~14:00 NZST)
Serve (container dsv41-flash-exl3, image dsv41-flash-exl3-sm120):
  PACK=/mnt/nvme0/bigmodels/dsv41-engram-q4 ENGRAM_DISK=1 EAGER=0
  DSV41_ENGRAM_DEDUP=1 ENGRAM_THREADS=64 MAX_MODEL_LEN=1048576
  NCCL_P2P_DISABLE=1 VLLM_PREFIX_CACHE_RETENTION_INTERVAL=64 MAX_NUM_SEQS=8
  + MOUNTS_EXTRA: instrumented kv_cache_coordinator.py + single_type_kv_cache_manager.py
  (all bind-mounted; retention env needs the -e line added to the serve script)
Measured on this exact stack:
  decode 2K/1 = 94.2 t/s; 8 agents @46K warm = 249.2 t/s agg (31.2/stream);
  cache gate PASS2 11.0x; vision both tests correct; smoke 391.
NOTE: first heavy batched request after boot runs at ~1/10 speed once (first-batch
warmup of engram/JIT paths); a CTX=2000 CONC=1 probe primes it. Prime after every boot.
OPEN DEFECT (parked): 16 agents @46K collapses to ~20 t/s (prefill bursts ~50K every
~20s, decode 4-9 t/s, zero preemption logs, KV 10%). Not eager-fallback: capture list
pads (batch 7 -> 48 graph exists; verified _set_cudagraph_sizes + generator code).
Suspect: wave-2 cache misses re-prefilling serially at MAX_BATCHED=4096. N<=8 proven safe.
CLI NOTE: this build has --cudagraph-capture-sizes (nargs ints) but it produced a worker
JSONDecodeError when passed; unused for now (default list fine for <=8 seqs).

## POST-MORTEM: PINNED-BOOT WEDGE 2026-09-16 (~17:20-18:53 NZST, hard reboot by Satinder)
The MXINT-3 block-16 pinned boot pinned 81 GiB with preflight-pin.sh PASSING (32 GiB
modeled headroom over 26 reserve) and STILL wedged the box: ping OK, sshd banner never
completed, ping jitter 10->55ms. Load-time peak = incremental pinned alloc + ~145 GiB of
EXL3 weights streaming through host page cache + CUDA host allocs, all at once. The guard
models steady state; the load peak ate the whole 32 GiB.
VERDICT: the pinned-Engram speed rung (~105-110 t/s) is CLOSED on this 125 GiB box.
Fitting the load peak needs tables <= ~60 GiB (2.5bpw Engram = quality-absurd) or a
staged loader that pins only AFTER weights are fully resident. Do not retry without one.
MXINT-3 VERDICTS (preserved, see memory pinned-engram-load-peak-wedge):
- block-32: FAILS loop gate 1/8 (review prompt LOOPs; q4 baseline 0/8). Loop-rate caught
  it; PPL would not.
- block-16: PASSES 0/8, rel_L2 0.232 (vs 0.104 at 4bpw), math sane, decode 83.6 t/s disk.
- Pack: /mnt/nvme0/bigmodels/dsv41-engram-q3b16 (80.2 GiB tables). Usable for disk-path
  serving; NOT pinnable on this box.
- Stack: writer --bits 3 --block 16 (source scales are per-32 ALWAYS; output block is
  independent); disk reader auto-detects 3-bit from width, Triton path via
  DSV41_ENGRAM_QBITS=3 + DSV41_ENGRAM_BLOCK=16 (both envs now wired through both serve
  scripts). Two complement decode: codes >=4 map to -8 (NOT -4).
RECOVERY AFTER REBOOT: production q4 serve relaunched from the standard command (see
FINAL PRODUCTION STATE above). ~15 min to ready from cold cache.

## ABLIT MODEL LIVE (2026-09-16 ~22:20 NZST)
Per Satinder order: switched the daily serve to the ABLITERATED weights.
- Hybrid pack: /mnt/nvme0/bigmodels/dsv41-ablit-engram-q4 = ablit weight shards
  (graft wo_b l10-35, 52 tensors, verified differs from stock) + q4 pack MXINT-4
  engram shards 47/48 (hardlinks) + ablit config files. Builder:
  dsv41-engram-q4/tools/build_ablit_hybrid.py (verifies graft + shard layout first).
- Gates on the hybrid: smoke 391 / math 225+10 / cache PASS2 live / decode 89.4 t/s
  @2K warm / vision both tests / loop v1 0/8 (0%).
- Stock pack swap-back: same command with PACK=/mnt/nvme0/bigmodels/dsv41-engram-q4.

## 2026-09-17: prefix-reuse semantics (source-confirmed) + orphan churn + outage

- `prefix_cache_retention_interval` IS segment-tail commit spacing (confirmed in
  vllm/config/cache.py + single_type_kv_cache_manager.py: ">0 -> a tail once per
  retention_interval-sized segment"; the deprecated env var is still honored).
  The original RCA read was correct. What the fix does NOT guarantee is
  cross-request reuse at long context: matchability of an older context's blocks
  is not reliable on this build (see docs/BENCH.md defect 1). Do not build
  measurements - or product claims - on assumed long-context prefix reuse.
- Client disconnects do not reliably abort in-flight prefills on this build:
  orphaned 218K-token requests re-admitted and re-prefilled for 20+ minutes at
  100% GPU with zero clients attached. Killed a benchmark mid-run? Restart the
  serve before trusting anything after it.
- Outage 2026-09-17 ~11:30 NZST: during a ladder relaunch boot the box wedged
  (ping alive, every listener RST, sshd banner stall - OOM-cascade class, second
  wedge of the campaign; prior HF staging pushed ~600GB through page cache and
  repeated cold Engram boots are the suspected accumulation). Serve relaunches
  on this box now follow: check for orphan processes, docker rm -f, ONE boot,
  verify health BEFORE load. NCCL_P2P_DISABLE=1 and
  VLLM_PREFIX_CACHE_RETENTION_INTERVAL=64 are persisted in serve.env.
