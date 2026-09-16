# DSV4.1-Flash EXL3 on the box (2x RTX PRO 6000 Blackwell) - 2026-09-14

Measured with the pack's own recipe/serve/serve.sh, TP=2, ENGRAM_DISK=1 EAGER=1,
--language-model-only, block-size 64, fp8_ds_mla KV (8 GiB), DSpark 5, max-num-seqs 8,
max-num-batched-tokens 4096, kernel-config autotune/jit-warmup off.
NCCL: the launcher forwards only NCCL_P2P_DISABLE (PASS_ENV covers VLLM_EXL3_*,
NCCL_PROTO, NCCL_ALGO, DENSE_HYBRID_M, XMOE_*; NCCL_P2P_LEVEL and NCCL_IB_DISABLE are
NOT forwarded into the container).

Engram tables (base shards 47/48, 2 x 94.56 GiB on disk) are read from NVMe
(DSV41_ENGRAM_DISK=1, 32 read threads).
Pinned-host-RAM Engram is not available on this box: measured host RAM is 125 GB (4 DIMMs
populated), and the board (ASUS ROG Crosshair X670E Extreme, AM5) is specified to 192 GB
(4 x 48 GB). The tables are 189.12 GiB = 203.1 GB, so they do not fit at EITHER capacity -
this rung is out by RAM size, not by tuning or by which DIMMs are installed.

## Arm A - 2.0 bpw (diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000) - LOADED

vllm-exl3 convention: bits 2, codebook mcg.

| | |
|---|---|
| EXL3 shard bytes on disk | 144.39 GiB total (estimate: 333.51 GiB of top-level shards minus the 2 x 94.56 GiB Engram shards); ~72.20 GiB per rank by storage arithmetic |
| card free at boot | 94.05 GiB/card |
| max_model_len accepted | 1,048,576 (the model ceiling; YaRN x16 over 65536) |
| largest prompt actually exercised | 327,661 tokens (deep smoke, ran clean) |
| KV | 8 GiB fp8_ds_mla -> 4,410,963 tokens; engine reports 4.21x concurrency at 1M |
| boot to ready | ~9 min |

bench (the pack's own bench/bench-sbs.py; numbers are the harness's):

| cell | prompt tok | prefill agg (per-stream, TTFT) | decode agg (per-stream) |
|---|---|---|---|
| 2K / 1  | 2,246   | 787 tok/s (787, 2.9s)      | 29.6 tok/s (29.6) |
| 2K / 4  | 2,208   | 3,034 tok/s (758, 5.2s)    | 134.9 tok/s (33.7) |
| 46K / 1 | 50,303  | 1,820 tok/s (1,820, 27.6s) | 29.8 tok/s (29.8) |
| 46K / 4 | 50,359  | 3,793 tok/s (948, 107.1s)  | 14.7 tok/s (3.7) |
| 300K / 1| 327,661 | 1,768 tok/s (1,768, 185.4s) | 27.5 tok/s (27.5) |

Pack README comparable row (eager + Engram on NVMe): 1,805 prefill / 31.5 decode.
Here: 1,820 / 29.8 at 46K/1 - same tier.

Not measured: KLD / perplexity / loop behaviour of this 2.0 bpw artifact.
The 46K/4 decode figure (3.7 tok/s per stream) is as reported by the harness and has
not been investigated for a cause.

## Arm B - 2.9 bpw (Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) - DID NOT LOAD

exllamav3-native convention: 2.90 bpw, codebook mul1, head_bits 6, version 1.4.2.
Shard bytes on disk: 196.14 GiB of EXL3 weights (385.26 GiB of top-level shards minus
the 2 x 94.56 GiB Engram shards) = ~98.07 GiB per rank by storage arithmetic, against
94.97 GiB card capacity / 94.05 GiB free at boot.

Three configurations, one identical failure each - CUDA OOM inside the exl3 MoE
`create_weights` path (`_init_fused_moe_experts` -> `vllm_exl3/exl3.py` -> torch.empty),
~93.3 GiB already in use, allocating 2.11 GiB with ~1.6 GiB free:

1. stock
2. `--cpu-offload-gb 6` (UVAOffloader)
3. `--offload-backend prefetch --offload-group-size 8 --offload-num-in-group 1`

Logs kept: serve-2.9bpw-oom.log, serve-2.9bpw-uvaoom.log, serve-2.9bpw.log.

The ~93.3 GiB-in-use figure is the same in all three, which suggests neither offload
path returns weight memory during the construction loop; that was not verified further
(parameters vs buffers, eligibility filtering, bytes actually moved, retained GPU
references all unexamined).

Speed and usable context for 2.9 bpw: **unmeasured** - it never reached a serve.

## Context ceiling

Configured/accepted ceiling on the box is 1,048,576 for this pack. Only 327,661 tokens
was actually exercised end to end; the 4.21x concurrency figure is the engine's KV
arithmetic, not a measured multi-request run at 1M.

## Weight budget arithmetic

94.05 GiB free per card at boot; the 2.0 bpw pack accounts for ~72.20 GiB of that by
storage arithmetic, leaving ~21.8 GiB/card for KV (8 GiB configured), activations and
workspace. Extrapolating that arithmetic to a pack of the same shape gives roughly
168 GiB of weights as the ceiling, i.e. about 2.5 bpw - an unmeasured estimate, not a
quote from anyone, and it assumes the same per-card split and the same KV size.

Engram-on-NVMe is load-bearing. It keeps 189.12 GiB of FP8 row tables off the cards;
pinned host RAM cannot hold them on this box.

## Quality of the 2.0 bpw artifact (measured 2026-09-14)

Read this section WITH the KL/PPL section below, not instead of it. On these two probes the
pack is indistinguishable from the recipe's own claim; on distribution fidelity it is not.
Both statements are true at once - see "capability probes are blind to it" at the end of the
KL section. Neither result alone is a verdict on the artifact.

Both probes are the pack's own (`recipe/bench/quality-probe.py`), greedy, temperature 0,
THINKING OFF - identical methodology to the 97.5% / 91.5% the pack itself claims.

| probe | this run | pack's own claim |
|---|---|---|
| GSM8K, first 200 | **196/200 = 98.0%** (0 truncated, 0 errors) | 97.5% |
| HumanEval pass@1, 164 | **151/164 = 92.1%** (0 truncated, 0 errors) | 91.5% |
| PPL, wikitext-2-raw test | **3.2753** (165,172 tokens, 1.7116 bits/token) | not published |

PPL method: `/v1/completions` echo logprobs over 8 chunks of the concatenated test split,
alignment verified by string round-trip, first token of each chunk excluded.

NOT measured (and these are where a 2 bpw quant is most likely to hurt): thinking ON,
long-context (>32K), agentic/multi-turn behaviour, and any loop/repetition probe.

## Why CUDA graphs are unavailable here (tested 2026-09-14)

`ENGRAM_DISK=1 EAGER=0` was launched to test whether the recipe's eager requirement is
real or conservative. It failed during graph capture, both ranks, same frame:

    vllm/v1/worker/gpu/model_runner.py:942 capture_model
      vllm/v1/worker/gpu/cudagraph_utils.py:399  forward_fn(CUDAGraphMode.NONE)
        vllm/models/deepseek_v4_1/common/engram.py:1136 prepare_embeddings
          engram.py:825 lookup -> engram.py:855 _disk_lookup
    RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture
    unless the CPU tensor is pinned.

Read as code, `engram.py:_disk_lookup` is host-side and data-dependent by construction:

    ids = indices.detach().to("cpu", dtype=torch.int64)   # D2H sync
    deq = self.disk.gather_dequant(file_rows, owned)      # blocking preads, pageable
    out[:t].copy_(deq.view(t, local_heads, self.dim))     # pageable H2D <- the error

So this is not a missing pin_memory: a device->host sync, blocking syscalls on the host,
and a pageable H2D cannot live inside one captured decode graph. Pinning the staging
buffer silences the reported error and leaves the D2H sync and the frozen-lookup problem.

Both fast rungs of the pack's own ladder therefore need the Engram tables resident in
host RAM (pinned) so the lookup is a Triton kernel over a resident table. Correction
(2026-09-14): dmidecode reports 128 GB in SMBIOS Type-16, but that field is unreliable on
this board - the ROG Crosshair X670E Extreme is specified to 192 GB, and the real installed
figure is 125 GB in 4 populated slots. Either way the tables are 189.12 GiB = 203.1 GB and
do not fit, so the rung is unreachable by RAM size, not by tuning.

The escape would be a graph break around the engram lookup (piecewise capture), so its
host code runs between graph segments. Not attempted; it means patching vLLM's
compilation config for a third-party pack, and success is not assured.

## KL divergence and perplexity (measured 2026-09-14)

### vs the OFFICIAL model (DeepSeek API `deepseek-flash` via Spock), on-policy, 4,000 positions

| arm | KL(official || arm), nats | top-1 agreement | support | official token inside arm top-20 |
|---|---|---|---|---|
| box EXL3 **2.0 bpw** | **0.43888** | 0.7710 | 13.8/20 | 1.0000 |
| spark EXL3 **2.9 bpw** | **0.08472** | 0.9035 | 17.4/20 | 1.0000 |

Protocol: the official API exposes logprobs only for tokens it generated, so the reference
positions are its own greedy trajectory (`/tmp/native_kld.py`). The token stream is rebuilt
from per-token bytes and must match the returned text byte-for-byte; each local arm renders
the same context through its own chat template (`/tokenize`, add_generation_prompt=true),
appends the official token text, and scores it with prompt_logprobs=20.
The load-bearing gate is the BYTE-EXACT token stream: the reference positions are rebuilt
from the API's per-token bytes and must match its returned text byte-for-byte, and each arm
must render the same context at the same token count. Corroboration only (NOT a proof of
alignment): the official's chosen token landed inside BOTH arms' top-20 at 100% of positions.
Top-20 inclusion is weak - a 20-wide window can contain a token for reasons other than a
shared template - so it is reported as a sanity check, not as the alignment argument.

The API is a sound native reference, not a stray model: KL(official || 2.9bpw) = 0.0847 sits
exactly where the community's numbers place a good quant (EXL3 3.5bpw vs native = 0.057).

### quant vs quant, corpus, 58,122 positions (wikitext-2-raw test)

KL(2.9bpw || 2.0bpw) = 0.35192 nats, KL(2.0bpw || 2.9bpw) = 0.54072, top-1 agreement 0.7995.

### the two measurements are consistent (but this is not a proof)

0.08472 (official -> 2.9) + 0.35192 (2.9 -> 2.0) = 0.4366, against 0.43888 measured directly
official -> 2.0: 0.5% apart. Read this as corroboration, NOT as cross-validation. KL is not
additive across an intermediate model - the triangle inequality bounds it, it does not
compose it - so the agreement is suggestive of a coherent error signal, not a derivation.

### perplexity, identical 165,172 tokens of wikitext-2-raw test

| arm | PPL | nats/token |
|---|---|---|
| box EXL3 2.0 bpw | **3.2753** | 1.1865 |
| spark EXL3 2.9 bpw | **2.5371** | 0.9310 |

The 2.0 bpw artifact costs 29% more perplexity and +0.256 nats/token.

### caveats

* Support-truncated KL (top-20 intersection, renormalised) biases every number DOWN. The box
  arm's support is 13.8/20 against the spark's 17.4/20, itself a divergence signal.
* The reference is DeepSeek's *served* V4.1-Flash. If the API model differs from the released
  weights these packs were quantised from, part of the gap is that difference. The 2.9bpw
  control argues it is small.
* GSM8K-200 (98.0%) and HumanEval (92.1%) did NOT detect any of this. Capability probes are
  blind to it.

## Torch-profiler breakdown, eager + Engram-on-NVMe (2026-09-14)

`recipe/bench/profile-run.sh` races the boot (it stops on "Traceback", which this serve emits
benignly while loading), so it profiled a server that was not up. `/tmp/profile2.sh` waits only
on "Application startup complete". One 26,801-token prompt + 64 decode tokens.

### DECODE, 128 steps (each step = DSpark k=5 + 1 = 6 rows)

| category | GPU kernel ms | share |
|---|---|---|
| **all-reduce (NCCL)** | 497.5 | **55.0%** |
| MoE (exl3 flat) | 175.1 | 19.4% |
| dense GEMM (FP8/marlin) | 144.6 | 16.0% |
| other (elementwise/norm/copy) | 44.5 | 4.9% |
| sparse attention | 14.9 | 1.7% |
| hyper-conn (mHC) | 14.2 | 1.6% |
| indexer | 9.3 | 1.0% |
| compressor/rope/quant | 4.1 | 0.5% |
| **engram** | **0.1** | **0.0%** |

Total kernel time in the 128 decode ranges: **904.4 ms**. Range wall times run 37-130 ms, most
around 75-100 ms. So the GPU is busy roughly **9% of decode wall time**: decode is launch- and
dispatch-bound, not compute- or bandwidth-bound.

NCCL detail: 2,028 `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` calls = ~16 per step at
**245 us each**. That is host-staged-PCIe territory, consistent with `NCCL_P2P_DISABLE=1`
being forced because P2P over the PHB bridge hangs `ncclCommInitRank` on this board.

### PREFILL, 21 chunk ranges (4092 and 2249-token chunks)

| category | GPU kernel ms | share |
|---|---|---|
| all-reduce | 3408.8 | 42.2% |
| MoE (exl3) | 2582.2 | 32.0% |
| dense GEMM | 1161.2 | 14.4% |
| sparse attention | 334.8 | 4.1% |
| other | 318.6 | 3.9% |
| hyper-conn | 173.4 | 2.1% |
| compressor/rope/quant | 86.0 | 1.1% |
| engram | 3.9 | 0.0% |

Total 8,080 ms of kernel over ranges of ~2.4 s (4092-token chunks) and ~0.57 s.

### What this means

1. **The Engram-on-NVMe path costs nothing measurable** (0.0-0.1% of kernel time). My earlier
   worry about the disk lookup being a major cost was wrong.
2. **The EXL3 MoE kernel is not the problem.** It is 19.4% of decode kernel time, which is
   ~1.8% of decode wall time. Faster kernels here would change almost nothing.
3. **The problem is that the GPU is idle ~91% of decode.** That is exactly what CUDA graphs
   remove, and CUDA graphs are unavailable: the disk-Engram lookup is host-side and
   data-dependent (`engram.py:_disk_lookup`), so it cannot live inside a captured region.
4. **NCCL all-reduce is the largest single kernel at 55% of decode kernel time** and runs at
   245 us/call, a host-staged figure. Worth attacking, but it only touches ~4 ms of a ~75 ms step.
5. Because the box is launch-bound rather than compute-bound, **a higher-bpw pack at the same
   architecture would cost almost nothing in speed** - and **raising DSpark's draft count is
   nearly free per step**, since extra accepted tokens ride on a step whose cost is overhead.

## EXL3 quantization cost, measured (2026-09-14)

Timed the recipe's own quantizer on ONE MoE layer (`model-00003-of-00048`, layer 0),
beam 16, batch 8, one GPU, in the shipping image:

    exl3 model-00003-of-00048.safetensors w2 376-384 n=8 1.4s tot=1152 5.57/s
    done experts=1152 elapsed=213s
    [quant-timing] rc=0  elapsed=228s for ONE MoE layer

**213 s of quantization per MoE layer** (1,152 experts = 384 x 3 matrices, 5.57 experts/s),
228 s wall including container start.

Extrapolated with the recipe's own parallelism (40 layers split 20/20 across the two GPUs):

* **full 40-layer pack on the box: ~76 minutes.**
* single GPU, all 40 layers sequentially: ~2.4 h.

So building a pack at another bitrate is a ~1-3 hour job, not a multi-day one. The recipe also
takes `--layer-bits '3:0-3,8,14,36-39'` for mixed per-layer K, so a mixed intermediate pack
(e.g. 2.3 / 2.5 / 2.7 bpw) is a supported build, not new tooling.

## RAM for pinned Engram: the arithmetic, definitively

Needed for the graphs-enabled tier: Engram tables resident in pinned host RAM.
Table size: 2 shards x 94.56 GiB = **189.12 GiB = 203.1 GB decimal**.

| | |
|---|---|
| board | ASUS ROG CROSSHAIR X670E EXTREME, AM5, Ryzen 9 7950X3D |
| board max per ASUS spec | 192 GB (4 x 48 GB) = 178.8 GiB |
| Engram tables alone | 189.12 GiB |
| shortfall | **~10.3 GiB before the OS, page cache or any host buffer** |

192 GB does not reach it. 256 GB would. AM5 is also unbuffered-DIMM only - ECC **RDIMM**
(registered) modules do not POST on this platform; that needs ECC UDIMM, and 4 x 64 GB is
beyond what AM5 officially supports. Conclusion stands, for a different reason than the
SMBIOS field that first suggested it.

---

# Addendum 2026-09-14 (late): measured memory, the real ceiling, and the speed verdict

Everything below replaces the estimate-based "Weight budget arithmetic" section above
where the two disagree. These are vLLM's own numbers and the pack's own byte counts.

## Measured memory profile (serve-profile.sh, util 0.92, --kv-cache-memory removed)

| quantity | per card |
|---|---|
| device total / free at boot | 94.97 / 94.05 GiB |
| desired utilisation 0.92 | 87.37 GiB |
| consumed: weights + non-torch | **78.29 GiB** |
| peak activation | 1.04 GiB |
| CUDA graph memory | 0.00 GiB (eager) |
| KV cache in use | 8.04 GiB -> 3,856,440 tokens, 7.36x @ 524,288 |

vLLM's own suggestion line: `--kv-cache-memory=8476497593` (7.89 GiB) fits the requested
budget, or `--kv-cache-memory=15645341696` (14.57 GiB) fully uses the GPU.

### the budget closes exactly

Pack composition, measured with `stat` over all 48 shards:

| shards | contents | bytes |
|---|---|---|
| 3-42 | MoE routed experts | 133.63 GiB |
| 1, 2, 43-46 | attention / dense / embed / head | 10.76 GiB |
| 47, 48 | Engram tables (NOT loaded to VRAM in disk mode) | 189.13 GiB |
| | total on disk | 333.51 GiB |

On-card weights = (133.63 + 10.76) / 2 = **72.19 GiB/card**, against a measured
78.29 consumed -> **non-torch overhead is 6.10 GiB/card** (CUDA context, NCCL, allocator).
Recomputing the weight budget from the measured profile:
87.37 - 6.10 - 1.04 - 8.04 = **72.20 GiB/card**, against 72.19 actually used.
**Headroom at the current settings is +0.00 GiB/card.**

The ~6.7 GiB/card that looked spare (94.05 free vs 87.37 budget) is OUTSIDE the utilisation
budget; it is not available for weights without raising `gpu_memory_utilization`.

Per MoE layer: 133.63/40 = **3.341 GiB at K=2**, **5.011 GiB at K=3** -> **1.670 GiB per
promoted layer**.

## The box's weight ceiling (this supersedes the "~2.5 bpw" estimate above)

| util | KV | weight budget/card | total | uniform-equivalent | K=3 layers possible |
|---|---|---|---|---|---|
| 0.92 | 8.04 GiB | 72.20 | 144.40 | 2.00 bpw | 0.0 of 40 |
| 0.96 | 8.04 | 76.00 | 152.00 | 2.11 | 4.5 |
| 0.96 | 4.00 | 80.04 | 160.08 | 2.22 | 9.4 |
| 0.98 | 4.00 | 81.94 | 163.88 | **2.27 (hard ceiling)** | 11.7 |

The 2.9bpw pack needs **196.14 GiB** of expert weights. The most the box can ever present is
**163.88 GiB**. Therefore:

* **2.9bpw is permanently unreachable on this box** - by 32 GiB, not by tuning.
* **The earlier "~2.45 bpw target" is above the real ceiling.** It came from storage
  arithmetic that did not subtract the measured non-torch overhead.
* Anything above util 0.92 also requires re-validating long-context memory, because the
  indexer workspace grows with context (the failure mode that killed the GLM 900K serve).

## Speed: the box is on the pack's own baseline rung

The pack's README ladder was measured on the SAME 2x RTX PRO 6000 hardware:

| rung | decode @46K, 1 stream |
|---|---|
| eager, Engram on NVMe | **31.5** |
| + CUDA graphs, Engram in pinned host RAM | 68.0 |
| + Marlin dense GEMMs | 90.5 |
| + multi-row prefill MoE | 96.7 |
| + flat decode MoE scheduler (default) | **116.6** |

This box measures **29.8**. It already runs Marlin (`--linear-backend marlin`), the xmoe
flat scheduler (`XMOE_EXT=xmoe`), multi-row prefill and the sm_120 fixes - so it is not
missing any kernel work. It is missing exactly one rung: CUDA graphs, which need the Engram
tables resident in host RAM. **The 3.9x is real and it is one problem, not five.**

## Engram quantization probe (2026-09-14)

Tables are MXFP8: `layers.{1,14}.engram.embed.weight` [384,006,168 x 256] F8_E4M3 plus
`.scale` [same rows x 8] F8_E8M0, one scale per 32 columns = 264 bytes/row.

Requantized to MXINT-K on a 2,097,152-row sample (the fp8 table is the reference, so this
is ADDED error only):

| K | bytes/row | both tables | rel_L2 vs fp8 | SNR |
|---|---|---|---|---|
| 2 | 72 | 51.5 GiB | 0.812 | 1.8 dB |
| 3 | 104 | 74.4 GiB | 0.311 | 10.2 dB |
| 4 | 136 | **97.3 GiB** | **0.152** | 16.4 dB |
| 8 | 264 | 188.8 GiB | 0.000 | - |

0.152 at K=4 is at the theoretical floor for 4-bit uniform quantization, so the encoder is
correct; the cost is intrinsic. Against 114 GB of available host RAM, **4 bpw fits with
~9 GB to spare, 3.5 bpw with ~22 GB**. Both tables must be resident - a partial pin does not
unlock capture, because the lookup happens at layer 1 AND layer 14.

Whether the model tolerates 4x the fp8 table error is **not** answerable from reconstruction
error; it needs an end-to-end run. The disk path (`DSV41_ENGRAM_DISK=1`) makes that a free
test: quantized tables, no RAM residency required, measure GSM8K / HumanEval / PPL / KLD.

## Abliterated pack verification (dsv41-exl3-2.0bpw-ablit)

Graft = 52 tensors (layers 10-35 `attn.wo_b.weight`/`.scale`), sidecar
`dsv41-ablate/wo_b_l10_35.safetensors`. Verified structurally, not by spot check:

* 188,245 tensors, source and destination index sets identical
* 52/52 sidecar tensors present in dst with matching shape, dtype and SHA-256
* 22/48 shards share dev+inode with the source -> provably untouched
* the 26 rewritten shards are exactly the 26 that hold edited tensors
* 52 sampled non-edited tensors (2 per rewritten shard) byte-identical to source
* `ABLIT_META.json` records method, sidecar and source

Not yet done: serving the abliterated pack, or any quality/refusal probe on it.

---

# Engram MXINT-4 quantization (2026-09-14)

The speed ladder above says every rung past ~31 t/s needs the Engram tables resident in
pinned host RAM. They are 189.12 GiB and the box has 125 GB, so the tables were quantized.

## What the tables are

`layers.{1,14}.engram.embed.weight` [384,006,168 x 256] F8_E4M3 plus `.scale`
[384,006,168 x 8] F8_E8M0 (one ue8m0 scale per 32 columns) = **264 bytes/row, 188.8 GiB
for both tables.** Layers 1 and 14 also carry small `q_weight`, `k_weight`, `wkv.*`.

## Encoding chosen: MXINT-4 with fp8-e4m3 block scales

Signed 4-bit codes, two per byte (low nibble = even column), one scale per 32 columns.

| K | bytes/row | both tables | rel_L2 vs fp8 | note |
|---|---|---|---|---|
| 2 | 72 | 51.5 GiB | 0.812 | |
| 3 | 104 | 74.4 GiB | 0.311 | |
| **4** | **136** | **97.3 GiB** | **0.099** | chosen |
| 4 (power-of-2 scales) | 136 | 97.3 GiB | 0.152 | rejected |
| 5 | 168 | 120.2 GiB | 0.047 | does not fit RAM |

Two findings worth keeping:

* **The scale dtype is worth 35% for free.** ue8m0 is power-of-two only, so a 32-block can
  waste nearly 2x of its range. Storing the scale as fp8-e4m3 costs the same 8 bytes/row and
  takes rel_L2 from 0.152 to 0.099.
* **4 bpw is the ceiling.** 5 bpw needs 120.2 GiB against 114 GB of available host RAM.
  A finer block (16) improves error but costs 103 GiB at 4 bpw - also out.

The step is stored pre-multiplied by a `prescale` of 8 so it lands in e4m3's normal range
(below 2**-6 e4m3 goes subnormal and sheds mantissa bits). Readers divide it back out; the
prescale is NOT baked into the codes.

## Artifacts

* `/mnt/nvme0/bigmodels/dsv41-engram-q4` - the pack. 242 GiB dir; 46 shards hardlinked to
  the 2.0bpw pack and only shards 47/48 rewritten. Tables total **97.58 GiB**
  (48.79 GiB each, verified byte counts). Verified: 46/48 shards share dev+inode with the
  source, 2 rewritten, and the four non-quantized tensors in the rewritten shard
  (`q_weight`, `k_weight`, `wkv.weight`, `wkv.scale`) are byte-identical to source.
* `/tmp/engram_write.py` - streaming writer. One read pass: codes stream straight out,
  scales are appended after the code region. 639k rows/s.
* `recipe/patches/vllm/engram_disk_q.py` - patched `DiskEngramTable` for the disk path.
  Detects the packed format from `width == dim/2`, so no config flag is needed.
* `recipe/patches/vllm/engram_q4.py` - patched `engram.py` for the **pinned-RAM** path
  (six edits: e4m3-aware scale loader, `PACKED`/`INV_PRESCALE` constexprs, packed kernel
  body using `tl.interleave`, module qbits flag, half-width allocation, call site).
* `recipe/serve/serve-engram-q4.sh` (Stage A, disk) and `serve-engram-q4-pinned.sh`
  (Stage B, pinned + CUDA graphs).

Two bugs that cost real time and would have shipped a corrupt artifact silently:

1. **Header sizes must use the dtype item size.** Sizing tensors as `prod(shape)` made the
   two BF16 tensors half their real length, so the file carried 40,960 trailing bytes and
   safetensors refused it. A truncated read of the *right* name and shape is the failure
   shape here, not an obvious crash.
2. **The output pack needs the index/config/tokenizer.** Hardlinking shards alone leaves a
   directory that cannot load at all.

## Round-trip

Quantize -> write -> read back -> dequantize gives **rel_L2 0.102** against the fp8
reference, matching the probe's prediction of 0.099 (the difference is e4m3 rounding of the
scale). Codes span [-7,7], 13.4% zero, scales within [0.5, 7.0].

## Smoke test (disk path, quantized tables)

All three greedy prompts correct: "capital of France" -> `Paris`; `17*23` -> `391`;
list completion -> `date, elderberry, fig`. The model is coherent on the quantized tables -
a broken table would not survive this. Model load 75.39 GiB vs 78.29 GiB for the fp8-Engram
pack. KV 3,836,841 tokens at 8 GiB.

## Stage A result: what the quantization costs (disk path, measured)

Identical probes, identical methodology, greedy, thinking OFF. The only difference between
the arms is the Engram table encoding.

| probe | fp8 Engram (baseline) | **MXINT-4 Engram** | delta |
|---|---|---|---|
| GSM8K first 200 | 196/200 = **98.0%** | 194/200 = **97.0%** | -1.0 pt, 2 truncated (baseline 0) |
| HumanEval pass@1, 164 | 151/164 = **92.1%** | 150/164 = **91.5%** | -0.6 pt |
| PPL, wikitext-2-raw test | 3.2753 | **3.3362** | **+1.9%** |
| tokens scored | 165,172 | 165,172 | identical |
| KV at 8 GiB | 3,856,440 tok | 3,836,841 tok | -0.5% |

**Verdict: usable.** For scale, the *weight* quantization gap on this model is
PPL 3.2753 (2.0bpw) vs 2.5371 (2.9bpw) = **+29%**, and KLD 0.439 vs 0.085. Halving the
Engram tables costs **+1.9% PPL**. The -1.0 pt GSM8K move is about 1 standard error at
n=200 (sigma ~0.99 pt), so it is not individually significant; the PPL move is systematic
and is the real cost.

KLD vs the official API was NOT obtained for this arm: `native_kld.py` needs the Spock
proxy on :8048 and it was not running. PPL is a teacher-forced distributional measure over
the same 165,172 tokens as the baseline, so the distribution shift is bounded by it.

Cross-arm note: GSM8K and HumanEval were ALREADY blind to a 0.439-nat divergence between
2.0bpw and 2.9bpw weights, so their silence here proves less than the PPL number does.

---

# SPEED WIN 2026-09-15: PIECEWISE CUDA graphs on the disk-Engram pack (2.95x)

The 30 t/s ceiling is broken. Disk-Engram + PIECEWISE CUDA graphs, no host-RAM pin.

| cell | eager (previous) | **PIECEWISE (now)** | speedup |
|---|---|---|---|
| 2K / 1 stream | 29.6 | **87.3 t/s** | **2.95x** |
| 46K / 1 stream | 29.8 | **75.9 t/s** | **2.55x** |
| 2K / 4 streams (agg) | 134.9 | **197.5** | 1.46x |
| 46K / 4 streams (agg) | 14.7 | 15.1 | ~1.0x |
| prefill 46K / 1 | 1,820 | 1,764 | -3% |
| boot to ready | ~9 min | 250 s | faster |

Correctness unchanged: smoke `17*23 -> 391`, capture succeeded (3 graphs), no capture error.
This is ~75% of the pack's own tuned 113.3/116.6 on the same 2-card hardware; the remaining
gap is believed to be the per-step eager break cost (two D2H syncs per decode step, one per
Engram layer).

Recipe: `recipe/serve/serve-engram-graphbreak.sh` +
`recipe/patches/vllm/engram_graphbreak.py` + `recipe/patches/vllm/engram_disk_q.py`.

## Why it took four attempts (each failure was a different layer)

1. **Reader mismatch.** The variant mounted only the engram.py patch, so the stock
   `DiskEngramTable` ran against the MXINT-4 pack's 128-wide tables ->
   `engram weight width (384006168, 128) != dim 256`. Both patches must be mounted.
2. **Wrong capture mode.** `FULL_DECODE_ONLY` (what serve.sh's EAGER=0 path uses) captures
   the WHOLE decode forward, Engram included. Needs PIECEWISE.
3. **Empty splitting_ops.** Registering the gather as a splitting op did nothing because
   `splitting_ops` is only populated when `optimization_level > O0`:
   `VllmConfig.__post_init__` sets `mode = VLLM_COMPILE` ONLY in that branch, and
   `set_splitting_ops_for_v1` bails to `splitting_ops = []` otherwise. serve.sh never
   passes `-O`, so the model runs at O0 with no split points. Confirmed by an in-patch
   diagnostic: `splitting_ops_was_None=False now_contains=True n=1` (mine only, not ~20).
   Also note `if pass_config.fuse_attn_quant and not use_inductor_graph_partition:` ->
   `set_splitting_ops_for_attn_fusion()`, which sets `splitting_ops=[]` AND forces
   `cudagraph_mode=FULL`.
4. **The actual fix: `@eager_break_during_capture`.** A decorator in
   `vllm/compilation/breakable_cudagraph.py` that makes capture end the current segment,
   run the function eagerly on the capture stream, record it for replay, and open a fresh
   segment. It needs NO compilation and no splitting_ops. Its stated requirements are
   exactly our situation: a custom-op Python kernel (`torch.ops.vllm.*`), an in-place
   caller-provided output buffer (`Engram.staged_rows`), and cudagraph_mode PIECEWISE
   (it deliberately does NOT break under FULL).

The splitting-op registration is retained in the patch: if compilation is ever enabled
(`-O1`+), it becomes the mechanism instead.

## SPEED 2026-09-15: what 87.3 t/s is, and why the recipe's 113/116 is not reachable here

Config under test: `serve-engram-graphbreak.sh` with `ENGRAM_DISK=1 EAGER=0` on the MXINT-4
pack. Harness is the recipe's own `bench/bench-sbs.py`, driven by `tools/bench-arm.sh`, so
every row below is the same measurement the recipe's README table uses.

| | 2K, 1 stream | 46K, 1 stream |
|---|---:|---:|
| eager, Engram on NVMe (the recipe's baseline rung) | 29.6 | 29.8 |
| **CUDA graphs, Engram still on NVMe (ours)** | **87.3** | **75.9** |
| recipe, graphs + Engram pinned in RAM, Marlin | 101.6 | 90.5 |
| recipe, + MoE kernel + flat scheduler | 113.3 | 116.6 |

**Graphs are worth 2.95x / 2.55x with the Engram tables held constant on NVMe.**

### The 113/116 rung needs ~190 GiB of pinned host RAM; the box has 125 GB total

The recipe's ladder says so directly: every rung above 68 t/s is "Engram in pinned RAM",
and its quick-start asks for "about 300 GB of free host RAM (the Engram tables are pinned,
~190 GiB, and serving used ~280 GB)". On the box:

```
               total        used        free      shared  buff/cache   available
Mem:             125          11           1           0         113         113
```

That is the wall, and it is silicon, not tuning. What we have is a configuration the
recipe never built: **CUDA graphs with the Engram tables still on NVMe.** Their NVMe rung
is eager-only (`serve.sh` line 9: "reads them from NVMe instead (needs EAGER=1)") and
scores 31.5. That is the rung we started from and the rung we beat by 2.95x.

### Acceptance length was a red herring, and my first number for it was wrong

I first derived acceptance 2.41 from `/metrics` counters. That was wrong twice over: I
divided by `spec_decode_num_draft_tokens_total` (rounds x 5) instead of
`spec_decode_num_drafts_total` (rounds), and I took the delta across a window that mixed
prefill and decode. The authoritative method is the recipe's own: average the
`Mean acceptance length:` lines vLLM logs per request.

| run | workload mix | mean acceptance | reports |
|---|---|---:|---:|
| ours, Q4 tables, eager + NVMe | GSM8K + HumanEval + PPL + bench | 4.55 | 43 |
| ours, Q4 tables, graphs + NVMe | 4 bench cells + smoke | 3.05 | 41 |
| recipe, fp8 tables, pinned, flat sched | needle + 4 bench cells + GSM8K | 3.66 | 36 |

**The MXINT-4 tables are not costing acceptance**: on their own workload they scored 4.55,
above the recipe's 3.66. Acceptance is workload-mix dependent and these three rows are not
comparable to each other -- a bench-only mix reads roughly 0.6 lower than a mix that
includes GSM8K. Any acceptance comparison across different suites is invalid.

### Knobs tested, with verdicts

| knob | verdict |
|---|---|
| `draft_sample_method=greedy` | **inconclusive; kept `probabilistic`** (also the recipe's shipped default). Greedy 83.0 / 210.2 / 90.3 / 15.3 vs probabilistic 87.3 / 197.5 / 75.9 / 15.1 on 2K/1, 2K/4, 46K/1, 46K/4. Log-derived acceptance is unchanged (3.14 over 35 reports vs 3.05 over 41), so the 46K/1 gap is NOT a drafting effect. |
| `NCCL_PROTO` / `NCCL_ALGO` | **dead by evidence.** The recipe's own decode-step table puts NCCL all-reduce at 0.6-1.6 ms of a ~40 ms step. Nothing to win. |
| `XMOE_EXT` flat decode scheduler | **already active.** `kernels/build/xmoe` is the only build present and the README marks the flat scheduler as default in it. |

### Run-to-run variance: the 46K/1 decode cell is noisy

Two runs of near-identical configuration (differing only in `draft_sample_method`) measured 46K/1 decode at
75.9 and 90.3 t/s, a 19% spread, while their acceptance lengths were 3.05 and 3.14 - statistically the same.
Acceptance cannot explain a 19% throughput move, so that 19% is session-level variance: clocks, the mixed-SKU
pair, or chunked-prefill interference. The pair is confirmed asymmetric: GPU0 is a 600 W Workstation
Edition, GPU1 a 300 W Max-Q, both 3090 MHz peak SM clock. At TP2 every step waits on the slower card, and
the recipe measured its whole ladder on TWO 300 W Max-Q cards - so we cannot beat their per-step time on
clocks, and our card 0 cannot help. A 3x power asymmetry is a plausible source of the 46K/1 spread.

Treat any single 46K/1 delta below ~20% as unresolved without a repeat. The 2K/1 cell is steadier
(83.0 vs 87.3, 5%) but not immune. Prefill cells reproduce much more tightly (1,378 / 1,444 / 1,748 / 1,764).

### Where the remaining 1.30x lives

There are two Engram layers (`engram_layer_ids: [1, 14]`), so exactly two eager breaks per
decode step. Step time from tokens/step and throughput:

- ours: (3.05+1) tokens/step at 87.3 t/s = 46.4 ms/step
- recipe: (3.66+1) at 113.3 t/s = 41.1 ms/step

Note the recipe's "23.5 ms per decode step" is the *target forward*, not the step:
23.5 ms x 113.3 t/s = 2.66 tokens/step, which contradicts its own 3.66 acceptance. The
wall-clock step also includes the draft model's forwards.

So the gap is ~5 ms/step, somewhere in the two NVMe gathers, the two graph re-entries, or
both. The `ENGRAM_NOOP` control separates them:

- `ENGRAM_NOOP=1` keeps the eager break and returns zeros instead of reading NVMe.
- near 113 t/s -> the read is the cost, and pinning the tables fixes it.
- still near 88 t/s -> the break itself is the cost, and pinning would NOT fix it.

### VERDICT: the remaining gap is the NVMe read, not the graph break, not acceptance

`ENGRAM_NOOP=1` keeps the `@eager_break_during_capture` break in place and returns zeros
instead of reading NVMe. So the difference between column 2 and column 3 is the disk read
alone, and the difference between columns 3 and 4 is what pinned RAM buys.

| | real disk read | ENGRAM_NOOP (break kept, read removed) | recipe, pinned |
|---|---:|---:|---:|
| decode, 2K, 1 stream | 87.3 | **104.4** | 113.3 |
| decode, 46K, 1 stream | 75.9 | **97.6** | 116.6 |
| prefill, 2K, 1 stream | 1,378 | **3,024** | 3,273 |
| prefill, 46K, 1 stream | 1,764 | **3,412** | 4,044 |
| TTFT, 46K, 1 stream | 28.5 s | **14.8 s** | 12.4 s |

Read off three ways:

1. **The graph-break patch is not the problem.** With the break intact and only the read
   removed we land at 104.4 / 97.6 t/s. That is 92% of the recipe's 113.3 and *above* its
   pinned + Marlin rung (101.6 / 90.5). The break is cheap; the read is not.
2. **Prefill is read-bound, not compute-bound.** 1,764 -> 3,412 t/s from removing a read is
   not a compute effect. Nor is TTFT halving.
3. **Everything else is exonerated.** Not acceptance (3.05 real vs 3.48 on a 4-report
   no-op sample, statistically the same), not NCCL, not the MoE scheduler, not the
   mixed-SKU pair.

So the lever is to make the read cheaper, not to buy RAM we do not have.

### The read itself: one preadv per row

`DiskEngramTable._read_rows` issues **one `os.preadv` per row** (`work()` loops row by row),
plus a `.tolist()` and a Python-level index per row. A 46K prefill is roughly:

    50,000 tokens x 8 heads x ~3 n-gram sizes ~ 1.2M rows, x2 (weight + scale) x2 layers

Order 1.5M syscalls, each of them a random 128-byte read. Deduplicating before the read is
the obvious fix: the same n-gram recurs constantly in real text, so distinct rows are a
small fraction of lookups. `DSV41_ENGRAM_DEDUP=1` does exactly that (unique + scatter).

This is also why prefill cannot be rescued by graphs: `CAPTURE_SIZES` tops out at 48 while
prefill chunks at 4096, so prefill always runs eager. Its only levers are the read and the
MoE kernel.

### ARM: `DSV41_ENGRAM_DEDUP=1` (+ disk threads 32 -> 64). Keeper.

Correctness gates first, because dedup must not move quality:

| gate | result |
|---|---|
| smoke, `17*23` | `391` OK |
| GSM8K-200 canary | **97.5%** (200 items, 169 s) vs 97.0% baseline -- no movement |

Speed, same 4-cell bench:

| cell | baseline (read per row) | dedup | delta | no-op ceiling |
|---|---:|---:|---:|---:|
| decode 2K/1 | 87.3 | **91.2** | +4.5% | 104.4 |
| decode 2K/4 | 197.5 | **203.5** | +3.0% | |
| decode 46K/1 | 75.9 | **86.1** | +13% | 97.6 |
| decode 46K/4 | 15.1 | **16.0** | +6.0% | |
| prefill 2K/1 | 1,378 | **1,642** | +19% | 3,024 |
| prefill 46K/1 | 1,764 | **1,833** | +4.0% | 3,412 |

A real gain everywhere, but it captures only about a third of the no-op headroom at 2K and
about half at 46K. That is expected on THIS prompt: `bench-sbs.py` builds its document from a
25-word vocabulary, so 4-grams are mostly distinct and there is little to collapse -- only the
2-gram head column repeats heavily (625 possible 2-grams over 4,096 tokens). Real code and
prose repeat n-grams far more, so the dedup win should be larger on the real agentic workload
than this bench shows. Unverified on a real corpus; `CORPUS=` is what `profile-run.sh` wants.

Note dedup also makes the reads ascend: `torch.unique` returns sorted rows, so the surviving
`preadv`s walk the file in file order instead of at random. That is a second, un-isolated
benefit bundled into this arm.

### ARM: preload the Engram shards into PAGE CACHE. The big one.

The last two shards (`model-00047/48-of-00048.safetensors`, 48.78 + 48.79 = **97.57 GiB**) hold only
the Engram tables, and the loader *skips* them (`is_engram_embed_tensor`), so they are never read
at load and therefore never cached -- every gather is a cold NVMe read, no matter how many tokens
revisit the same n-gram. So warm them once, after the model is up:

    97.57 GiB read in 20 s (4.85 GiB/s)

That is the whole trick, and it is safe: page cache is **reclaimable**, unlike the `pin_memory`
attempt that wedged the box. If the kernel needs the RAM back it evicts and we degrade to the
disk path we already measured. The failure mode is a slowdown, not a wedge, which is exactly why
this is acceptable where pinning was not.

| cell | baseline | dedup | **warm** | no-op ceiling |
|---|---:|---:|---:|---:|
| decode 2K/1 | 87.3 | 91.2 | **97.1** | 104.4 |
| decode 46K/1 | 75.9 | 86.1 | **93.7** | 97.6 |
| prefill 2K/1 | 1,378 | 1,642 | 1,131 | 3,024 |
| prefill 46K/1 | 1,764 | 1,833 | 1,859 | 3,412 |

**+11% at 2K and +23% at 46K on decode, reaching 93% and 96% of the read-free no-op ceiling.**
For zero extra hardware margin.

The prefill 2K/1 cell reads 1,131 here against 1,378-1,642 in other arms; that cell is noisy (its
TTFT moves between 1.4 s and 2.0 s for a 2.2K prompt), so treat the single number as unresolved.
The 46K prefill cell is steadier at 1,859 and only 45% of its no-op ceiling, which says prefill
still has real room -- a warm cache helps less there because a 50K prefill streams far more
distinct rows than fit in what is left of RAM afterwards.

### FULL SPEED LADDER, and which config ships

All cells from the recipe's own `bench/bench-sbs.py`. "no-op ceiling" is the read removed
entirely, i.e. the most this hardware could ever do with these tables.

| arm | 2K dec/1 | 2K dec/4 | 46K dec/1 | 46K dec/4 | 2K prefill/1 | 46K prefill/1 |
|---|---:|---:|---:|---:|---:|---:|
| eager + NVMe (where we started) | 29.6 | 134.9 | 29.8 | | | |
| graphs only | 87.3 | 197.5 | 75.9 | 15.1 | 1,378 | 1,764 |
| + dedup | 91.2 | 203.5 | 86.1 | 16.0 | 1,642 | 1,833 |
| + warm cache | **97.1** | | 93.7 | | 1,131 | 1,859 |
| + dedup + warm | 91.1 | 204.9 | **94.5** | 15.6 | 1,741 | 1,876 |
| no-op ceiling (read removed) | 104.4 | | 97.6 | | 3,024 | 3,412 |
| recipe, pinned RAM + MoE kernel | 113.3 | 277.2 | 116.6 | 30.7 | 3,273 | 4,044 |

**We went 29.8 -> ~94 t/s at 46K, 3.2x, with no RAM we cannot afford.**

#### dedup and warm are SUBSTITUTES, not complements

They attack the same cost from opposite ends: warm makes each `preadv` cheap (cache hit),
dedup makes there be fewer of them. Once reads are hits, paying `torch.unique` CPU to
remove reads is a wash -- which is why dedup+warm (91.1) does NOT stack over warm alone
(97.1) at 2K, and the two are tied at 46K (94.5 vs 93.7).

Do not read that as dedup being useless. Read it as: **ship both, because you cannot count
on the cache being warm in production.** A long agentic session reads 200+ GiB of model
shards and KV churn, which evicts Engram pages; dedup is then the floor that keeps the cold
case at 91 rather than 87. Warm is the upside, dedup is the insurance.

#### Correctness across every arm

| gate | result |
|---|---|
| smoke, `17*23` | `391` in every arm |
| GSM8K-200, baseline / dedup / final | 97.0% / 97.5% / 96.0% (n=200; one item = 0.5%, so this is flat) |

No arm moved quality. The dedup path is byte-exact against the non-dedup path in the sense
that matters: same rows, same scatter, same output.

#### Unresolved

The 2K/1 decode cell has spread 83.0 to 97.1 across arms of near-identical config, so the
warm-vs-dedup+warm delta (97.1 vs 91.1) is NOT resolved -- it is inside that cell's noise.
The 46K/1 cell is at least self-consistent (warm 93.7, dedup+warm 94.5). If this needs to be
settled rather than shipped, run warm-only and dedup+warm three times each at 2K/1.

## CONTEXT CEILING 2026-09-15: 1,048,576 tokens, verified serving at 998,858

`MAX_MODEL_LEN` is a **serve setting** (default 524288), not a model limit. The model's
`max_position_embeddings` is 1,048,576 with YaRN (factor 16 over a 64K base), so 1M is what
it was built for. The recipe's 512K was a round number, not a constraint.

| prompt tokens | % of cap | TTFT | prefill | result |
|---:|---:|---:|---:|---|
| 137,826 | 13% | 79 s | 1,738 tok/s | OK |
| 551,095 | 53% | 316 s | 1,743 tok/s | OK |
| 826,326 | 79% | 502 s | 1,645 tok/s | OK |
| **998,858** | **95%** | 632 s | 1,581 tok/s | **OK** |
| 1,048,513 | 100% | 2 s | — | **400 rejected** |

The rejection is exact and self-describing, so it cannot be mistaken for a hardware wall:

    This model's maximum context length is 1048576 tokens. However, you requested 64 output
    tokens and your prompt contains at least 1048513 input tokens, for a total of at least
    1048577 tokens.

One token over. **The cap is 1,048,576 total (prompt + generation), and it is enforced to
the token.**

### Prefill does NOT degrade at long context

1,738 / 1,743 / 1,645 / 1,581 tok/s at 138K / 551K / 826K / 999K. Flat within 10% across a
7x range of context length -- the DSA sparse attention (`index_topk: 512`) is doing its job.
The practical cost is wall-clock only: **a full 1M prefill is ~11 minutes** at 1,581 tok/s.

### KV headroom at 1M

Raising `MAX_MODEL_LEN` to 1048576 grew the KV pool from 3,836,841 to **4,410,963 tokens**
(vLLM sizes the pool from what the wider window lets it allocate), giving **4.21x
concurrency even at a full 1M per request**. KV is not the constraint at any context we can
actually use.

### Method note

The first 1M attempt returned 400 in 2 seconds and could easily have been written up as a
capacity ceiling. It was not: the prompt builder targeted 1,048,576 and produced ~1,048,776
tokens, past the cap. **Capture and print the HTTP error BODY on any reject** -- a 2-second
400 is a validation error, not an OOM, and the two look identical if you only record the
status code.
