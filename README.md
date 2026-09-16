# DeepSeek v4.1 Flash EXL3 2.0bpw + Engram Q4, Abliterated - SM120, Dual RTX Pro 6000

Abliterated DeepSeek v4.1 Flash on 2x RTX PRO 6000 (SM120): **1M context + vision +
DSpark speculative decoding + prefix caching that actually works, simultaneously.**

Two defects had to be fixed to get here, both documented below with receipts:
prefix caching is dead-on-arrival in this vLLM line under DSpark (a retention-mask
default silently discards draft-KV hash commits - one env var fixes it), and
P2P-enabled NCCL init deadlocks on asymmetric Blackwell pairs (disable it).

Checkpoint on Hugging Face:
[satgeze/DeepSeek-v4.1-Flash-EXL3-2.0bpw-Ablit-EngramQ4-SM120-Dual-RTX-Pro-6000](https://huggingface.co/satgeze/DeepSeek-v4.1-Flash-EXL3-2.0bpw-Ablit-EngramQ4-SM120-Dual-RTX-Pro-6000)

**Quality gates are complete:** GSM8K-style math clean (225 / 10 with visible
reasoning), loop battery 0/8 at greedy temp-0, both vision tests correct
(red|blue split; two-image identification), prefix-cache second-pass 8-11x,
decode 89-94 tok/s single stream at 2K and 249 tok/s aggregate at 8x46K warm
agents. Numbers below and in [docs/BENCH.md](docs/BENCH.md).

## What this checkpoint is

| | |
|---|---|
| Base | deepseek-ai/DeepSeek-V4.1-Flash, EXL3 2.0bpw weights (stock tensor-for-tensor except where noted) |
| Abliteration | 52 grafted `attn.wo_b` tensors, layers 10-35 (graft method; experts, Engram, MTP, layers 0-9 and 36-39 remain stock) |
| Engram tables | recompressed fp8 -> MXINT-4 with e4m3 block scales: 189.1 GiB -> 97.6 GiB. Served from NVMe with dedup + page-cache warm, or pinned on boxes that can afford it |
| Hybrid builder | `tools/build_ablit_hybrid.py` verifies the graft against stock, then hardlinks ablit weight shards + MXINT-4 Engram shards into one pack |
| Runtime | vLLM `0.1.dev20904+g179dd0fa9` + `vllm_exl3` plugin, `tokenizer_mode=deepseek_v41`, DSpark `dspark_block_size=5` |

## Quick start

```bash
cp serve/serve.env.example serve/serve.env   # set API_KEY
./start.sh                                   # boots on :8000, waits for ready
```

Boot is 12-15 min from cold page cache (48 shards + JIT + graph capture).
After every boot, fire one small request before loading real work: the first
heavy batched request runs at a tenth of speed once (JIT/Engram path warmup).

## What runs (the stack, layer by layer)

| Layer | What | Where |
|---|---|---|
| API | OpenAI-compatible, tool calling (deepseek_v41 parser), vision | serve/serve-engram-vision.sh |
| Model | DeepSeek v4.1 Flash, CED arch, Engram at layers 1 and 14, native DSpark | checkpoint config.json |
| Weights | EXL3 2.0bpw + abliteration graft (see above) | checkpoint |
| Engram | MXINT-4 NVMe reader with n-gram dedup + eager graph break | patches/vllm/engram_disk_q.py, engram_graphbreak.py |
| KV cache | fp8 MLA, 8 GiB pool (~4.4M tokens) | serve script |
| Spec decode | DSpark, 5 draft tokens, probabilistic sampling | serve script |
| Parallelism | TP2 over PCIe, custom all-reduce disabled | serve script |
| Context | 1,048,576 max | MAX_MODEL_LEN |
| Caching | prefix caching with `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=64` - see RCA below | serve script |

## The two root causes (why your serve is slow or hangs)

**1. DSpark silently disables prefix caching.** This vLLM line defaults
`prefix_cache_retention_interval=0`. At 0, the sliding-window manager commits only the
newest boundary block per prefill; every decode-generated draft-SWA block is never
hash-committed, so the second identical prompt re-prefills from zero. The failure is
a one-request lag, not zero caching - which is exactly why every 2-pass measurement
"proves" caching is broken and every 3rd request suddenly hits. Fix:
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=64` (= block size; the contiguous-hit walk
needs every block committed, any sparser value collapses back to zero). Receipt:
PASS1 5.85s -> PASS2 0.53s, walk `match=True` on the second lookup.

**2. NCCL P2P init deadlocks on asymmetric GPU pairs.** Workstation + Max-Q
Blackwell pair, P2P enabled: both workers spin forever inside `ncclCommInitRank`
(100% CPU, 964 MiB VRAM, zero IO, log silence after the pynccl line). 2/2
reproductions; py-spy stacks in docs/ROOT-CAUSES.md. Fix: `NCCL_P2P_DISABLE=1`
on every boot of every script.

**Known wall:** 125 GiB boxes cannot pin the Engram tables for the ~110 tok/s
pinned rung - the load-time peak (pin + weight streaming) wedges the host even at
81 GiB pinned with 32 GiB modeled headroom. Full post-mortem in docs/ROOT-CAUSES.md.

## Results (2026-09-16/17, this exact weight set)

Policy: if a number is not in a dated table, treat it as unverified.

| Metric | Value | Verified |
|---|---|---|
| Decode, single stream, 2K ctx | 92-94 tok/s stock, 89.4 tok/s abliterated | bench, non-stream |
| Prefill, 46K ctx | 1622-1840 tok/s | bench TTFT |
| Concurrency aggregate decode, 8 streams @46K warm | 244-249 tok/s (31/stream) | warm-cache 2-phase |
| Warm-turn prefill, 12.7K prefix | 5.85 s -> 0.53-0.67 s (8-11x) | identical-prompt gate |
| Vision | red/blue split + two-image identification correct | image tests |
| Loop battery v1 (greedy, temp 0, 1500 tokens) | 0/8 loops | loop_rate.py |
| Tool calling | OpenAI format roundtrip verified (streaming + not) | live serve |
| Context | 1,048,576 max | serving |
| Boot to ready | 12-15 min cold | boot log |
| Checkpoint | 48 shards; 46 abliterated-weight + 2 MXINT-4 Engram | pack receipt |

Open defect, documented: 12-16 concurrent agents at 46K collapse (~20 tok/s
aggregate; repeated 50K prefill bursts, no preemption logs). 8 streams and under
are proven-safe territory. Analysis in docs/ROOT-CAUSES.md.

## Repo map

| Path | What |
|---|---|
| serve/serve-engram-vision.sh | the launcher (vision + DSpark + Engram NVMe + caching fix) |
| serve/serve-engram-q4-pinned.sh | pinned-RAM variant for boxes that can hold the tables |
| patches/vllm/ | instrumented KV coordinator + sliding-window manager, Engram NVMe reader, graph-break wiring, MXINT-4/MXINT-3 Triton lookup |
| patches/flashinfer/ | sm_120 sparse-MLA prefill/decode topk-1152 patches (vision raises text prefill topk) |
| bench/ | bench matrix, acceptance, concurrency ladder, loop battery |
| tools/ | Engram quant writer (MXINT-3/4, block 16/32), page-cache warmer, pin guard, ablit hybrid builder |
| docs/ | root-cause chain + full bench receipts |

Credits: FlashInfer patches build on the upstream sparse_mla_sm120 kernels
(JIT topk-1152 extension); vLLM patches are annotated edits of
`0.1.dev20904+g179dd0fa9`. The abliteration graft was produced by the wo_b
sidecar method - run `tools/build_ablit_hybrid.py` to verify any pack against it.
