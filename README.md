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

## Testing status: NOT rigorously tested

**Warning:** this is a hobbyist derivative checkpoint. It has NOT been through a full benchmark suite, refusal-behavior literature-standard evaluation, or safety red-teaming. What WAS run, and what was not, is below in full.

## Abliteration before/after (2026-09-17, same serve stack, back-to-back)

| Bench | Stock (pre-graft) | Abliterated (this checkpoint) |
|---|---|---|
| Refusal battery, harmful prompts (20) | **20/20 refused** | **2/20 refused** |
| Refusal battery, benign controls (10) | 0/10 refused | 1/10 refused |
| GSM8K, 50-problem test slice | 92.0% (46/50) | **96.0%** (48/50) |
| Loop battery v1, short context | 0/8 | 0/8 |
| Loop battery v1, 400K-token prefill | 0/8 | **1/8** (`enum` prompt, ttr 0.20) |
| Decode @2K, single stream | 92-94 tok/s | 89.4 tok/s |
| Concurrency, 8x46K warm | 244-249 tok/s | 260.2 tok/s |
| Concurrency, 12x46K | 17.3 tok/s (collapse) | 17.3 tok/s (same collapse) |
| Multi-tool (single / no-call / parallel-3) | 3/3 with schemas wired | 2/3 - parallel-3 answered single-call-then-wait; **0 DSML leaks** |
| Vision (2 synthetic image tests) | pass | pass |

What this says: the graft does exactly what an abliteration should - near-total
refusal removal on harmful archetypes (20/20 -> 2/20) - at the cost of one benign
false-positive (0 -> 1) and one repetition-attractor appearance at 400K depth
(0/8 -> 1/8). No cognitive cost measured: GSM8K 96% >= 92% stock (noise range),
concurrency equal-or-better, decode within noise. The 12-16 stream collapse
reproduces on BOTH weight sets: engine defect, not the graft.

Measurement caveats, stated plainly: 50-problem GSM8K slice (not the full 1319);
20-prompt refusal battery of archetype phrasings (not the Keys refusal32 set);
refusal classified by marker heuristic on the visible reply; loop-prefill uses a
repeated wikitext test split (732 KiB cycled to 400K), not unique prose.

Still not tested: full GSM8K, MMLU/HumanEval, 1M-token recall, real-photo vision,
multi-turn agentic soak, any safety red-team.

## What this checkpoint is

| | |
|---|---|
| Base quant | EXL3 2.0bpw by [diffbot](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000) - this checkpoint is a derivative of that quant (graft + Engram recompression on top) |
| Abliteration | 52 grafted `attn.wo_b` tensors, layers 10-35 (graft method; experts, Engram, MTP, layers 0-9 and 36-39 remain stock) |
| Engram tables | recompressed fp8 -> MXINT-4 with e4m3 block scales: 189.1 GiB -> 97.6 GiB. Served from NVMe with dedup + page-cache warm, or pinned on boxes that can afford it |
| Hybrid builder | `tools/build_ablit_hybrid.py` verifies the graft against stock, then hardlinks ablit weight shards + MXINT-4 Engram shards into one pack |
| Runtime | vLLM `0.1.dev20904+g179dd0fa9` + `vllm_exl3` plugin, `tokenizer_mode=deepseek_v41`, DSpark `dspark_block_size=5` |

## Quick start

```bash
./start.sh                                   # boots on :8000, waits for ready
```

Serves WITHOUT an API key by default. To require a Bearer token, set `API_KEY`
in the environment or `serve/serve.env` (see `serve/serve.env.example`).

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

## Results (2026-09-16/17; stock-pack rows marked)

Policy: if a number is not in a dated table, treat it as unverified. The
concurrency row was measured on the stock pack - the abliterated weights were
gated on everything in the testing section above, not the full matrix.

| Metric | Value | Verified |
|---|---|---|
| Decode, single stream, 2K ctx | 92-94 tok/s stock, 89.4 tok/s abliterated | bench, non-stream |
| Prefill, 46K ctx | 1622-1840 tok/s | bench TTFT |
| Concurrency aggregate decode, 8 streams @46K warm | 244-249 tok/s (31/stream) | warm-cache 2-phase (stock pack) |
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

## Sources and credits

This checkpoint exists on top of other people's work, in order:

| Source | What we took |
|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | the base model, architecture, tokenizer, DSpark draft |
| [diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000) | **the EXL3 2.0bpw quantization itself** - all 46 weight shards are diffbot's; we grafted the ablit tensors and recompressed the Engram tables on top. Without this quant there is no release |
| [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) | the 2.9bpw DGX Sparks quant, used as the reference for KLD cross-checks (kld-2.0-vs-2.9.json in the pack), and the release format |
| Abliteration | **drowzeys' Keys anchored-tensors method** - rank-1 `attn.wo_b` projection (lambda=3.5, layers 10-35), sidecar recaptured on TR3 and applied here to the 2.0bpw quant. See [drowzeys' packs](https://huggingface.co/drowzeys) for the method family |
| vLLM `0.1.dev20904+g179dd0fa9` + the `vllm_exl3` plugin + ExLlamaV3 kernels | the runtime this whole stack serves through |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) sm_120 sparse-MLA kernels | the prefill/decode paths our topk-1152 patches extend |
| DeepSeek `deepseek_v41` tokenizer and tool parser | chat template, reasoning split, DSML tool-call grammar |

License: inherits the DeepSeek V4.1 model license. Quantizations and derivative
checkpoints - check the source repos' terms before redistribution.
