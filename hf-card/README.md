---
license: other
license_name: deepseek
language:
- en
base_model: deepseek-ai/DeepSeek-V4.1-Flash
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
- deepseek
- v4.1-flash
- exl3
- abliterated
- engram
- dspark
- sm120
- 2xp6000
---

# DeepSeek v4.1 Flash EXL3 2.0bpw + Engram Q4, Abliterated (SM120, 2x RTX PRO 6000)

Abliterated DeepSeek v4.1 Flash at EXL3 2.0bpw with the Engram tables recompressed
fp8 -> MXINT-4 (189.1 GiB -> 97.6 GiB). Builds and measured on 2x RTX PRO 6000
Blackwell (SM120, TP2): **1,048,576 context + vision + DSpark speculative decoding
+ working prefix caching.**

Serving recipe, patches, and every root cause:
[github.com/satindergrewal/DeepSeek-v4.1-Flash-EXL3-2.0bpw-Ablit-EngramQ4-SM120-Dual-RTX-Pro-6000](https://github.com/satindergrewal/DeepSeek-v4.1-Flash-EXL3-2.0bpw-Ablit-EngramQ4-SM120-Dual-RTX-Pro-6000)

## What is in this checkpoint

| | |
|---|---|
| Base quant | EXL3 2.0bpw by [diffbot](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000) - this checkpoint is a derivative of that quant |
| Abliteration | 52 grafted `attn.wo_b` tensors, layers 10-35; experts, Engram, MTP, and layers 0-9/36-39 remain stock |
| Engram | layers 1 and 14 tables, fp8 -> MXINT-4 codes + e4m3 per-32 scales (shards 47-48) |
| Spec decode | native DSpark (`dspark_block_size=5`), draft experts included - no separate drafter |
| Runtime | vLLM `0.1.dev20904+g179dd0fa9` + `vllm_exl3` plugin, `tokenizer_mode=deepseek_v41`, ExLlamaV3 kernel path |

## Serving requirements (non-obvious, all measured)

1. `NCCL_P2P_DISABLE=1` - P2P-enabled NCCL init deadlocks on asymmetric Blackwell
   pairs (Workstation + Max-Q). Both workers spin in `ncclCommInitRank` forever.
2. `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=64` - this vLLM line defaults to 0, which
   silently discards draft-SWA hash commits under speculative decoding: the second
   identical prompt re-prefills from zero. 64 (= block size) restores it
   (5.9 s -> 0.5 s warm turn). Full root cause in the repo.
3. `--tool-call-parser deepseek_v41 --enable-auto-tool-choice --tokenizer-mode deepseek_v41`
   for tool calling; FlashInfer sm_120 patches for vision at topk 1152 (in the repo).
4. After boot, fire one small request before real work: the first heavy batched
   request runs at a tenth of speed once (JIT/Engram path warmup).

## Measured (2026-09-16/17, 2x RTX PRO 6000, TP2)

| Metric | Value |
|---|---|
| Decode, single stream, 2K ctx | 89.4 tok/s (92-94 stock, non-ablit) |
| Prefill, 46K ctx | 1622-1840 tok/s |
| Aggregate decode, 8 streams @46K warm | 244-249 tok/s (measured on the stock pack) |
| Warm-turn prefill, 12.7K prefix | 0.53-0.67 s (8-11x vs cold) |
| Context | 1,048,576 |
| Loop battery v1 (greedy, temp 0) | 0/8 loops |
| Math sanity | 17x23=391; GSM8K-style word problems correct with clean reasoning |
| Vision | red/blue split + two-image identification correct |
| Tool calling | OpenAI-format roundtrip verified, streaming and non-streaming |

## Abliteration notes

Graft method: attention `wo_b` output projection, layers 10-35, 52 tensors,
verified tensor-for-tensor against stock before release. No other tensors were
touched - this is a surgical refusal-direction ablation, not a full-network
retrain. Gates above were run on exactly these weights. Behavior deltas vs stock
are confined to refusal behavior; decoding speed is within noise (89.4 vs 92-94).

## Rebuilding / verifying

`tools/build_ablit_hybrid.py` (in the repo) re-verifies the graft hashes and
rebuilds the hybrid pack from a stock 2.0bpw pack + this repo's Engram shards.
The Engram quantizer (`tools/engram_write.py`) supports MXINT-3 and MXINT-4,
block 16/32; block-16 MXINT-3 also passes the loop gate (rel_L2 0.232) if you
need the extra 17 GiB back.

## Testing status: NOT rigorously tested

**Warning:** this is a hobbyist derivative checkpoint. It has NOT been through a
benchmark suite, a refusal-behavior evaluation, or any safety red-teaming. Do not
use it where wrong answers are expensive. The complete, honest list of every test
actually run on these weights follows.

### Run on the abliterated weights (this checkpoint, 2026-09-16/17)

| Test | Result |
|---|---|
| Greedy loop battery v1 (8 prompts, temp 0, 1500 tok) | 0/8 loops |
| Arithmetic sanity (17x23) | correct (391) |
| Two GSM8K-style word problems | correct (225, 10), clean reasoning |
| Vision, 2 synthetic images (red/blue split; green+white pair) | both correct |
| Decode speed @2K, single stream | 89.4 tok/s |
| Prefill @46K | 1667 tok/s |
| Prefix-cache second pass | functional (prompt served warm) |
| Text smoke on tool-call serve path | clean |
| Boot + engine init | clean, graphs PIECEWISE, vision warmup OK |

### Run on the STOCK (non-ablit) pack only, same serve stack

| Test | Result |
|---|---|
| Full bench matrix (2K/46K x 1/4 streams) | decode 80-94 tok/s, prefill 1.5-3.8K tok/s |
| Warm-cache concurrency, 8 streams @46K | 244-249 tok/s aggregate |
| Prefix-cache identical-prompt gate | 5.85 s -> 0.53 s (8-11x), walk-level receipts |
| 512K-context loop battery | 0/8 loops |
| 1M context boot + serving | max_model_len=1048576 accepted |
| N=16 concurrency @46K | COLLAPSES (~20 tok/s) - open defect, documented |
| Tool calling (single tool, get_weather) | correct tool_calls, streaming + non-stream |

### NOT tested (explicitly)

- Any standard benchmark suite (GSM8K/MMLU/HumanEval/etc.) on the abliterated weights - the three math problems above are the entire math evaluation
- Refusal behavior before/after the graft - "abliterated" is the graft method's claim, unmeasured here
- KLD/PPL fidelity of the graft on these weights (sidecar receipts exist from its creation; not re-measured)
- Long-context (400K+) runs on the abliterated weights
- 1M-token end-to-end recall (the serve accepts 1M; tonight's runs stayed short)
- Multi-turn agentic soak, parallel/multi-tool calls, tool-rejection paths
- Real photographs, OCR, or video (vision tests are two synthetic images)
- Safety evaluation of any kind

## Sources and credits

- Base model: [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- **EXL3 2.0bpw quantization: [diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000)** - all weight shards are diffbot's quant; this repo adds the ablit graft and the MXINT-4 Engram recompression
- 2.9bpw reference quant used for KLD cross-checks: [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
- Runtime: vLLM + vllm_exl3 plugin + ExLlamaV3 kernels; FlashInfer sm_120 kernels under the vision patches

License inherits the DeepSeek V4.1 model license.
