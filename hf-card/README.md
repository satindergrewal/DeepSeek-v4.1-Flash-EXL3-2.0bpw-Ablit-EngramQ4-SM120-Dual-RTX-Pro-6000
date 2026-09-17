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
| Aggregate decode, 8 streams @46K warm | 244-249 tok/s (stock pack; superseded by decode-window protocol, see quality section) |
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

**Warning:** hobbyist derivative checkpoint - no full benchmark suite, no safety red-team.

## Abliteration before/after (2026-09-17, same serve stack, back-to-back)

| Bench | Stock (pre-graft) | Abliterated (this checkpoint) |
|---|---|---|
| Refusal battery, harmful prompts (20) | **20/20 refused** | **2/20 refused** |
| Refusal battery, benign controls (10) | 0/10 refused | 1/10 refused |
| GSM8K, 50-problem test slice | 92.0% (46/50) | **96.0%** (48/50) |
| Loop battery v1, short context | 0/8 | 0/8 |
| Loop battery v1, 400K-token prefill | 0/8 | **1/8** (`enum` prompt, ttr 0.20) |
| Decode @2K, single stream | 92-94 tok/s | 89.4 tok/s |
| Concurrency, 8x46K warm | 244-249 tok/s | 260.2 tok/s (superseded - see caveat) |
| Concurrency, 12x46K | ~~17.3 tok/s (collapse)~~ | **measurement artifact - retracted, see below** |
| Multi-tool (single / no-call / parallel-3) | 3/3 with schemas wired | 2/3 - parallel-3 answered single-call-then-wait; **0 DSML leaks** |
| Vision (2 synthetic image tests) | pass | pass |

What this says: the graft does exactly what an abliteration should - near-total
refusal removal on harmful archetypes (20/20 -> 2/20) - at the cost of one benign
false-positive (0 -> 1) and one repetition-attractor appearance at 400K depth
(0/8 -> 1/8). No cognitive cost measured: GSM8K 96% >= 92% stock (noise range),
decode within noise.

**Retracted rows (2026-09-17), disclosed in full:** the "12x46K collapse to
17.3 tok/s" and the warm-cache concurrency numbers came from a two-phase harness
that assumed long-context prefix-cache reuse. Server logs show that reuse is
unreliable on this build (only the most recently prefilled context stays
matchable), so those "decode" phases silently re-prefilled - the 17.3 tok/s wall
time was 11x50K tokens at prefill rate, a prefill measurement mislabeled as
decode. Replacement protocol (streaming decode-window, cache-independent) is in
the repo; corrected numbers in the quality section below.

Measurement caveats, stated plainly: 50-problem GSM8K slice (not the full 1319);
20-prompt refusal battery of archetype phrasings (not the Keys refusal32 set);
refusal classified by marker heuristic on the visible reply; loop-prefill uses a
repeated wikitext test split (732 KiB cycled to 400K), not unique prose.

Still not tested: full GSM8K, MMLU/HumanEval, 1M-token recall, real-photo vision,
multi-turn agentic soak, any safety red-team.

## Sources and credits

- Base model: [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- **EXL3 2.0bpw quantization: [diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000)** - all weight shards are diffbot's quant; this repo adds the ablit graft and the MXINT-4 Engram recompression
- 2.9bpw reference quant used for KLD cross-checks: [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
- Abliteration method: **drowzeys' Keys anchored-tensors wo_b graft** (rank-1, layers 10-35, lambda=3.5) - sidecar from [drowzeys' abliteration packs](https://huggingface.co/drowzeys), applied to the 2.0bpw quant
- Runtime: vLLM + vllm_exl3 plugin + ExLlamaV3 kernels; FlashInfer sm_120 kernels under the vision patches

License inherits the DeepSeek V4.1 model license.

## Quality vs the official DeepSeek API and vs 2.9bpw

Measured 2026-09-17. All arms answer the identical prompts; greedy protocol
everywhere; the official arm runs through the DeepSeek API with thinking
disabled for the trajectory capture.

- **GSM8K**: 50 problems from the official test split, greedy, `max_tokens=8000`.
  With a reasoning model the completion budget IS the protocol: at tight budgets
  the official model truncates before answering and scores artificially low
  (measured: 82% at mt=1200 vs 98% at mt=8000 for the same weights).
- **Refusal battery**: 20 harmful-archetype prompts + 10 benign controls, marker
  classifier. The abliterated row's low harmful-refusal count is the point of the
  abliteration, not a defect; benign false-refusals stay at official level.
- **Tool calling**: 3 scenarios (single call, parallel calls, no-call discipline),
  strict OpenAI-format parse + DSML-leak check.
- **Divergence (KL / PPL / top-1)**: on-policy protocol - the official model's own
  greedy trajectories (40 contexts x 128 tokens, byte-gated token streams) are
  scored under each local arm via `prompt_logprobs`. KL is the mean per-token
  `log p_official(t*) - log p_arm(t*)` over the trajectory; PPL is the arm's
  perplexity of the official trajectory. Lower = closer to official.

| Arm | GSM8K % | Harmful refused | Benign refused (false) | Tool calls | KL vs official (nats/tok) | Trajectory PPL | Top-1 agree |
|---|---|---|---|---|---|---|---|
| Official DeepSeek API | 98.0 | 20/20 | 1/10 | 3/3 | reference | reference | reference |
| Mia EXL3 2.9bpw | 96.0 | 20/20 | 1/10 | 3/3 | 0.1238 | 1.132 | 0.984 |
| **This 2.0bpw (abliterated)** | 96.0 | 2/20 | 1/10 | 2/3 | - | - | - |
| This 2.0bpw (stock, earlier session) | 92.0 | 20/20 | 0/10 | - | 0.4389 | 3.275 | 0.771 |

The stock 2.0bpw KL/PPL row is from an earlier session (replay harness,
165K tokens); it is the quantization-only reference point. The abliterated
row is measured on the exact shipped weights.

![gsm8k](charts/gsm8k.png)

![refusal](charts/refusal.png)

![divergence](charts/divergence.png)

![multitool](charts/multitool.png)

