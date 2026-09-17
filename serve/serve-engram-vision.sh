#!/usr/bin/env bash
# VARIANT (lab 2026-09-14): disk Engram + PIECEWISE CUDA graphs, via the splitting-op
# patch. Run with ENGRAM_DISK=1 EAGER=0. Mounts BOTH patches: engram_graphbreak.py
# (splitting op) over engram.py AND engram_disk_q.py (MXINT-4 reader) over
# engram_disk.py - the q4 pack's tables are 128-wide and the stock reader rejects them. NO large host-RAM allocation - this stays on
# NVMe, so it carries no memory-pressure risk (unlike the pinned-Engram path).
# Serve DeepSeek-V4.1-Flash (EXL3 2.0 bpw routed experts) with vLLM TP2 on 2x RTX PRO 6000 Blackwell (sm_120).
# Engram tables live in pinned host RAM (~95 GiB per rank, read over UVA, CUDA-graph capturable); ENGRAM_DISK=1
# reads them from NVMe instead (needs EAGER=1). Text only. Foreground; Ctrl-C removes the container.
# Requires API_KEY in the environment (clients send it as the Bearer token).
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${API_KEY:?set API_KEY}"
# Downloaded from Hugging Face, this script lives in <pack>/recipe/serve, so the pack is the recipe's parent directory.
if [[ -z ${PACK:-} && -f $REPO/../config.json ]]; then PACK=$(cd "$REPO/.." && pwd); fi
MODELS_DIR=${MODELS_DIR:-$HOME/models}
PACK=${PACK:-$MODELS_DIR/DeepSeek-V4.1-Flash-EXL3-2.0bpw}
[[ $PACK == "$MODELS_DIR"/* ]] || MODELS_DIR=$(dirname "$PACK")   # the pack's parent is mounted at the same path
IMAGE=${IMAGE:-dsv41-flash-exl3-sm120}
NAME=${NAME:-dsv41-flash-exl3}
PORT=${PORT:-8000}
SERVED=${SERVED:-deepseek-v4.1-flash}
CACHE_DIR=${CACHE_DIR:-$REPO/.cache}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-524288}   # V4.1 KV is tiny: 8 GiB holds ~3.8M tokens
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_BATCHED=${MAX_BATCHED:-4096}
KV_MEM=${KV_MEM:-8589934592}             # 8 GiB
NO_CUSTOM_AR=${NO_CUSTOM_AR-1}           # vLLM custom all-reduce fails CUDA graph capture here (custom_all_reduce.cuh:164)
LINEAR_BACKEND=${LINEAR_BACKEND-marlin}  # dense MXFP8: Marlin W8A16 wins decode over FlashInfer W8A8
XMOE_EXT=${XMOE_EXT-xmoe}                # kernels/build/xmoe (kernels/build.sh); empty = stock ExLlamaV3 exl3_moe
UTIL=${UTIL:-0.92}
EAGER=${EAGER:-0}                        # 1 = no CUDA graphs
SPEC=${SPEC:-dspark}                     # none = no speculative decoding
FI_ARCH=${FI_ARCH:-12.0f}
THINKING=${THINKING:-default}            # default = model decides (on, effort 50); off = force thinking off
INDEXER_KV=${INDEXER_KV-mxfp4}           # V4.1 native FP4 indexer cache (sm_120 needs it for 32-token compressed pages)
FI_PATCH=${FI_PATCH-1}                   # 1 = JIT-build FlashInfer sparse_mla_sm120 prefill with 32-token extra pages
[[ -f $PACK/config.json ]] || { echo "no config.json in PACK=$PACK"; exit 1; }
[[ -z ${XMOE_EXT:-} || -d $REPO/kernels/build/$XMOE_EXT ]] || { echo "kernels/build/$XMOE_EXT missing: run kernels/build.sh (or XMOE_EXT= for the stock kernel)"; exit 1; }
mkdir -p "$CACHE_DIR"
extra=()
CAPTURE_SIZES=${CAPTURE_SIZES:-"1 2 4 6 8 12 16 24 32 48"}   # 8 seqs x (1 + 5 DSpark tokens) = 48
# VARIANT (lab 2026-09-14): PIECEWISE, not FULL_DECODE_ONLY. FULL captures the whole
# decode forward, which still contains the Engram NVMe gather and fails capture. PIECEWISE
# breaks the graph at splitting ops, and engram_graphbreak.py registers the gather as one,
# so its host code runs eagerly between graph segments.
[[ $EAGER == 1 ]] && extra+=(--enforce-eager) || # use_inductor_graph_partition=true is LOAD-BEARING. Without it, set_splitting_ops_for_v1
# hits `if pass_config.fuse_attn_quant and not use_inductor_graph_partition:` and calls
# set_splitting_ops_for_attn_fusion(), which sets splitting_ops=[] AND forces
# cudagraph_mode=FULL. An empty split list makes the "piecewise" graph a single segment
# spanning the whole forward, so the Engram gather is captured and capture fails.
# With it true, the else branch runs and splitting_ops = the default attention ops, which
# our engram_disk_lookup is appended to.
extra+=(--compilation-config '{"cudagraph_mode":"PIECEWISE","custom_ops":["all"],"use_inductor_graph_partition":true}' --cudagraph-capture-sizes $CAPTURE_SIZES)
[[ -n ${NO_CUSTOM_AR:-} ]] && extra+=(--disable-custom-all-reduce)
[[ -n ${INDEXER_KV:-} ]] && extra+=(--attention-config "{\"indexer_kv_dtype\":\"$INDEXER_KV\"}")
[[ -n ${LINEAR_BACKEND:-} ]] && extra+=(--linear-backend "$LINEAR_BACKEND")
[[ $THINKING == off ]] && extra+=(--default-chat-template-kwargs '{"thinking":false,"reasoning_effort":"low"}')
[[ $SPEC == dspark ]] && extra+=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}')
# pass-through into the container: VLLM_EXL3_*, NCCL_PROTO/ALGO, DENSE_HYBRID_M (opt-in, see README), XMOE_* tuning
PASS_ENV=$(env | grep -E "^(VLLM_EXL3_[A-Z0-9_]+|NCCL_PROTO|NCCL_ALGO|DENSE_HYBRID_M|XMOE_[A-Z_]+)=" | sed "s/^/-e /" | tr "\n" " ")
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1; }
trap 'cleanup; exit 130' INT TERM
docker run --rm --name "$NAME" --gpus all --network host --ipc host --shm-size 32g ${DOCKER_MEM:+--memory $DOCKER_MEM --memory-swap $DOCKER_MEM} \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -v "$MODELS_DIR":"$MODELS_DIR":ro -v "$CACHE_DIR":/root/.cache \
  -v "$REPO/patches/vllm/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro \
  -v "$REPO/patches/vllm/engram_graphbreak.py":/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py:ro \
  -v "$REPO/patches/vllm/engram_disk_q.py":/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram_disk.py:ro \
  ${FI_PATCH:+-v "$REPO/patches/flashinfer/sparse_mla_sm120_prefill.cu":/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/sparse_mla_sm120_prefill.cu:ro} \
  ${FI_PATCH:+-v "$REPO/patches/flashinfer/empty-aot":/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120:ro} \
  -v "$REPO/patches/flashinfer/sparse_mla_sm120_decode_dsv4.cu":/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/sparse_mla_sm120_decode_dsv4.cu:ro \
  -v "$REPO/patches/flashinfer/_sparse_mla_sm120.py":/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_sparse_mla_sm120.py:ro \
  ${XMOE_EXT:+-e XMOE_EXT=$XMOE_EXT -e XMOE_DIR=/opt/xmoe/build -v "$REPO/kernels/build":/opt/xmoe/build:ro} \
  -e TORCH_CUDA_ARCH_LIST=12.0a -e FLASHINFER_CUDA_ARCH_LIST="$FI_ARCH" -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0} \
  ${NO_CUSTOM_AR:+-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True} \
  -e VLLM_PLUGINS=vllm_exl3 -e DSV41_ENGRAM_DEDUP=${DSV41_ENGRAM_DEDUP:-0} -e DSV41_ENGRAM_DISK=${ENGRAM_DISK:-0} -e DSV41_ENGRAM_DISK_THREADS=${ENGRAM_THREADS:-32} -e DSV41_ENGRAM_BLOCK=${ENGRAM_BLOCK:-32} \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-0} \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN \
  ${MOUNTS_EXTRA:-} \
  $PASS_ENV \
  --entrypoint vllm "$IMAGE" serve "$PACK" \
  --tensor-parallel-size 2 --host 0.0.0.0 --port "$PORT" --api-key "$API_KEY" \
  --max-model-len "$MAX_MODEL_LEN" --kv-cache-dtype fp8 --kv-cache-memory "$KV_MEM" \
  --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
  ${CUDAGRAPH_SIZES:+--cudagraph-capture-sizes $CUDAGRAPH_SIZES} \
  --kernel-config '{"enable_flashinfer_autotune":false,"enable_jit_warmup":false}' \
  --block-size 64 --quantization exl3 --hf-overrides '{"vision_max_n_token": 1024, "vision_patch_size": 14, "vision_downsample_ratio": 3}'  --limit-mm-per-prompt "{\"image\": 4}" \
  --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --served-model-name "$SERVED" --trust-remote-code "${extra[@]}" ${EXTRA_ARGS:-}
rc=$?; cleanup; exit $rc
