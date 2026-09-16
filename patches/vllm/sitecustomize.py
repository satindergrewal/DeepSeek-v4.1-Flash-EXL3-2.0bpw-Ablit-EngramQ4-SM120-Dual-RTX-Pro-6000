# Runtime patches for DeepSeek-V4.1-Flash EXL3 on vLLM (sm_120). Mounted over the image's sitecustomize.py.
# Derived from the sfxnz recipe's docker/patch/sitecustomize.py (MIT); sm_120 additions are marked 'sm_120:'.

# install the apport exception handler if available
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

import sys

if "/opt/dsv41-patch" not in sys.path:
    sys.path.insert(0, "/opt/dsv41-patch")

# Load vLLM general plugins in every process (API, EngineCore, workers).
# VLLM_PLUGINS=vllm_exl3 is not enough on this image: EngineCore can resolve
# --quantization exl3 before load_general_plugins() runs.
try:
    from vllm.plugins import load_general_plugins

    load_general_plugins()
except Exception:
    pass

# DSV4.1 mapper picks weight_scale vs weight_scale_inv from
# quant_config.weight_block_size == [32, 32]. Exl3Config keeps that field
# inside non_routed_quantization, so copy it onto the config object.
try:
    from vllm_exl3.exl3 import Exl3Config

    _exl3_from_config = Exl3Config.from_config.__func__

    @classmethod
    def _exl3_from_config_with_block_size(cls, config):
        inst = _exl3_from_config(cls, config)
        nr = getattr(inst, "non_routed_quantization", None) or {}
        wbs = nr.get("weight_block_size") or config.get("weight_block_size")
        if wbs is not None:
            try:
                inst.weight_block_size = list(wbs)
            except AttributeError:
                pass  # vllm-exl3 >= 8f4517e8 exposes weight_block_size as a read-only property
        return inst

    Exl3Config.from_config = _exl3_from_config_with_block_size
except Exception:
    pass

# DSv4 sparse-MLA mixed warmup still dummy-forwards through DeepGEMM paged-MQA
# (block_kv must be 32 or 64) after autotune is disabled. Skip that warmup.
try:
    import vllm.model_executor.warmup.kernel_warmup as _kw

    _kw.deepseek_v4_sparse_mla_attention_warmup = lambda worker: None
    _kw.kernel_warmup = lambda worker: None
except Exception:
    pass

try:
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.worker.worker_base import CompilationTimes

    import os as _os
    # sm_120: the GB10 recipe no-op'd this to dodge a DeepGEMM warmup assert, but it also skips CUDA graph
    # capture (decode ran eager at ~31 tok/s). With the MXFP4 indexer that assert no longer fires.
    if _os.environ.get("DSV41_SKIP_WARMUP", "0") == "1":
        Worker.compile_or_warm_up_model = lambda self: CompilationTimes(0.0, 0.0)
except Exception:
    pass

# FlashInfer SM120 DSV4 decode is compiled only for page_block_size=64.
# Upstream V4.1 hardcodes SWA pages to 32 (DeepGEMM paged-MQA).
try:
    from sm120_page import coerce_swa_block_size
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

    _swa_init = DeepseekV4SWACache.__init__

    def _swa_init_sm120_page(self, *args, **kwargs):
        if "block_size" in kwargs:
            kwargs["block_size"] = coerce_swa_block_size(kwargs["block_size"])
        elif len(args) >= 7:
            args = list(args)
            args[6] = coerce_swa_block_size(args[6])
            args = tuple(args)
        return _swa_init(self, *args, **kwargs)

    DeepseekV4SWACache.__init__ = _swa_init_sm120_page
except Exception:
    pass

# Indexer + compressed MLA share a packed KV group, so they must agree.
# DeepGEMM paged-MQA asserts block_kv in {32, 64}; FlashInfer DSV4 decode
# wants page 64. Upstream reports 128 on SM12, which then has no common size
# if only the indexer is pinned to 64.
try:
    from sm120_page import indexer_kernel_block_sizes
    from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferMLASparseBackend,
    )
    from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend

    _kbs = staticmethod(lambda: list(indexer_kernel_block_sizes()))
    DeepseekV4IndexerBackend.get_supported_kernel_block_sizes = _kbs
    DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes = _kbs
except Exception:
    pass

# --language-model-only still flattens vision_max_n_token onto hf_config, so
# SWA prefill index rows widen to window+1024=1152. SM120 DSV4 decode topk is
# {128,192,256,512,1024}. Do not zero vision_n_layers: VL checkpoints ship
# gate.bias_vl and load_weights KeyErrors without that param.
try:
    from sm120_page import text_only_max_image_tokens
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    _v41_cfg_init = DeepseekV41Config.__init__

    def _v41_cfg_init_text_only(self, *args, **kwargs):
        _v41_cfg_init(self, *args, **kwargs)
        self.vision_max_n_token = text_only_max_image_tokens(
            getattr(self, "vision_max_n_token", 0), True
        )

    DeepseekV41Config.__init__ = _v41_cfg_init_text_only
except Exception:
    pass

# sm_120: allow the MXFP4 sparse-indexer K cache on SM120. vLLM gates it to sm_10x, but DeepGEMM's
# paged MQA logits accept FP4 at block_kv 32/64 on arch 12, and V4.1's ratio-2 compressed layers
# page at block_size/2 = 32 tokens, which the FP8 indexer path rejects on SM120 (needs 64).
try:
    import vllm.v1.attention.backends.mla.indexer as _dsa_idx

    def _dsa_indexer_uses_fp4_sm120(vllm_config):
        kv = vllm_config.attention_config.resolve_indexer_kv_dtype("fp8")
        if kv not in _dsa_idx.DSA_INDEXER_KV_DTYPES:
            raise ValueError(f"indexer_kv_dtype={kv!r} is not supported by the DeepSeek sparse indexer")
        return kv == "mxfp4"

    _dsa_idx.dsa_indexer_uses_fp4 = _dsa_indexer_uses_fp4_sm120
    for _m in ("vllm.models.deepseek_v4_1.attention", "vllm.models.deepseek_v4.attention"):
        try:
            __import__(_m, fromlist=["_"]).dsa_indexer_uses_fp4 = _dsa_indexer_uses_fp4_sm120
        except Exception:
            pass
except Exception:
    pass

# sm_120: optional custom exl3_moe kernel (XMOE_DIR/<XMOE_EXT>, built by kernels/build.sh), e.g. XMOE_EXT=xmoe.
# The plugin resolves exllamav3_ext.exl3_moe at call time, so rebinding the module attribute is enough.
try:
    import os as _os3
    _xm = _os3.environ.get("XMOE_EXT")
    if _xm:
        import sys as _sys3, importlib as _il3, torch as _torch3  # torch first: the extension links libtorch
        _p = _os3.path.join(_os3.environ.get("XMOE_DIR", "/opt/xmoe/build"), _xm)
        if _p not in _sys3.path:
            _sys3.path.insert(0, _p)
        _xmod = _il3.import_module(_xm)
        import exllamav3_ext as _ev3
        _ev3.exl3_moe = _xmod.exl3_moe
        _ev3.exl3_moe_max_concurrency = _xmod.exl3_moe_max_concurrency
except Exception as _e3:
    import sys as _sys4
    print(f"[sitecustomize] XMOE_EXT hook failed: {_e3!r}", file=_sys4.stderr)

# sm_120: hybrid dense MXFP8 (DENSE_HYBRID_M=<rows>, needs --linear-backend marlin). Marlin (W8A16) wins at decode row
# counts, FlashInfer CUTLASS (W8A8, FP8 tensor cores) at prefill row counts (dense was 9% of prefill on FlashInfer vs
# 20% on Marlin). Keep both layouts and route per call by rows. Compilation mode is NONE here, so the branch runs at
# call time: decode CUDA graphs (<=48 rows) capture Marlin, eager prefill takes FlashInfer. Costs one extra FP8 copy
# of the dense weights (~3.3 GiB/rank on V4.1): lower KV_MEM to compensate.
try:
    import os as _os5
    _hm = int(_os5.environ.get("DENSE_HYBRID_M", "0") or 0)
    if _hm > 0:
        import torch as _t5
        from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel as _MK5
        _mk5_pwal = _MK5.process_weights_after_loading
        _mk5_apply = _MK5.apply_weights
        _hyb5_n = [0, 0]

        def _hyb5_pwal(self, layer):
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import MXFP8_BLOCK_SIZE, swizzle_mxfp8_scale
            w = layer.weight.data
            fi = None
            if w.dim() == 2 and w.dtype == _t5.float8_e4m3fn:
                N, K = w.shape
                if K % MXFP8_BLOCK_SIZE == 0 and K >= 128 and N >= 128:
                    s2d = layer.weight_scale.data[:N, :K // MXFP8_BLOCK_SIZE].contiguous()
                    b = getattr(layer, "bias", None)
                    fi = (w.contiguous(), swizzle_mxfp8_scale(s2d, M=N, K=K).contiguous(), N, K,
                          None if b is None else b.data.clone())
            _mk5_pwal(self, layer)  # rebinds layer.weight/weight_scale/bias to Marlin layouts; fi keeps the originals
            layer._hyb5_fi = fi
            _hyb5_n[0 if fi is not None else 1] += 1
            if sum(_hyb5_n) in (1, 50, 100, 200, 400):
                print(f"[sitecustomize] DENSE_HYBRID_M={_hm}: {_hyb5_n[0]} layers hybrid, {_hyb5_n[1]} Marlin-only", file=sys.stderr)

        def _hyb5_apply(self, layer, x, bias=None):
            fi = getattr(layer, "_hyb5_fi", None)
            if fi is not None and x.numel() // x.shape[-1] > _hm:
                from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
                from vllm.utils import flashinfer as _vfi5
                w, s, N, K, b0 = fi
                xs = x.shape
                xq, xsc = mxfp8_e4m3_quantize(x.reshape(-1, K).contiguous(), is_sf_swizzled_layout=True)
                out = _vfi5.mm_mxfp8(xq, w.t(), xsc, s, out_dtype=x.dtype, backend="cutlass")
                if bias is not None:
                    out = out + (b0 if b0 is not None else bias)
                return out.view(*xs[:-1], N)
            return _mk5_apply(self, layer, x, bias)

        _MK5.process_weights_after_loading = _hyb5_pwal
        _MK5.apply_weights = _hyb5_apply
except Exception as _e5:
    print(f"[sitecustomize] DENSE_HYBRID hook failed: {_e5!r}", file=sys.stderr)
