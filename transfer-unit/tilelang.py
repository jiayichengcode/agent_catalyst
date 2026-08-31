# PYTHONPATH shim: the tilelang/tvm_ffi install in the shared qwen35-vllm env
# crashes with AttributeError at import (breaking fla-core's backend probe,
# which only catches ImportError). Raising ImportError here makes the probe
# fail cleanly and fla falls back to its Triton kernels.
raise ImportError("tilelang disabled for transfer-unit runs (broken tvm_ffi)")
