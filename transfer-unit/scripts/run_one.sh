#!/usr/bin/env bash
# usage: run_one.sh <config.json> <gpu_pair e.g. 0,1>
set -euo pipefail
ROOT=/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst
ENVP=/mnt/bn/chobits-wx/jiayicheng/envs/qwen35-vllm
export TORCH_BLAS_PREFER_CUBLASLT=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$2"
export PYTHONPATH="$ROOT/transfer-unit"
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TU_POOL_WHITELIST="${TU_WL:-$ROOT/transfer-unit/data/pool_whitelist.json}"
export TU_EVAL_WHITELIST="${TU_EWL:-}"
# NVML symlink drift fix (2026-08-25: system libnvidia-ml.so.1 flipped to a
# version mismatching the kernel module, breaking pynvml/vllm platform probe)
KV=$(sed -n 's/.*for x86_64 *\([0-9.]*\) .*/\1/p' /proc/driver/nvidia/version 2>/dev/null | head -1)
if [ -n "$KV" ] && [ -e "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.$KV" ]; then
  mkdir -p "$HOME/nvml_fix_$(hostname)"
  ln -sf "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.$KV" "$HOME/nvml_fix_$(hostname)/libnvidia-ml.so.1"
  export LD_LIBRARY_PATH="$HOME/nvml_fix_$(hostname):${LD_LIBRARY_PATH:-}"
fi
exec "$ENVP/bin/python" -m tu.training.run --config "$1"
