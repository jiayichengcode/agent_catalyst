#!/usr/bin/env bash
# usage: eval_one.sh <config.json> <gpu>
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
exec "$ENVP/bin/python" -m tu.training.run --config "$1" --eval-only
