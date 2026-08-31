#!/usr/bin/env bash
# One-GPU SFT condition pipeline: train -> export -> eval.
# usage: sft_pipeline.sh <run_name> <c_select> <order> <b_tokens> <c_tokens> <gpu>
set -euo pipefail
NAME=$1; CSEL=$2; ORDER=$3; BT=$4; CT=$5; GPU=$6
ROOT=/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst
ENVP=/mnt/bn/chobits-wx/jiayicheng/envs/qwen35-vllm
D=$ROOT/transfer-unit/data
export TORCH_BLAS_PREFER_CUBLASLT=1 HF_HUB_OFFLINE=1 PYTHONPATH=$ROOT/transfer-unit
export VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1 FLA_TILELANG=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 CUDA_VISIBLE_DEVICES=$GPU
KV=$(sed -n 's/.*for x86_64 *\([0-9.]*\) .*/\1/p' /proc/driver/nvidia/version 2>/dev/null | head -1)
if [ -n "$KV" ] && [ -e "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.$KV" ]; then
  mkdir -p "$HOME/nvml_fix_$(hostname)"; ln -sf "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.$KV" "$HOME/nvml_fix_$(hostname)/libnvidia-ml.so.1"
  export LD_LIBRARY_PATH="$HOME/nvml_fix_$(hostname):${LD_LIBRARY_PATH:-}"
fi
PY=$ENVP/bin/python
CARGS=""
if [ "$CSEL" != "none" ]; then CARGS="--c-data $D/donor_pyfix.jsonl --c-tokens $CT"; fi
$PY -m tu.offline.sft --b-data $D/donor_sql.jsonl --c-select "$CSEL" --order "$ORDER" \
    --b-tokens "$BT" $CARGS --run-name "$NAME"
$PY -m tu.offline.export_ckpt --sft $ROOT/runs/$NAME/ckpt \
    --base /mnt/bn/chobits-wx/jiayicheng/models/Qwen3.5-4B --out $ROOT/runs/$NAME/ckpt_vllm
export TU_WL=/nonexistent TU_EWL=
$PY -m tu.training.run --config $ROOT/transfer-unit/configs/sfteval.json --eval-only \
    --model $ROOT/runs/$NAME/ckpt_vllm
echo "PIPELINE_DONE $NAME"
