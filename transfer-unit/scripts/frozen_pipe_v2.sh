#!/usr/bin/env bash
# Frozen pipeline v2 (math->coding grid). NEVER edit while runs are live —
# copy to frozen_pipe_v3.sh for changes.
# usage: frozen_pipe_v2.sh NAME BDATA CDATA CSEL CT SEED GPU EVALTASKS [BT] [SE] [ORD]
set -euo pipefail
NAME=$1; BDATA=$2; CDATA=$3; CSEL=$4; CT=$5; SEED=$6; GPU=$7; EVT=$8
BT=${9:-30000}; SE=${10:-2}; ORD=${11:-mixed}
ROOT=/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst
ENVP=/mnt/bn/chobits-wx/jiayicheng/envs/qwen35-vllm
export TORCH_BLAS_PREFER_CUBLASLT=1 HF_HUB_OFFLINE=1 PYTHONPATH=$ROOT/transfer-unit
export VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1 FLA_TILELANG=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 CUDA_VISIBLE_DEVICES=$GPU
export TU_WL=/nonexistent TU_EWL=
KV=$(sed -n 's/.*for x86_64 *\([0-9.]*\) .*/\1/p' /proc/driver/nvidia/version 2>/dev/null | head -1)
if [ -n "$KV" ] && [ -e "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.$KV" ]; then
  mkdir -p "$HOME/nvml_fix_$(hostname)"
  ln -sf "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.$KV" "$HOME/nvml_fix_$(hostname)/libnvidia-ml.so.1"
  export LD_LIBRARY_PATH="$HOME/nvml_fix_$(hostname):${LD_LIBRARY_PATH:-}"
fi
PY=$ENVP/bin/python
CARGS=""
if [ "$CSEL" != "none" ]; then CARGS="--c-data $CDATA --c-tokens $CT"; fi
$PY -m tu.offline.sft --b-data "$BDATA" --c-select "$CSEL" --order "$ORD" $CARGS \
    --b-tokens "$BT" --seed "$SEED" --save-every "$SE" --accum 2 --c-skip-b \
    --run-name "$NAME"
$PY -m tu.offline.eval_ckpts --run $ROOT/runs/$NAME --tasks "$EVT" \
    --delete-after --include-base
echo "CURVE_DONE $NAME"
