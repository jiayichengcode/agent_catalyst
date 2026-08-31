#!/usr/bin/env bash
# SFT condition with checkpoint curve: train (saving vllm-ready ckpts) ->
# hot-swap eval every checkpoint -> curve.jsonl
# usage: sft_curve_pipeline.sh <run_name> <c_data|-> <c_select> <b_tokens> <c_tokens> <seed> <gpu>
set -euo pipefail
NAME=$1; CDATA=$2; CSEL=$3; BT=$4; CT=$5; SEED=$6; GPU=$7; SE=${8:-2}; REP=${9:-1}; ORD=${10:-mixed}
ROOT=/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst
ENVP=/mnt/bn/chobits-wx/jiayicheng/envs/qwen35-vllm
D=$ROOT/transfer-unit/data
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
$PY -m tu.offline.sft --b-data $D/donor_sql.jsonl --c-select "$CSEL" --order "$ORD" $CARGS \
    --b-tokens "$BT" --seed "$SEED" --save-every $SE --accum 2 --b-repeat $REP --c-skip-b --run-name "$NAME"
$PY -m tu.offline.eval_ckpts --run $ROOT/runs/$NAME --tasks sql_query:48,pyfix:24 \
    --delete-after --include-base
echo "CURVE_DONE $NAME"
