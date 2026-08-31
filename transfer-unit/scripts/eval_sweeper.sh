#!/usr/bin/env bash
# Serial checkpoint-eval sweeper: repeatedly finds sftc2_* runs that have
# saved ckpts but no completed curve, and evaluates them one at a time.
# mkdir locks make multiple sweepers safe. usage: eval_sweeper.sh <gpu>
GPU=$1
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
idle=0
while [ $idle -lt 20 ]; do
  found=0
  for r in $ROOT/runs/sftc2_*; do
    [ -d "$r/ckpts" ] || continue
    n=$(ls "$r"/ckpts/*.pt 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && continue
    rows=$(wc -l < "$r/curve.jsonl" 2>/dev/null || echo 0)
    [ "$rows" -ge 3 ] && continue
    mkdir "$r/.eval_lock" 2>/dev/null || continue
    echo "[sweeper$GPU] evaluating $(basename $r) ($n ckpts)"
    $PY -m tu.offline.eval_ckpts --run "$r" --tasks sql_query:48 \
        --delete-after --include-base && echo "CURVE_DONE $(basename $r)"
    rmdir "$r/.eval_lock" 2>/dev/null
    found=1
  done
  [ "$found" -eq 0 ] && idle=$((idle+1)) || idle=0
  sleep 60
done
echo "SWEEPER_$GPU_EXIT (idle)"
