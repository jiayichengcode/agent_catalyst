#!/usr/bin/env bash
# Round-4 smoke: verify the worker environment, then run ONE seed end-to-end.
# Logs to NFS so the devbox can tail it.
P=/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst
LOG=$P/runs/r4_smoke.log
exec > >(tee -a "$LOG") 2>&1
echo "===== r4 smoke $(date -Is) host=$(hostname) ====="
echo "--- gpu ---";  nvidia-smi -L 2>&1 | head -4
echo "--- nfs ---";  ls -d $P/transfer-unit $P/models/Qwen3-4B >/dev/null 2>&1 && echo "nfs OK" || { echo "NFS MISSING - abort"; exit 1; }
echo "--- venv ---"; /mnt/bn/chobits-wx/jiayicheng/envs/qwen35-vllm/bin/python -c \
  "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())" 2>&1 | tail -3
echo "--- data ---"; ls -la $P/transfer-unit/data/donor_codegen.jsonl 2>&1 | awk '{print $5,$9}'
echo "===== launching r4_cgfail60_s1 ====="
bash $P/transfer-unit/scripts/round4_queue.sh 0 cgfail60 1 1
rc=$?
echo "===== smoke finished rc=$rc $(date -Is) ====="
exit $rc
