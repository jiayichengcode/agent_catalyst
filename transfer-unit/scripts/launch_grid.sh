#!/usr/bin/env bash
# Launch a grid of runs on this worker, one config per GPU pair.
# usage: launch_grid.sh cfg1.json,cfg2.json,...   (max 4 per 8-GPU worker)
set -euo pipefail
ROOT=/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst
IFS=',' read -ra CFGS <<< "$1"
PAIRS=("0,1" "2,3" "4,5" "6,7")
i=0
for cfg in "${CFGS[@]}"; do
  name=$(basename "$cfg" .json)
  pair=${PAIRS[$i]}
  echo "launching $name on GPUs $pair"
  cd "$ROOT"
  setsid nohup bash transfer-unit/scripts/run_one.sh "$cfg" "$pair" \
      > "runs/${name}.out" 2>&1 < /dev/null &
  i=$((i+1))
done
echo "launched $i runs"
