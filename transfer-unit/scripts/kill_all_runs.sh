#!/usr/bin/env bash
# Kill ALL training runs on this (dedicated) worker, including vllm EngineCore
# children that pkill on the front-end pattern misses (they hold GPU memory).
pkill -f 'tu[.]training[.]run'
sleep 2
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null
done
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
