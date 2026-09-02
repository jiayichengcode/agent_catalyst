#!/usr/bin/env bash
# Probe a candidate image for the exact failure that killed the r4 smoke:
# flashinfer JIT-compiles the GDN prefill kernel with -std=c++20, which needs
# nvcc >= CUDA 12. Verdict is written to runs/r4_probe_$TAG.log on NFS.
TAG=${TAG:-unknown}
P=/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst
LOG=$P/runs/r4_probe_$TAG.log
exec > >(tee -a "$LOG") 2>&1
echo "===== image probe [$TAG] $(date -Is) host=$(hostname) ====="
echo "--- nvcc ---"; (nvcc --version || /usr/local/cuda/bin/nvcc --version) 2>&1 | tail -2
echo "--- c++20 compile test ---"
T=$(mktemp -d); echo '__global__ void k(){} int main(){auto x=0;return x;}' > $T/t.cu
if (nvcc -std=c++20 -c $T/t.cu -o $T/t.o) 2>&1 | tail -2; then echo "CXX20: OK"; else echo "CXX20: FAIL"; fi
rm -rf $T
echo "--- vllm load base model (triggers the GDN JIT path) ---"
export TORCH_BLAS_PREFER_CUBLASLT=1 HF_HUB_OFFLINE=1 PYTHONPATH=$P/transfer-unit
export VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1 FLA_TILELANG=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
/mnt/bn/chobits-wx/jiayicheng/envs/qwen35-vllm/bin/python - <<'PY' 2>&1 | tail -20
from vllm import LLM, SamplingParams
llm = LLM(model="/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst/models/Qwen3-4B",
          max_model_len=1024, gpu_memory_utilization=0.85,
          enforce_eager=True, trust_remote_code=True)
o = llm.generate(["def add(a,b):"], SamplingParams(max_tokens=16, temperature=0))
print("GENERATED:", repr(o[0].outputs[0].text[:60]))
print("VLLM_PROBE: PASS")
PY
echo "===== probe [$TAG] done rc=$? $(date -Is) ====="
