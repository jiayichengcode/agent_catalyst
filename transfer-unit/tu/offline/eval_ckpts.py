"""Hot-swap checkpoint evaluator: load the vllm engine ONCE with the base
model, then for each saved SFT checkpoint (vllm-named .pt), load weights via
collective_rpc and run the greedy eval — giving pass@1 vs training-progress
curves at ~2 min per checkpoint instead of ~12 (engine reload).

Usage:
  python -m tu.offline.eval_ckpts --run runs/sftc_b30m60 \
      --tasks sql_query:48,pyfix:24 --out runs/sftc_b30m60/curve.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _load(worker, p):        # executed inside the vllm worker process
    import torch as _t
    sd = _t.load(p, map_location="cpu")
    mr = getattr(worker, "model_runner", None)
    if mr is None:
        mr = worker.worker.model_runner
    mr.model.load_weights(sd.items())
    return len(sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/bn/chobits-wx/jiayicheng/models/Qwen3.5-4B")
    ap.add_argument("--run", required=True)
    ap.add_argument("--tasks", default="sql_query:48")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--include-base", action="store_true")
    ap.add_argument("--delete-after", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "curve.jsonl")
    tasks = [(t.split(":")[0], int(t.split(":")[1]))
             for t in args.tasks.split(",")]

    from vllm import LLM
    from transformers import AutoTokenizer
    from ..training.rollout import ChatFormat, eval_detail
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.85,
              enforce_eager=True, trust_remote_code=True,
              enable_prefix_caching=False, seed=args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    fmt = ChatFormat(tok)
    f = open(out, "a")

    def eval_now(tag):
        row = {"ckpt": tag}
        for task, n in tasks:
            recs = eval_detail(llm, fmt, task, n, seed=args.seed)
            row[task] = sum(r["success"] for r in recs) / max(1, len(recs))
        f.write(json.dumps(row) + "\n")
        f.flush()
        print("[curve]", json.dumps(row))

    if args.include_base:
        eval_now("base")
    ckpts = sorted(glob.glob(os.path.join(args.run, "ckpts", "step*.pt")))
    final = os.path.join(args.run, "ckpt_vllm", "model.safetensors")
    for cp in ckpts:
        llm.collective_rpc(_load, args=(cp,))
        eval_now(os.path.basename(cp))
        if args.delete_after:
            os.remove(cp)
    if os.path.exists(final):
        from safetensors.torch import load_file

        def _load_sf(worker, p):
            from safetensors.torch import load_file as lf
            sd = lf(p)
            mr = getattr(worker, "model_runner", None)
            if mr is None:
                mr = worker.worker.model_runner
            mr.model.load_weights(sd.items())
            return len(sd)
        llm.collective_rpc(_load_sf, args=(final,))
        eval_now("final")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
