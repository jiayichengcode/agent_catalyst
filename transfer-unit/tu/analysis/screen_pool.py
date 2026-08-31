"""Offline difficulty screening of the train pool (frozen, condition-shared).

For each train instance, sample K rollouts with the BASE model at the training
temperature and record pass rate; write a whitelist of boundary-band instances
(0 < rate < 1 band configurable). All experimental conditions then share the
same frozen whitelist -> no adaptive-sampling confound.

Usage (on a GPU worker):
  python -m tu.analysis.screen_pool --tasks sql_query,pyfix --n 240 --k 8 \
      --temp 1.2 --lo 0.1 --hi 0.9 --out data/pool_whitelist.json
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/bn/chobits-wx/jiayicheng/models/Qwen3.5-4B")
    ap.add_argument("--tasks", default="sql_query,pyfix")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.2)
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM
    from transformers import AutoTokenizer
    from ..training.rollout import ChatFormat, run_episodes
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.85,
              enforce_eager=True, trust_remote_code=True,
              enable_prefix_caching=True, seed=7)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    fmt = ChatFormat(tok)

    result = {}
    for task in args.tasks.split(","):
        specs = [(task, i, args.split, f"scr/{task}/{i}", g)
                 for i in range(args.n) for g in range(args.k)]
        recs = run_episodes(llm, fmt, specs, temperature=args.temp,
                            logprobs_k=0, seed=99)
        rates = {}
        for r in recs:
            idx = int(r["instance_id"].rsplit("/", 1)[-1])
            rates.setdefault(idx, []).append(r["success"])
        rate_by_idx = {i: sum(v) / len(v) for i, v in rates.items()}
        wl = sorted(i for i, p in rate_by_idx.items()
                    if args.lo <= p <= args.hi)
        result[task] = {"rates": rate_by_idx, "whitelist": wl,
                        "n": args.n, "k": args.k, "temp": args.temp}
        hist = {}
        for p in rate_by_idx.values():
            b = round(p * 8) / 8
            hist[b] = hist.get(b, 0) + 1
        print(f"{task}: {len(wl)}/{args.n} in boundary band "
              f"[{args.lo},{args.hi}]; rate histogram {dict(sorted(hist.items()))}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
