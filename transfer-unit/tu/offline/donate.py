"""Donation-run collector (plan §4 捐赠 trace 库): sample full episodes with
the BASE model and store complete token ids + loss masks + trace features,
as SFT-ready parallel datasets (plan §3 A5 / E6).

Usage (2 GPUs not needed — vllm only):
  python -m tu.offline.donate --task pyfix --n-episodes 1500 --out data/donor_pyfix.jsonl
  python -m tu.offline.donate --task sql_query --n-episodes 3000 --out data/donor_sql.jsonl
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/bn/chobits-wx/jiayicheng/models/Qwen3.5-4B")
    ap.add_argument("--task", required=True)
    ap.add_argument("--n-episodes", type=int, default=1500)
    ap.add_argument("--pool", type=int, default=240)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM
    from transformers import AutoTokenizer
    from ..training.rollout import ChatFormat, run_episodes
    from ..tracing.features import compute_features
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.85,
              enforce_eager=True, trust_remote_code=True,
              enable_prefix_caching=True, seed=args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    fmt = ChatFormat(tok)

    per = max(1, args.n_episodes // args.pool)
    specs = [(args.task, i, "train", f"don/{args.task}/{i}", g)
             for i in range(args.pool) for g in range(per)]
    specs = specs[:args.n_episodes]
    # chunk to keep memory bounded
    n_done = 0
    with open(args.out, "w") as f:
        CH = 512
        for s0 in range(0, len(specs), CH):
            chunk = specs[s0:s0 + CH]
            recs = run_episodes(llm, fmt, chunk, temperature=args.temp,
                                logprobs_k=0, seed=args.seed + s0)
            for r in recs:
                feat = compute_features(r)
                f.write(json.dumps({
                    "task": r["task"], "instance_id": r["instance_id"],
                    "ids": r["ids"], "loss_mask": r["loss_mask"],
                    "assistant_turns": [{k: a[k] for k in
                                         ("start", "end", "turn_idx",
                                          "follows_error_obs", "action_tool",
                                          "fence_char")} | {"text": a["text"][:50]}
                                        for a in r["assistant_turns"]],
                    "feat": {k: feat[k] for k in
                             ("success", "err_event", "err_recovery",
                              "err_giveup", "verify_cnt", "len",
                              "n_gen_tokens")},
                }) + "\n")
            n_done += len(chunk)
            print(f"[donate] {n_done}/{len(specs)} episodes")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
