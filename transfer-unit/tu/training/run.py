"""Entry point: python -m tu.training.run --config path.json
GPU pinning is external: CUDA_VISIBLE_DEVICES="<vllm_gpu>,<train_gpu>"."""
from __future__ import annotations

import argparse
import json

DEFAULTS = {
    "model_path": "/mnt/bn/chobits-wx/jiayicheng/models/Qwen3.5-4B",
    "out_dir": "/mnt/bn/chobits-wx/jiayicheng/project/agent_catalyst/runs",
    "mixture": {"sql_query": 4, "pyfix": 4, "neutral_format": 2},
    "condition": "F6",          # noC | F0..F6
    "c_task": "pyfix",
    "b_task": "sql_query",
    "seed": 1,
    "steps": 200,
    "group_size": 8,
    "train_pool_size": 400,
    "lr": 1.5e-6,
    "temperature": 1.0,
    "eval_every": 10,
    "eval_sizes": {"sql_query": 48, "pyfix": 24, "neutral_format": 12},
    "logprobs_k": 5,
    "micro_token_budget": 4096,
    "c_oversample": 1,
    "max_model_len": 4096,
    "vllm_gpu_util": 0.85,
    "ckpt_every": 100,
    "span_mask": None,   # {"task": "pyfix", "mode": "named"|"random_match", "spans": ["S_err_react"]}
}


def eval_only(cfg):
    """Task-difficulty calibration: base model, greedy + temp-1 eval with
    per-tier breakdown. No policy model, no training."""
    from vllm import LLM
    from transformers import AutoTokenizer
    from .rollout import ChatFormat, eval_detail
    from ..tracing.features import compute_features
    llm = LLM(model=cfg["model_path"], max_model_len=cfg["max_model_len"],
              gpu_memory_utilization=0.85, enforce_eager=True,
              trust_remote_code=True, enable_prefix_caching=True,
              seed=cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    fmt = ChatFormat(tok)
    import os
    fail_path = os.path.join(cfg["out_dir"], f"{cfg['run_name']}_fails.jsonl")
    ff = open(fail_path, "w")
    for temp in (0.0, 1.0):
        print(f"===== temperature {temp} =====")
        for task, nn in cfg["eval_sizes"].items():
            recs = eval_detail(llm, fmt, task, nn, seed=cfg["seed"],
                               temperature=temp)
            n_dump = 0
            for r in recs:
                if not r["success"] and n_dump < 4:
                    ff.write(json.dumps({
                        "task": task, "tier": r["tier"], "temp": temp,
                        "instance_id": r["instance_id"],
                        "n_actions": r["n_actions"], "truncated": r["truncated"],
                        "turns": [a["text"][-1500:] for a in r["assistant_turns"]],
                        "obs": [o["text"][:600] for o in r["obs_turns"]],
                    }) + "\n")
                    n_dump += 1
            ff.flush()
            by_tier = {}
            agg = {"n": 0, "succ": 0, "err": 0, "errrec": 0, "fmt_err": 0}
            for r in recs:
                f = compute_features(r)
                d = by_tier.setdefault(r["tier"], dict(agg))
                for dd in (d,):
                    dd["n"] += 1
                    dd["succ"] += f["success"]
                    dd["err"] += f["err_event"]
                    dd["errrec"] += f["err_recovery"]
            for t in sorted(by_tier):
                d = by_tier[t]
                print(f"  {task} tier{t}: pass {d['succ']}/{d['n']}"
                      f" err_event {d['err']}/{d['n']}"
                      f" err_recovery {d['errrec']}/{d['n']}")
            tot = sum(d["succ"] for d in by_tier.values())
            n_all = sum(d["n"] for d in by_tier.values())
            print(f"  {task} TOTAL pass@1 {tot}/{n_all} = {tot / max(1, n_all):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--model", help="override model_path (e.g. an SFT ckpt)")
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(json.load(open(args.config)))
    if args.model:
        cfg["model_path"] = args.model
    if args.eval_only:
        eval_only(cfg)
        return
    if cfg["condition"] == "noC":
        # A1-style: C removed from sampling, filler padded to keep batch size
        mix = dict(cfg["mixture"])
        c = cfg["c_task"]
        n_c = mix.pop(c, 0)
        mix["neutral_format"] = mix.get("neutral_format", 0) + n_c
        cfg["mixture"] = mix
    from .grpo import Trainer
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
