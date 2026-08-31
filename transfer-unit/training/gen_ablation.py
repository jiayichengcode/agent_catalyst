#!/usr/bin/env python3
"""Unified condition generator (plan §5): emits config JSONs for a run grid.

Usage:
  python gen_ablation.py --level trace --conditions noC,F6,F3,F5 --seeds 1 \
      --steps 200 --out configs/
Every emitted config differs from the base ONLY in the target variable(s).
"""
import argparse
import json
import os

BASE = {
    "steps": 200,
    "group_size": 12,
    "temperature": 1.0,
    "adv_baseline": "task",
    "mixture": {"sql_query": 6, "pyfix": 4, "neutral_format": 2},
}

SPAN_GRID = {
    "span_err":  {"task": "pyfix", "mode": "named", "spans": ["S_err_react"]},
    "span_ver":  {"task": "pyfix", "mode": "named", "spans": ["S_verify"]},
    "span_rand": {"task": "pyfix", "mode": "random_match", "spans": ["S_err_react"]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=["trace", "span"])
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", default="configs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = []
    for cond in args.conditions.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            cfg = dict(BASE)
            cfg["steps"] = args.steps
            cfg["seed"] = seed
            if args.level == "trace":
                cfg["condition"] = cond
                if cond in ("F3", "F5"):
                    cfg["c_oversample"] = 3
            else:
                cfg["condition"] = "F6"
                cfg["span_mask"] = SPAN_GRID[cond]
            name = f"{args.level}_{cond}_s{seed}"
            cfg["run_name"] = name
            with open(os.path.join(args.out, name + ".json"), "w") as f:
                json.dump(cfg, f, indent=1)
            names.append(name)
    print("\n".join(names))


if __name__ == "__main__":
    main()
