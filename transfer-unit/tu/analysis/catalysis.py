"""Catalysis-signature readouts: does C's co-presence amplify B's OWN
learning signal (vs merely transferring skill)?

Per-step B metrics from rollouts.jsonl:
  - mixed-group fraction (groups with 0 < succ < G) -> B's usable GRPO signal
  - share of B rollouts with nonzero advantage (adv_mag > 0)
  - B success rate at sampling temperature
  - mean policy entropy on B tokens (top-k approx)
True catalysis predicts: mixed-training curves of these exceed B-alone (noC)
and sequential C->B at matched steps of the B phase.

Usage: python -m tu.analysis.catalysis --runs runs/hard2_noC_s1,runs/hard2_F6_s1 \
    --b-task sql_query --window 10
"""
from __future__ import annotations

import argparse
import collections
import json
import os


def b_signal(run_dir, b_task="sql_query", window=10):
    groups = collections.defaultdict(list)
    with open(os.path.join(run_dir, "rollouts.jsonl")) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("task") != b_task:
                continue
            groups[(d.get("step", 0), d.get("group_id"))].append(d)
    by_step = collections.defaultdict(
        lambda: {"n": 0, "succ": 0, "mixed": 0, "groups": 0, "nz_adv": 0,
                 "ent_sum": 0.0, "ent_n": 0})
    for (step, gid), rows in groups.items():
        s = by_step[step]
        s["groups"] += 1
        ss = sum(r["success"] for r in rows)
        s["mixed"] += int(0 < ss < len(rows))
        for r in rows:
            s["n"] += 1
            s["succ"] += r["success"]
            s["nz_adv"] += int((r.get("adv_mag") or 0) > 1e-9)
            if r.get("ent") is not None:
                s["ent_sum"] += r["ent"]
                s["ent_n"] += 1
    rows = []
    for step in sorted(by_step):
        s = by_step[step]
        rows.append({
            "step": step,
            "succ": s["succ"] / max(1, s["n"]),
            "mixed_frac": s["mixed"] / max(1, s["groups"]),
            "nz_adv_frac": s["nz_adv"] / max(1, s["n"]),
            "ent": s["ent_sum"] / max(1, s["ent_n"]),
        })
    # window-average
    out = []
    for i in range(0, len(rows), window):
        w = rows[i:i + window]
        out.append({
            "step0": w[0]["step"],
            "succ": round(sum(r["succ"] for r in w) / len(w), 3),
            "mixed": round(sum(r["mixed_frac"] for r in w) / len(w), 3),
            "nz_adv": round(sum(r["nz_adv_frac"] for r in w) / len(w), 3),
            "ent": round(sum(r["ent"] for r in w) / len(w), 3),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--b-task", default="sql_query")
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()
    for rd in args.runs.split(","):
        name = os.path.basename(rd.rstrip("/"))
        print(f"== {name} (B self-signal per {args.window}-step window)")
        for w in b_signal(rd, args.b_task, args.window):
            print(f"  step {w['step0']:4d}: train-succ {w['succ']:.3f} "
                  f"mixed-groups {w['mixed']:.3f} nz-adv {w['nz_adv']:.3f} "
                  f"entropy {w['ent']:.3f}")


if __name__ == "__main__":
    main()
