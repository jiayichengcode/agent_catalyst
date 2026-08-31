"""E4 (L3 observation channel): regress B's eval improvement on the trace
composition of C rollouts entering each gradient step (plan §2).

Pilot version: OLS of ΔB-eval(t→t+w) on per-window C-feature shares
(err_recovery, err_giveup, clean success, verify_cnt mean), controlling for
current B level. With 200 steps / eval-every-10 this is underpowered —
directional evidence only; the interventional E5 is the arbiter.

Usage: python -m tu.analysis.batch_reg --run runs/trace_F6_s1
"""
from __future__ import annotations

import argparse
import json
import os


def load(run_dir, b_task="sql_query", c_task="pyfix"):
    evals = []
    with open(os.path.join(run_dir, "eval.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if b_task in d:
                evals.append((d["step"], d[b_task]))
    feats = {}
    with open(os.path.join(run_dir, "rollouts.jsonl")) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("task") != c_task:
                continue
            feats.setdefault(d.get("step", 0), []).append(d)
    return evals, feats


def window_rows(evals, feats):
    rows = []
    for (s0, y0), (s1, y1) in zip(evals, evals[1:]):
        cs = [d for s in range(s0, s1) for d in feats.get(s, [])]
        if not cs:
            continue
        n = len(cs)
        rows.append({
            "dy": y1 - y0, "y0": y0,
            "sh_errrec": sum(c["err_recovery"] for c in cs) / n,
            "sh_giveup": sum(c["err_giveup"] for c in cs) / n,
            "sh_clean": sum(1 for c in cs
                            if c["success"] and not c["err_event"]) / n,
            "verify_mu": sum(c["verify_cnt"] for c in cs) / n,
        })
    return rows


def ols(rows, xs_keys):
    import numpy as np
    X = np.array([[1.0] + [r[k] for k in xs_keys] for r in rows])
    y = np.array([r["dy"] for r in rows])
    beta, res, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-9
    return dict(zip(["intercept"] + xs_keys, [round(float(b), 4) for b in beta])), \
        round(1 - ss_res / ss_tot, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--b-task", default="sql_query")
    ap.add_argument("--c-task", default="pyfix")
    args = ap.parse_args()
    evals, feats = load(args.run, args.b_task, args.c_task)
    rows = window_rows(evals, feats)
    print(f"{len(rows)} eval windows")
    if len(rows) < 8:
        print("too few windows for regression")
        return
    keys = ["y0", "sh_errrec", "sh_giveup", "sh_clean", "verify_mu"]
    beta, r2 = ols(rows, keys)
    print("OLS dy ~", beta, "R2", r2)


if __name__ == "__main__":
    main()
