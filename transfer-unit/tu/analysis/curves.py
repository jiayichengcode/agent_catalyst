"""Learning-curve analysis: time-to-tau, AUC, condition comparison plots.

Usage:
  python -m tu.analysis.curves --runs runs/trace_noC_s1,runs/trace_F6_s1,... \
      --b-task sql_query --tau 0.5 --out reports/pilot
"""
from __future__ import annotations

import argparse
import json
import os


def load_eval(run_dir: str, task: str):
    by_step = {}
    with open(os.path.join(run_dir, "eval.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if task in d:
                by_step[d["step"]] = d[task]   # dedup relaunch appends: keep last
    xs = sorted(by_step)
    return xs, [by_step[x] for x in xs]


def time_to_tau(xs, ys, tau):
    """First step where the curve reaches tau (linear interp); None if never."""
    for i, y in enumerate(ys):
        if y >= tau:
            if i == 0:
                return xs[0]
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if y1 == y0:
                return x1
            return x0 + (tau - y0) * (x1 - x0) / (y1 - y0)
    return None


def auc(xs, ys):
    if len(xs) < 2:
        return 0.0
    s = 0.0
    for i in range(1, len(xs)):
        s += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return s / (xs[-1] - xs[0])


def summarize(run_dirs, task, taus):
    rows = []
    for rd in run_dirs:
        name = os.path.basename(rd.rstrip("/"))
        xs, ys = load_eval(rd, task)
        row = {"run": name, "n_evals": len(xs),
               "final": ys[-1] if ys else None,
               "best": max(ys) if ys else None,
               "auc": round(auc(xs, ys), 4) if ys else None}
        for t in taus:
            v = time_to_tau(xs, ys, t)
            row[f"tt{t}"] = round(v, 1) if v is not None else None
        rows.append(row)
    return rows


def plot(run_dirs, task, out_png, tau=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rd in run_dirs:
        name = os.path.basename(rd.rstrip("/"))
        xs, ys = load_eval(rd, task)
        ax.plot(xs, ys, marker="o", ms=3, lw=1.5, label=name)
    if tau:
        ax.axhline(tau, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel(f"{task} eval pass@1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--b-task", default="sql_query")
    ap.add_argument("--taus", default="0.3,0.4,0.5,0.6")
    ap.add_argument("--out", default="reports/pilot")
    args = ap.parse_args()
    run_dirs = args.runs.split(",")
    taus = [float(t) for t in args.taus.split(",")]
    os.makedirs(args.out, exist_ok=True)
    rows = summarize(run_dirs, args.b_task, taus)
    print(json.dumps(rows, indent=1))
    with open(os.path.join(args.out, f"summary_{args.b_task}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    plot(run_dirs, args.b_task,
         os.path.join(args.out, f"curves_{args.b_task}.png"), tau=taus[0])


if __name__ == "__main__":
    main()
