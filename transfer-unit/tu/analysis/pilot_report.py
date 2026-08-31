"""Generate the pilot results bundle: figures + markdown summary.

Usage: python -m tu.analysis.pilot_report --runs-dir runs \
           --conditions trace_noC_s1,trace_F6_s1,trace_F3_s1,trace_F5_s1 \
           --out reports/pilot_results
"""
from __future__ import annotations

import argparse
import json
import os

from .curves import load_eval, time_to_tau, auc


def read_train_log(run_dir):
    rows = []
    p = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.exists(p):
        return rows
    with open(p) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def c_token_stats(rows, c_task="pyfix"):
    """Per-run: mean C tokens entering gradient per step + share of steps with
    any C gradient."""
    toks = [r["per_task"].get(c_task, {}).get("kept_tok", 0)
            for r in rows if "per_task" in r]
    if not toks:
        return {}
    return {"mean_c_kept_tok": round(sum(toks) / len(toks), 1),
            "steps": len(toks)}


def rollout_err_stats(run_dir, c_task="pyfix", max_rows=200000):
    """err_recovery share among C rollouts, early vs late."""
    p = os.path.join(run_dir, "rollouts.jsonl")
    if not os.path.exists(p):
        return {}
    early, late, all_rows = [], [], []
    with open(p) as f:
        for i, line in enumerate(f):
            if i > max_rows:
                break
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("task") != c_task:
                continue
            all_rows.append(d)
    if not all_rows:
        return {}
    steps = [d.get("step", 0) for d in all_rows]
    mid = (min(steps) + max(steps)) / 2
    early = [d for d in all_rows if d.get("step", 0) <= mid]
    late = [d for d in all_rows if d.get("step", 0) > mid]

    def sh(rows, k):
        return round(sum(r.get(k) or 0 for r in rows) / max(1, len(rows)), 3)
    return {"n_c_rollouts": len(all_rows),
            "errrec_early": sh(early, "err_recovery"),
            "errrec_late": sh(late, "err_recovery"),
            "succ_early": sh(early, "success"), "succ_late": sh(late, "success"),
            "entered_grad_share": sh(all_rows, "entered_gradient")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--b-task", default="sql_query")
    ap.add_argument("--c-task", default="pyfix")
    ap.add_argument("--taus", default="0.75,0.8,0.85")
    ap.add_argument("--out", default="reports/pilot_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    conds = args.conditions.split(",")
    taus = [float(t) for t in args.taus.split(",")]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    md = ["# Pilot phase-1 results\n"]
    md.append(f"| run | evals | B final | B best | B AUC | " +
              " | ".join(f"tt@{t}" for t in taus) +
              " | C kept tok/step | C errrec early→late | C succ early→late |")
    md.append("|" + "---|" * (8 + len(taus)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cond in conds:
        rd = os.path.join(args.runs_dir, cond)
        xs, ys = load_eval(rd, args.b_task)
        cxs, cys = load_eval(rd, args.c_task)
        axes[0].plot(xs, ys, marker="o", ms=3, lw=1.5, label=cond)
        axes[1].plot(cxs, cys, marker="o", ms=3, lw=1.5, label=cond)
        tl = read_train_log(rd)
        ct = c_token_stats(tl, args.c_task)
        rs = rollout_err_stats(rd, args.c_task)
        tts = [time_to_tau(xs, ys, t) for t in taus]
        md.append(f"| {cond} | {len(xs)} | {ys[-1] if ys else '-'} | "
                  f"{max(ys) if ys else '-'} | "
                  f"{round(auc(xs, ys), 3) if ys else '-'} | " +
                  " | ".join(str(round(t, 0)) if t is not None else "never"
                             for t in tts) +
                  f" | {ct.get('mean_c_kept_tok', '-')} | "
                  f"{rs.get('errrec_early', '-')}→{rs.get('errrec_late', '-')} | "
                  f"{rs.get('succ_early', '-')}→{rs.get('succ_late', '-')} |")
    for ax, t in zip(axes, (f"B = {args.b_task} (greedy eval)",
                            f"C = {args.c_task} (greedy eval)")):
        ax.set_title(t)
        ax.set_xlabel("step")
        ax.set_ylabel("pass@1")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.out, "curves.png")
    fig.savefig(png, dpi=150)
    md.append(f"\n![curves](curves.png)\n")

    out_md = os.path.join(args.out, "summary.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out_md} and {png}")


if __name__ == "__main__":
    main()
