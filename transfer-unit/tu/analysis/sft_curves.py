"""Aggregate SFT checkpoint curves across seeds per condition.

Reads runs/sftc_<cond>_s<seed>/{curve.jsonl,sft_log.jsonl}; x = optimizer step
(ckpt tag), y = sql pass@1. Emits a per-condition mean±range plot + final table.

Usage: python -m tu.analysis.sft_curves --runs-dir runs --out reports/sft_curves
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re


def load_run(run_dir):
    step2btok = {}
    final_btok = 0
    try:
        with open(os.path.join(run_dir, "sft_log.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                if "ckpt" in d:
                    step2btok[d["ckpt"]] = d["b_tok_seen"]
                final_btok = max(final_btok, d.get("b_tok_seen", 0))
    except FileNotFoundError:
        pass
    rows = []
    try:
        with open(os.path.join(run_dir, "curve.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                tag = d["ckpt"]
                if tag == "base":
                    x = 0
                elif tag == "final":
                    x = final_btok
                else:
                    st = int(re.search(r"step(\d+)", tag).group(1))
                    x = step2btok.get(st)
                    if x is None:
                        continue
                rows.append((int(round(x / 2000.0) * 2000), d.get("sql_query")))
    except FileNotFoundError:
        return []
    dedup = {}
    for x, v in sorted(rows):
        dedup[x] = v
    return sorted(dedup.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--pattern", default="sftc_*_s*")
    ap.add_argument("--out", default="reports/sft_curves")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    by_cond = collections.defaultdict(dict)
    for rd in sorted(glob.glob(os.path.join(args.runs_dir, args.pattern))):
        if not os.path.isdir(rd):
            continue
        name = os.path.basename(rd)
        m = re.match(r"sftc2?_(.+)_s(\d+)$", name)
        if not m:
            continue
        rows = load_run(rd)
        if rows:
            by_cond[m.group(1)][int(m.group(2))] = rows

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10.colors
    table = []
    for ci, (cond, seeds) in enumerate(sorted(by_cond.items())):
        xs_all = sorted({x for rows in seeds.values() for x, _ in rows})
        series = []
        for x in xs_all:
            vals = [dict(rows).get(x) for rows in seeds.values()]
            vals = [v for v in vals if v is not None]
            if vals:
                series.append((x, sum(vals) / len(vals),
                               min(vals), max(vals), len(vals)))
        xs = [s[0] for s in series]
        mu = [s[1] for s in series]
        lo = [s[2] for s in series]
        hi = [s[3] for s in series]
        c = colors[ci % 10]
        ax.plot(xs, mu, marker="o", ms=3, color=c,
                label=f"{cond} (n={len(seeds)})")
        ax.fill_between(xs, lo, hi, color=c, alpha=0.15)
        fin = series[-1]
        table.append((cond, len(seeds), round(fin[1], 3),
                      round(fin[2], 3), round(fin[3], 3)))
    ax.axhline(0.667, color="gray", ls="--", lw=0.8)
    ax.text(0.5, 0.669, "base model", fontsize=7, color="gray")
    ax.set_xlabel("B tokens seen (acceleration axis)")
    ax.set_ylabel("sql_query pass@1 (greedy, n=48)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.out, "curves.png")
    fig.savefig(png, dpi=150)
    print(f"{'cond':12s} seeds final_mean final_min final_max")
    for cond, n, mu, lo, hi in table:
        print(f"{cond:12s} {n:5d} {mu:10.3f} {lo:9.3f} {hi:9.3f}")
    print("saved", png)


if __name__ == "__main__":
    main()
