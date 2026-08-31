"""Publication-style catalyst figure: (A) smoothed acceleration curves for the
four story-carrying conditions; (B) endpoint dot plot for the full factorial.

Usage: python -m tu.analysis.final_figure --runs-dir ../runs --out reports/fig_catalyst.png
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re


def load_run(run_dir):
    step2btok, final_btok = {}, 0
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
                    x = step2btok.get(int(re.search(r"step(\d+)", tag).group(1)))
                    if x is None:
                        continue
                rows.append((x, d.get("sql_query")))
    except FileNotFoundError:
        return []
    dedup = {}
    for x, v in sorted(rows):
        dedup[x] = v
    return sorted(dedup.items())


def smooth(series, w=3):
    out = [series[0]]          # anchor the base point exactly
    for i in range(1, len(series)):
        lo = max(0, i - w // 2)
        hi = min(len(series), i + w // 2 + 1)
        xs = [series[j][0] for j in range(lo, hi)]
        ys = [series[j][1] for j in range(lo, hi)]
        out.append((series[i][0], sum(ys) / len(ys)))
    return out


def collect(runs_dir, min_cover=0.95):
    by_cond = collections.defaultdict(dict)
    for rd in sorted(glob.glob(os.path.join(runs_dir, "sftc2_*_s*"))):
        if not os.path.isdir(rd):
            continue
        m = re.match(r"sftc2_(.+)_s(\d+)$", os.path.basename(rd))
        if not m:
            continue
        rows = load_run(rd)
        if len(rows) < 3:
            continue
        fb = 0
        try:
            with open(os.path.join(rd, "sft_log.jsonl")) as f:
                for line in f:
                    d = json.loads(line)
                    fb = max(fb, d.get("b_tok_seen", 0))
        except FileNotFoundError:
            pass
        if fb and rows[-1][0] < min_cover * fb:
            continue          # partial coverage: exclude from aggregation
        by_cond[m.group(1)][int(m.group(2))] = rows
    return by_cond


def band(seed_series, n_grid=40):
    """Interpolate each smoothed seed curve onto a shared token grid."""
    xmax = min(max(x for x, _ in s) for s in seed_series)
    grid = [xmax * i / (n_grid - 1) for i in range(n_grid)]
    curves = []
    for s in seed_series:
        xs = [p[0] for p in s]
        ys = [p[1] for p in s]
        cy = []
        for g in grid:
            j = 0
            while j + 1 < len(xs) and xs[j + 1] <= g:
                j += 1
            if j + 1 >= len(xs):
                cy.append(ys[-1])
            elif xs[j + 1] == xs[j]:
                cy.append(ys[j])
            else:
                t = (g - xs[j]) / (xs[j + 1] - xs[j])
                cy.append(ys[j] * (1 - t) + ys[j + 1] * t)
        curves.append(cy)
    mu = [sum(c[i] for c in curves) / len(curves) for i in range(n_grid)]
    lo = [min(c[i] for c in curves) for i in range(n_grid)]
    hi = [max(c[i] for c in curves) for i in range(n_grid)]
    return grid, mu, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="../runs")
    ap.add_argument("--out", default="reports/fig_catalyst.png")
    args = ap.parse_args()
    by_cond = collect(args.runs_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    BASE = 0.646

    # ---- Panel A: the four story conditions, smoothed ----
    styleA = {
        "none":  dict(color="#555555", label="B-only 30k (1 epoch)", lw=2.2),
        "rep3":  dict(color="#8c564b", label="B-only 30k ×3 epochs (token-matched)", lw=2.0, ls="--"),
        "py60":  dict(color="#d62728", label="+ related traces 60k (same total tokens as ×3)", lw=2.6),
        "neu230": dict(color="#9467bd", label="+ unrelated data 230k (inert)", lw=1.8),
        "bfail": dict(color="#1f77b4", label="+ own failures 230k (poison)", lw=2.0),
    }
    for cond, st in styleA.items():
        seeds = by_cond.get(cond, {})
        if not seeds:
            continue
        series = [smooth(s, 3) for s in seeds.values()]
        grid, mu, lo, hi = band(series)
        gx = [g / 1000 for g in grid]
        axA.plot(gx, mu, **st)
        axA.fill_between(gx, lo, hi, color=st["color"], alpha=0.13)
        axA.annotate(f"{mu[-1]:.2f}", (gx[-1], mu[-1]),
                     textcoords="offset points", xytext=(6, -3),
                     fontsize=9, color=st["color"], fontweight="bold")
    axA.axhline(BASE, color="gray", ls=":", lw=1)
    axA.text(0.3, BASE + 0.006, "base model (no SFT)", fontsize=8, color="gray")
    axA.set_xlabel("target-task (B) tokens trained on, ×1000")
    axA.set_ylabel("target-task pass@1 (greedy, n=48)")
    axA.set_title("A · Same B data, different additives  (mean ± seed range, n=3)",
                  fontsize=11, loc="left")
    axA.legend(fontsize=9, loc="lower left", framealpha=0.9)
    axA.grid(alpha=0.25)
    axA.set_ylim(0.28, 1.0)

    # ---- Panel B: endpoint dot plot, full factorial ----
    order = [
        ("bfail",     "own failures 230k",       "#1f77b4"),
        ("rep3",      "B-only ×3 epochs (90k total)", "#8c564b"),
        ("b90k",      "B-only 90k unique (more own data)", "#3b3b3b"),
        ("neu230",    "unrelated 230k",          "#9467bd"),
        ("none",      "B-only",                  "#555555"),
        ("py230",     "related 230k (overdose)", "#f4a4c0"),
        ("m60",       "far-domain 60k",          "#ff9d5c"),
        ("m230",      "far-domain 230k",         "#e07020"),
        ("py60",      "related 60k ★",           "#d62728"),
        ("errrec230", "err-recovery traces 230k", "#2ca02c"),
    ]
    ys, labels = [], []
    for i, (cond, label, color) in enumerate(order):
        seeds = by_cond.get(cond, {})
        if not seeds:
            continue
        finals = [s[-1][1] for s in seeds.values()]
        mu = sum(finals) / len(finals)
        axB.plot([min(finals), max(finals)], [i, i], color=color, lw=2.5, alpha=0.55)
        axB.plot(mu, i, "o", color=color, ms=9)
        for f in finals:
            axB.plot(f, i, "|", color=color, ms=11, mew=1.6)
        ys.append(i)
        labels.append(f"{label}  ({len(finals)} seeds)")
    axB.axvline(BASE, color="gray", ls=":", lw=1)
    nb = [s[-1][1] for s in by_cond.get("none", {}).values()]
    if nb:
        axB.axvspan(min(nb), max(nb), color="#555555", alpha=0.10)
    axB.set_yticks(ys)
    axB.set_yticklabels(labels, fontsize=9)
    axB.set_xlabel("final pass@1 (equal 90k-total-token budget where applicable)")
    axB.set_title("B · Endpoint by additive: dose, domain, polarity",
                  fontsize=11, loc="left")
    axB.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    print("saved", args.out)


if __name__ == "__main__":
    main()
