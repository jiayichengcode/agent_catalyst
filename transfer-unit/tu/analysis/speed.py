"""Convergence-SPEED analysis (the dependent variable that matters):
how many B tokens does each condition need to reach a target level τ?

Metrics per run:
  tt(τ)  : B tokens to first reach τ (linear interpolation between ckpts)
  auc    : normalized area under the curve (speed-integrated quality)
  peak   : best pass@1 anywhere on the curve
  t_peak : B tokens at which the peak was first reached

Usage:
  python -m tu.analysis.speed --runs-dir ../runs --pattern "mc_*_s*" \
      --task codegen --taus 0.6,0.7,0.8
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re


def load_curve(run_dir, task):
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
                if d.get(task) is None:
                    continue
                tag = d["ckpt"]
                if tag == "base":
                    x = 0
                elif tag == "final":
                    x = final_btok
                else:
                    x = step2btok.get(int(re.search(r"step(\d+)", tag).group(1)))
                    if x is None:
                        continue
                rows.append((x, d[task]))
    except FileNotFoundError:
        return []
    dedup = {}
    for x, v in sorted(rows):
        dedup[x] = v
    return sorted(dedup.items())


def time_to(rows, tau):
    """B tokens to first reach tau (None if never)."""
    for i, (x, y) in enumerate(rows):
        if y >= tau:
            if i == 0:
                return 0.0
            x0, y0 = rows[i - 1]
            if y == y0:
                return float(x)
            return x0 + (tau - y0) * (x - x0) / (y - y0)
    return None


def auc(rows):
    if len(rows) < 2:
        return 0.0
    s = 0.0
    for i in range(1, len(rows)):
        s += 0.5 * (rows[i][1] + rows[i - 1][1]) * (rows[i][0] - rows[i - 1][0])
    return s / (rows[-1][0] - rows[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="../runs")
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--task", default="codegen")
    ap.add_argument("--taus", default="0.6,0.7,0.8")
    ap.add_argument("--strip", default="")
    args = ap.parse_args()
    taus = [float(t) for t in args.taus.split(",")]

    by = collections.defaultdict(list)
    for rd in sorted(glob.glob(os.path.join(args.runs_dir, args.pattern))):
        if not os.path.isdir(rd):
            continue
        rows = load_curve(rd, args.task)
        if len(rows) < 3:
            continue
        name = os.path.basename(rd)
        cond = re.sub(r"_s\d+$", "", name)
        if args.strip:
            cond = cond.replace(args.strip, "")
        by[cond].append((name, rows))

    hdr = f"{'condition':12s} {'n':>2s} {'AUC':>6s} {'peak':>6s} {'t_peak':>8s}"
    for t in taus:
        hdr += f" {'tt@'+str(t):>10s}"
    print(hdr)
    print("-" * len(hdr))
    for cond in sorted(by):
        runs = by[cond]
        aucs = [auc(r) for _n, r in runs]
        peaks = [max(y for _x, y in r) for _n, r in runs]
        tpeaks = [next(x for x, y in r if y == max(yy for _xx, yy in r))
                  for _n, r in runs]
        line = (f"{cond:12s} {len(runs):2d} {sum(aucs)/len(aucs):6.3f} "
                f"{sum(peaks)/len(peaks):6.3f} {sum(tpeaks)/len(tpeaks):8.0f}")
        for t in taus:
            tts = [time_to(r, t) for _n, r in runs]
            hit = [v for v in tts if v is not None]
            if not hit:
                line += f" {'never':>10s}"
            else:
                mu = sum(hit) / len(hit)
                line += f" {mu:7.0f}({len(hit)})"
        print(line)


if __name__ == "__main__":
    main()
