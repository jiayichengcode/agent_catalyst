"""AUC on a COMMON B-token grid — robustness check for uneven checkpoint spacing.

`tu.analysis.speed` integrates each curve on whatever B-token points that run
happened to produce. That is fine when arms are sampled at the same density,
but conditions whose auxiliary episodes are shorter pack more sequences per
token budget, take more optimizer steps, and are therefore sampled more densely
on the B-token axis even at identical --save-every. Sparser sampling delays the
detected tt@tau and coarsens the AUC integral, so a density difference can look
like an effect.

This module removes that degree of freedom: every run is linearly interpolated
onto one shared grid spanning [0, min over runs of max b_tok] and the AUC is
recomputed there. Report it next to the native-grid AUC; a conclusion that only
survives on one of the two is a sampling artifact, not a finding.

  python -m tu.analysis.speed_grid --runs-dir ../runs --task codegen \
      --patterns "r3_none_s*,r3_pyf60_s*,r4_cgfail60_s*,r4_pyfail60_s*"
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as st

from .speed import load_curve


def interp(rows, x):
    """pass@1 at B-token x by linear interpolation between checkpoints."""
    if x <= rows[0][0]:
        return rows[0][1]
    for i in range(1, len(rows)):
        x0, y0 = rows[i - 1]
        x1, y1 = rows[i]
        if x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return rows[-1][1]


def grid_auc(rows, xs):
    ys = [interp(rows, x) for x in xs]
    s = sum(0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1]) for i in range(1, len(xs)))
    return s / (xs[-1] - xs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="../runs")
    ap.add_argument("--patterns", required=True, help="comma-separated run globs")
    ap.add_argument("--task", default="codegen")
    ap.add_argument("--points", type=int, default=200)
    args = ap.parse_args()

    arms = {}
    for pat in args.patterns.split(","):
        for rd in sorted(glob.glob(os.path.join(args.runs_dir, pat.strip()))):
            if not os.path.isdir(rd):
                continue
            rows = load_curve(rd, args.task)
            if len(rows) < 3:
                continue
            arm = re.sub(r"_s\d+$", "", os.path.basename(rd))
            arms.setdefault(arm, []).append((os.path.basename(rd), rows))
    if not arms:
        print("no runs matched")
        return

    # shared range: never extrapolate past the shortest run
    xmax = min(r[-1][0] for runs in arms.values() for _n, r in runs)
    xs = [xmax * i / (args.points - 1) for i in range(args.points)]
    print(f"common grid: 0 .. {xmax:.0f} B tokens, {args.points} points\n")
    print(f"{'arm':16s} {'n':>2s} {'AUC_grid':>9s} {'AUC_native':>11s} {'delta':>7s}")
    print("-" * 50)
    from .speed import auc as native_auc
    for arm in sorted(arms):
        runs = arms[arm]
        g = [grid_auc(r, xs) for _n, r in runs]
        nat = [native_auc(r) for _n, r in runs]
        print(f"{arm:16s} {len(runs):2d} {st.mean(g):9.3f} {st.mean(nat):11.3f} "
              f"{st.mean(g) - st.mean(nat):+7.3f}")
    print("\nper-seed AUC on the common grid:")
    for arm in sorted(arms):
        vals = sorted(grid_auc(r, xs) for _n, r in arms[arm])
        print(f"  {arm:16s} " + " ".join(f"{v:.3f}" for v in vals))


if __name__ == "__main__":
    main()
