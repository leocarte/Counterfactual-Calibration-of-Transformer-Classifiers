#!/usr/bin/env python3
"""Is CCI's CFR advantage over CDA at f=0.25 prevalence, or invariance?

THE PROBLEM
-----------
At n=18, CCI cuts CFR_hard 33% below CDA at a labelled fraction of 0.25
(p_boot=0.0066, 15/18 seeds). But its F1+ there is 0.0095 lower, and a model
that predicts positive less often has fewer predictions available to flip. At
n=6 the two were F1-matched to within 0.0002, which turned out to be
small-sample luck. So the gap has to be accounted for before the 33% can be
quoted.

THE TEST
--------
Per seed, take dF1 = F1(CCI) - F1(CDA) and dCFR = CFR(CCI) - CDA(CFR), then fit

    dCFR = a + b * dF1

over the seeds. The intercept a is the CFR difference extrapolated to dF1 = 0,
i.e. what CCI's advantage would be at exactly matched F1+. If a stays negative
with a bootstrap interval excluding zero, the advantage is not explained by the
F1 gap. The slope b carries the prevalence effect itself: under a pure
prevalence account, seeds with a wider F1 deficit should show a wider CFR
advantage, so b > 0 and a should collapse toward zero.

Reported for every fraction, not just f=0.25, because f=0.1 is a built-in
positive control: CCI's F1+ is 6.7 points lower there, so if the prevalence
signature is real it should be visible in that row and absent at f=0.25.

Bootstrap resamples seeds, matching the paper's unit of analysis. CPU-only.

Usage:
    python3 analysis/direction1/20_semisup_f1_confound.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

BASES = {
    "model_c_deberta_focal_mask": "Model C",
    "cda_deberta_base": "CDA",
    "cci_v2_fixed_js": "CCI",
}
B = 10_000


def split_experiment(exp: str) -> tuple[str, float]:
    for base, label in BASES.items():
        if exp == base:
            return label, 1.0
        if exp.startswith(base + "_lf"):
            return label, float(exp[len(base) + 3:])
    raise ValueError("unrecognised experiment %r" % exp)


def fit(dx: np.ndarray, dy: np.ndarray) -> tuple[float, float]:
    """OLS of dy on dx; returns (intercept, slope)."""
    if np.allclose(dx, dx[0]):
        return float(dy.mean()), 0.0
    b, a = np.polyfit(dx, dy, 1)
    return float(a), float(b)


def main() -> None:
    ap = argparse.ArgumentParser(description="Does the F1 gap explain the CFR advantage?")
    ap.add_argument("--parquet", default=str(_HERE / "results" / "semisup.parquet"))
    ap.add_argument("--rng-seed", type=int, default=20260909)
    args = ap.parse_args()

    src = Path(args.parquet)
    if not src.exists():
        raise SystemExit("no %s; run 12_semisup.py first" % src)

    df = pd.read_parquet(src)
    df[["method", "fraction"]] = df["experiment"].apply(
        lambda e: pd.Series(split_experiment(e)))

    rng = np.random.default_rng(args.rng_seed)
    rows = []

    for frac in sorted(df.fraction.unique()):
        sub = df[df.fraction == frac]
        cci = sub[sub.method == "CCI"].set_index("seed").sort_index()
        cda = sub[sub.method == "CDA"].set_index("seed").sort_index()
        seeds = sorted(set(cci.index) & set(cda.index))
        if len(seeds) < 4:
            print("[skip] f=%g: %d shared seeds, too few to regress" % (frac, len(seeds)))
            continue

        d_f1 = (cci.loc[seeds, "f1_positive"] - cda.loc[seeds, "f1_positive"]).to_numpy()
        d_cfr = (cci.loc[seeds, "cfr_hard"] - cda.loc[seeds, "cfr_hard"]).to_numpy()

        a, b = fit(d_f1, d_cfr)
        r = float(np.corrcoef(d_f1, d_cfr)[0, 1])

        n = len(seeds)
        boots = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, n, n)
            boots[i] = fit(d_f1[idx], d_cfr[idx])[0]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = 2.0 * min((boots >= 0).mean(), (boots <= 0).mean())
        p = float(min(1.0, max(p, 1.0 / B)))

        rows.append({
            "fraction": frac, "n": n,
            "mean_d_f1": d_f1.mean(), "mean_d_cfr": d_cfr.mean(),
            "corr": r, "slope": b, "intercept": a,
            "ci_lo": lo, "ci_hi": hi, "p_intercept": p,
        })

    res = pd.DataFrame(rows)

    print()
    print("Does the F1 gap explain CCI's CFR advantage over CDA?")
    print("dCFR = a + b*dF1 over seeds; a is the advantage at exactly matched F1+")
    print()
    print("%-6s %3s %10s %10s %7s %9s %11s %9s"
          % ("f", "n", "dF1", "dCFR", "corr", "slope b", "intercept a", "p(a)"))
    print("-" * 82)
    for _, r in res.iterrows():
        print("%-6g %3d %+10.4f %+10.5f %+7.2f %+9.4f %+11.5f %9.4f"
              % (r["fraction"], r["n"], r["mean_d_f1"], r["mean_d_cfr"],
                 r["corr"], r["slope"], r["intercept"], r["p_intercept"]))
    print()
    for _, r in res.iterrows():
        print("  f=%-5g intercept 95%% CI [%+.5f, %+.5f]"
              % (r["fraction"], r["ci_lo"], r["ci_hi"]))

    print()
    print("READING")
    print("-" * 82)
    print("Under a pure prevalence account the slope b is positive (a wider F1")
    print("deficit buys a wider CFR advantage) and the intercept a collapses toward")
    print("zero. The advantage survives matching only where a stays negative with a")
    print("CI excluding zero. f=0.1 is the positive control: CCI's F1+ is 6.7 points")
    print("lower there, so the prevalence signature should show up in that row.")

    out = _HERE / "results" / "semisup_f1_confound.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print()
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
