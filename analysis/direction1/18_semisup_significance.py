#!/usr/bin/env python3
"""Is CCI's CFR advantage over CDA at low label fractions real, or seed noise?

WHY THIS EXISTS
---------------
12_semisup.py reports means. They show CCI's CFR_hard advantage over CDA
growing monotonically as labels shrink (CDA better by 56% at f=1, CCI better by
13%, 33% and 64% at f=0.5, 0.25, 0.1), with F1+ identical between the two at
f=0.25 and f=0.5, which rules out the "a weaker positive-predictor flips less"
explanation at those two points.

Means over six seeds are not a result. The paper holds itself to a paired
bootstrap over seeds plus an exact sign test, and 13% at f=0.5 is small enough
that it may not clear either. This applies the same machinery so the numbers are
directly comparable to Table 1's.

PROTOCOL, matched to the paper
------------------------------
Paired over seeds, the unit being the seed-level test-set scalar, not individual
examples or pairs. 10^4 resamples, so the bootstrap p floors at 10^-4. The exact
two-sided sign test over six seeds floors at 2*2^-6 = 0.031, which is reported
uncorrected as a direction-consistency check exactly as in the paper.

MULTIPLICITY
------------
The pre-registered prediction concerns the sub-1.0 fractions, so those three
form the family and get Holm-Bonferroni at family-wise alpha = 0.05. f=1.0 is
the already-published cell and is reported for reference, outside the family.
Both corrected and uncorrected p-values are printed; nothing here should be
quoted without saying which.

CPU-only. Usage:
    python3 analysis/direction1/18_semisup_significance.py
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
FAMILY_FRACS = (0.1, 0.25, 0.5)
B = 10_000


def split_experiment(exp: str) -> tuple[str, float]:
    """'cci_v2_fixed_js_lf0.25' -> ('CCI', 0.25); 'cci_v2_fixed_js' -> ('CCI', 1.0)."""
    for base, label in BASES.items():
        if exp == base:
            return label, 1.0
        if exp.startswith(base + "_lf"):
            return label, float(exp[len(base) + 3:])
    raise ValueError("unrecognised experiment %r" % exp)


def sign_test(deltas: np.ndarray) -> float:
    """Exact two-sided sign test. Zeros are dropped, as is conventional."""
    from math import comb

    nz = deltas[deltas != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    k = int((nz < 0).sum())  # CCI lower = better for CFR
    tail = min(k, n - k)
    p = 2.0 * sum(comb(n, i) for i in range(tail + 1)) / (2 ** n)
    return min(1.0, p)


def paired_bootstrap(deltas: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided p by resampling the seeds with replacement, floored at 1/B."""
    n = len(deltas)
    stars = np.empty(B)
    for i in range(B):
        stars[i] = deltas[rng.integers(0, n, n)].mean()
    p = 2.0 * min((stars >= 0).mean(), (stars <= 0).mean())
    return float(min(1.0, max(p, 1.0 / B)))


def holm(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down; threshold alpha/(K-i), stop at first failure."""
    K = len(pvals)
    order = np.argsort(pvals)
    ps = np.asarray(pvals)[order]
    rej = np.zeros(K, bool)
    for i in range(K):
        if ps[i] <= alpha / (K - i):
            rej[i] = True
        else:
            break
    out = np.zeros(K, bool)
    out[order] = rej
    return out.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="Significance for the semi-sup curves.")
    ap.add_argument("--parquet", default=str(_HERE / "results" / "semisup.parquet"))
    ap.add_argument("--rng-seed", type=int, default=20260902)
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
        if len(seeds) < 2:
            print("[skip] f=%g: only %d shared seeds" % (frac, len(seeds)))
            continue

        d_cfr = (cci.loc[seeds, "cfr_hard"] - cda.loc[seeds, "cfr_hard"]).to_numpy()
        d_f1 = (cci.loc[seeds, "f1_positive"] - cda.loc[seeds, "f1_positive"]).to_numpy()

        rows.append({
            "fraction": frac,
            "n_seeds": len(seeds),
            "mean_d_cfr": d_cfr.mean(),
            "mean_d_f1": d_f1.mean(),
            "wins": int((d_cfr < 0).sum()),
            "p_sign": sign_test(d_cfr),
            "p_boot": paired_bootstrap(d_cfr, rng),
            "f1_matched": bool(abs(d_f1.mean()) < 0.002),
        })

    res = pd.DataFrame(rows)

    fam = res[res.fraction.isin(FAMILY_FRACS)].copy()
    fam["rejects_holm"] = holm(fam.p_boot.tolist())
    res = res.merge(fam[["fraction", "rejects_holm"]], on="fraction", how="left")

    print()
    print("CCI vs CDA on CFR_hard, paired over seeds (negative = CCI flips less)")
    print("%d bootstrap resamples; sign test uncorrected; Holm over f in %s"
          % (B, str(FAMILY_FRACS)))
    print()
    print("%-6s %3s %11s %10s %7s %9s %9s %8s  %s"
          % ("f", "n", "dCFR", "dF1", "wins", "p_sign", "p_boot", "Holm", "note"))
    print("-" * 92)
    for _, r in res.iterrows():
        note = "F1 MATCHED" if r["f1_matched"] else (
            "F1 confound" if r["mean_d_f1"] < -0.01 else "")
        holm_s = "-" if pd.isna(r["rejects_holm"]) else (
            "REJECT" if r["rejects_holm"] else "no")
        print("%-6g %3d %+11.5f %+10.4f %4d/%d %9.4f %9.4f %8s  %s"
              % (r["fraction"], r["n_seeds"], r["mean_d_cfr"], r["mean_d_f1"],
                 r["wins"], r["n_seeds"], r["p_sign"], r["p_boot"], holm_s, note))

    print()
    print("READING")
    print("-" * 92)
    print("A cell is quotable only if it clears Holm AND is not marked 'F1 confound':")
    print("at an unmatched F1 the lower flip rate may be prevalence, not invariance.")
    print("The sign test floors at 2*2^-6 = 0.031 with six seeds, so it can never")
    print("clear a Holm threshold here; it is a direction check, not a test.")

    out = _HERE / "results" / "semisup_significance.csv"
    res.to_csv(out, index=False)
    print()
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
