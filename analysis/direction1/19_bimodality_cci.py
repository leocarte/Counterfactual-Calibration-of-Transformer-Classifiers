#!/usr/bin/env python3
"""Does the bimodal calibration split appear in CCI, or only in the baseline?

Reviewer #1, Q2, verbatim: "On the bimodal calibration split in Table 6 (seeds
44/45 landing in a poorly-calibrated, high-recall regime); does this bimodality
also appear in CCI-trained checkpoints, or is it specific to the focal-mask
baseline? If specific to the baseline, does that suggest CCI incidentally
stabilizes calibration convergence as a side effect?"

Table 6 documents the split for Model C only, so the paper does not answer this.
Nothing new needs training: the per-seed logits for both methods are already in
cfr_cache, and ECE is recomputed here with the same 15-bin estimator the paper
uses (src.evaluation.metrics.expected_calibration_error), so the Model C column
should reproduce Table 6 and acts as the self-check.

The regime in question is "poorly calibrated AND high recall", so both are
reported per seed. A method shows the split if its two worst-ECE seeds sit far
above the rest rather than in a continuum; the printed gap ratio makes that
visible without asserting a threshold, since with six seeds any cutoff would be
arbitrary.

CPU-only. Usage:
    python3 analysis/direction1/19_bimodality_cci.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

from _utils import (  # noqa: E402
    cfr_cache_root,
    positive_class_z,
    seeds_for,
    sigmoid,
)
from src.evaluation.metrics import expected_calibration_error  # noqa: E402

METHODS = [
    ("model_c_deberta_focal_mask", "Model C"),
    ("cci_v2_fixed_js", "CCI"),
]
TABLE6_SEEDS = (44, 45)  # the two the paper flags for Model C


def per_seed(experiment: str, seed: int, n_bins: int) -> dict | None:
    d = cfr_cache_root() / ("%s_seed%d" % (experiment, seed))
    f = d / "test_originals.parquet"
    if not f.exists():
        return None
    o = pd.read_parquet(f)
    z = positive_class_z(np.stack(
        [o.logit_0.to_numpy(np.float32), o.logit_1.to_numpy(np.float32)], axis=1))
    p = sigmoid(z)
    y = o.true_label.to_numpy(np.int64)
    yhat = (p >= 0.5).astype(np.int64)

    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")

    return {
        "seed": seed,
        "ece": expected_calibration_error(y.tolist(), p.tolist(), n_bins=n_bins),
        "recall": rec,
        "f1_positive": f1,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Is the bimodal calibration split CCI-specific?")
    ap.add_argument("--n-bins", type=int, default=15, help="matches the paper's 15-bin ECE")
    args = ap.parse_args()

    all_rows = []
    for exp, label in METHODS:
        rows = [r for s in seeds_for(exp) if (r := per_seed(exp, s, args.n_bins))]
        if not rows:
            print("[skip] no cache for %s" % exp)
            continue
        df = pd.DataFrame(rows).sort_values("ece", ascending=False).reset_index(drop=True)
        df["method"] = label
        all_rows.append(df)

        print()
        print("%s  (n=%d seeds, %d-bin ECE, sorted worst-calibrated first)"
              % (label, len(df), args.n_bins))
        print("  %-6s %8s %9s %9s  %s" % ("seed", "ECE", "recall", "F1+", ""))
        print("  " + "-" * 46)
        for _, r in df.iterrows():
            mark = "<- flagged in Table 6" if int(r["seed"]) in TABLE6_SEEDS else ""
            print("  %-6d %8.4f %9.4f %9.4f  %s"
                  % (r["seed"], r["ece"], r["recall"], r["f1_positive"], mark))

        worst2 = df.ece.iloc[:2].mean()
        rest = df.ece.iloc[2:].mean()
        ratio = worst2 / rest if rest else float("nan")
        print("  worst two %.4f vs rest %.4f   ratio %.2fx" % (worst2, rest, ratio))
        top2 = set(int(s) for s in df.seed.iloc[:2])
        print("  two worst-calibrated seeds: %s%s"
              % (sorted(top2), "  (= the Table 6 pair)" if top2 == set(TABLE6_SEEDS) else ""))

    if len(all_rows) < 2:
        raise SystemExit("need both methods cached to answer the question")

    out = pd.concat(all_rows, ignore_index=True)
    print()
    print("ANSWER TO REVIEWER 1, Q2")
    print("-" * 62)
    for label in out.method.unique():
        d = out[out.method == label].sort_values("ece", ascending=False)
        r = d.ece.iloc[:2].mean() / d.ece.iloc[2:].mean()
        print("  %-9s spread %.2fx between the two worst seeds and the other four"
              % (label, r))
    print()
    print("  A ratio near 1 means the seeds form a continuum, not two regimes.")
    print("  Read the recall column alongside it: the regime the paper names is")
    print("  poorly calibrated AND high recall, so a high ECE on its own is not")
    print("  the same phenomenon.")

    p = _HERE / "results" / "bimodality_cci.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print()
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
