#!/usr/bin/env python3
"""Was lambda = 1 the choice dev F1+ would have made?

The Method states the consistency multiplier was "chosen on dev F1+", but every
CCI config carries lambda_cci: 1.0 and no alternative existed in the repo, so
the claim had no record. This aggregates the sweep that supplies one.

Two questions, kept apart on purpose:

  SELECTION  which lambda maximises dev F1+. That is the criterion the paper
             names, and it is the only one that can justify the parenthetical.
             Dev metrics come from the training run, not from the test cache.
  FRONTIER   what each lambda costs and buys on TEST F1+ and CFR_hard. This is
             what bears on Reviewer 1's first concern: if a higher multiplier
             reaches CDA's flip rate at acceptable F1, the gap at lambda = 1 is
             an operating point rather than a limit.

Reporting test numbers as if they were the selection criterion would be
selecting on the test set, so the two blocks never share a verdict.

CPU-only. Usage:
    python3 analysis/direction1/21_lambda_sweep.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

from _utils import cfr_cache_root, positive_class_z, resolve_scratch_root, sigmoid  # noqa: E402

ARMS = [("cci_v2_fixed_js_lam05", 0.5), ("cci_v2_fixed_js", 1.0),
        ("cci_v2_fixed_js_lam2", 2.0), ("cci_v2_fixed_js_lam5", 5.0)]
SEEDS = (42, 43, 44, 45, 46, 47)
CDA = "cda_deberta_base"


B = 10000


def sign_test(deltas: np.ndarray) -> float:
    """Exact two-sided sign test. Zeros are dropped, as is conventional."""
    from math import comb

    nz = deltas[deltas != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    k = int((nz < 0).sum())
    tail = min(k, n - k)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(tail + 1)) / (2 ** n))


def paired_bootstrap(deltas: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided p by resampling the seeds with replacement, floored at 1/B."""
    n = len(deltas)
    stars = np.empty(B)
    for i in range(B):
        stars[i] = deltas[rng.integers(0, n, n)].mean()
    return float(min(1.0, max(2.0 * min((stars >= 0).mean(), (stars <= 0).mean()),
                              1.0 / B)))


def dev_scores(exp: str, seed: int) -> dict | None:
    """Dev metrics at the epoch each run actually kept.

    train_cci.py writes {"train_loss": [...], "dev_metrics": [per-epoch dict]},
    and the per-epoch dict carries macro_f1 and f1_positive. None of the sweep
    configs sets checkpoint_strategy, so trainer.py falls back to
    BestF1Strategy: the kept epoch is the argmax of dev macro-F1. Dev F1+ is
    read AT that epoch rather than at its own argmax, which is a criterion no
    run ever applied. The free maximum is carried alongside so the gap between
    the two is visible instead of assumed away.
    """
    f = (resolve_scratch_root() / "metrics" / ("%s_seed%d" % (exp, seed))
         / "training_history.json")
    if not f.exists():
        return None
    try:
        blob = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    hist = blob.get("dev_metrics") if isinstance(blob, dict) else None
    rows = [h for h in (hist or [])
            if isinstance(h, dict) and "macro_f1" in h and "f1_positive" in h]
    if not rows:
        return None
    k = max(range(len(rows)), key=lambda i: rows[i]["macro_f1"])
    return {
        "dev_f1_pos": float(rows[k]["f1_positive"]),
        "dev_macro": float(rows[k]["macro_f1"]),
        "dev_f1_pos_free": float(max(r["f1_positive"] for r in rows)),
        "kept_epoch": k + 1,
        "n_epochs": len(rows),
    }


def test_metrics(exp: str, seed: int) -> dict | None:
    d = cfr_cache_root() / ("%s_seed%d" % (exp, seed))
    o_p, s_p = d / "test_originals.parquet", d / "test_swap_logits.parquet"
    if not (o_p.exists() and s_p.exists()):
        return None
    o, sw = pd.read_parquet(o_p), pd.read_parquet(s_p)
    p_o = sigmoid(positive_class_z(np.stack(
        [o.logit_0.to_numpy(np.float32), o.logit_1.to_numpy(np.float32)], axis=1)))
    p_s = sigmoid(positive_class_z(np.stack(
        [sw.logit_0.to_numpy(np.float32), sw.logit_1.to_numpy(np.float32)], axis=1)))
    idx = sw.test_idx.to_numpy(np.int64)
    y = o.true_label.to_numpy(np.int64)
    yh = (p_o >= 0.5).astype(np.int64)
    tp = int(((yh == 1) & (y == 1)).sum())
    fp = int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "cfr": float(((p_o[idx] >= 0.5) != (p_s >= 0.5)).mean()),
        "pos_rate": float(yh.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate the lambda sweep.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    rows = []
    for exp, lam in ARMS:
        for sd in args.seeds:
            t = test_metrics(exp, sd)
            if t is None:
                print("[skip] no cache for %s seed %d" % (exp, sd))
                continue
            d = dev_scores(exp, sd) or {}
            rows.append({"lambda": lam, "seed": sd, **d, **t})
    if not rows:
        raise SystemExit("no cached arms found; run the forward passes first")
    df = pd.DataFrame(rows)

    print()
    print("SELECTION  (dev only; the criterion the Method names)")
    print("-" * 62)
    if "dev_f1_pos" not in df.columns or df.dev_f1_pos.isna().all():
        print("  no training_history.json under %s"
              % (resolve_scratch_root() / "metrics"))
        print("  Without it the selection claim cannot be substantiated; say")
        print("  lambda was fixed at 1 rather than selected.")
    else:
        dev = df.dropna(subset=["dev_f1_pos"])
        g = dev.groupby("lambda").agg(
            f1p=("dev_f1_pos", "mean"), f1p_sd=("dev_f1_pos", "std"),
            macro=("dev_macro", "mean"), macro_sd=("dev_macro", "std"),
            free=("dev_f1_pos_free", "mean"), n=("dev_f1_pos", "count"))
        print("  %-8s %16s %16s %6s" % ("lambda", "dev F1+", "dev macro-F1", "n"))
        for lam, r in g.iterrows():
            print("  %-8g %8.4f+-%.4f %8.4f+-%.4f %6d"
                  % (lam, r.f1p, r.f1p_sd, r.macro, r.macro_sd, r.n))
        b_pos, b_mac = g.f1p.idxmax(), g.macro.idxmax()
        print()
        print("  dev F1+      selects lambda = %g%s" % (
            b_pos, "  (matches the reported choice)" if b_pos == 1.0
            else "  (NOT 1)"))
        print("  dev macro-F1 selects lambda = %g%s" % (
            b_mac, "  (matches the reported choice)" if b_mac == 1.0
            else "  (NOT 1)"))
        print()
        print("  The runs set metric_for_best_model: macro_f1 and no")
        print("  checkpoint_strategy, so macro-F1 is what each run's own")
        print("  checkpoint rule used. dev F1+ above is read at that epoch;")
        print("  its free maximum over epochs averages %.4f." % g.free.mean())
        d_ord = list(g.f1p.sort_values(ascending=False).index)
        t_ord = list(df.groupby("lambda").f1.mean().sort_values(ascending=False).index)
        print()
        print("  dev F1+  ranks lambda %s" % " > ".join("%g" % x for x in d_ord))
        print("  test F1+ ranks lambda %s" % " > ".join("%g" % x for x in t_ord))
        print("  A reversed pair of orderings means dev F1+ carries no signal")
        print("  about test F1+ here, so no dev-based selection story survives.")

    print()
    print("FRONTIER  (test, reported but never used to select)")
    print("-" * 62)
    print("  %-8s %14s %14s %10s" % ("lambda", "F1+", "CFR_hard", "pos rate"))
    for lam, g in df.groupby("lambda"):
        print("  %-8g %7.4f+-%.4f %7.4f+-%.4f %9.3f"
              % (lam, g.f1.mean(), g.f1.std(), g.cfr.mean(), g.cfr.std(), g.pos_rate.mean()))

    cda = [test_metrics(CDA, sd) for sd in args.seeds]
    cda = [c for c in cda if c]
    if cda:
        cf = float(np.mean([c["cfr"] for c in cda]))
        f1 = float(np.mean([c["f1"] for c in cda]))
        print("\n  CDA reference: CFR_hard %.4f at F1+ %.4f" % (cf, f1))
        reach = df.groupby("lambda").cfr.mean()
        hit = [l for l, v in reach.items() if v <= cf]
        print("  lambda reaching CDA's flip rate: %s"
              % (", ".join("%g" % l for l in hit) if hit else "none"))
        print("  If one does at acceptable F1+, the gap at lambda = 1 is an")
        print("  operating point we chose, not a structural limit.")

    # The arms share the seed grid, so what carries a p-value is the set of
    # per-seed differences, not the gap between two means. The positive rate
    # travels with lambda, from 0.177 down to 0.147, so it is printed beside
    # CFR: a flip rate that falls only because fewer tweets are called positive
    # is a move along the operating curve, not invariance.
    if cda:
        print()
        print("CONTRAST  (paired over seeds, each lambda against CDA)")
        print("-" * 62)
        by_seed = {}
        for sd in args.seeds:
            m = test_metrics(CDA, sd)
            if m is not None:
                by_seed[sd] = m
        print("  CDA positive rate %.3f"
              % float(np.mean([c["pos_rate"] for c in by_seed.values()])))
        print("  %-7s %-22s %-22s %8s"
              % ("lambda", "dCFR  (sign/boot)", "dF1+  (sign/boot)", "d pos"))
        rng = np.random.default_rng(0)
        for lam in sorted(df["lambda"].unique()):
            sub = df[df["lambda"] == lam].set_index("seed")
            ss = [x for x in sub.index if x in by_seed]
            if len(ss) < 3:
                continue
            dc = np.array([sub.cfr[x] - by_seed[x]["cfr"] for x in ss])
            dfp = np.array([sub.f1[x] - by_seed[x]["f1"] for x in ss])
            dpr = np.array([sub.pos_rate[x] - by_seed[x]["pos_rate"] for x in ss])
            print("  %-7g %+.4f (%.3f/%.4f)  %+.4f (%.3f/%.4f) %+8.3f"
                  % (lam, dc.mean(), sign_test(dc), paired_bootstrap(dc, rng),
                     dfp.mean(), sign_test(dfp), paired_bootstrap(dfp, rng),
                     dpr.mean()))
        print("  Negative dCFR with positive dF1+ is a win on both. Read d pos")
        print("  first: a large negative one means the flip rate was bought by")
        print("  predicting fewer positives, which CFR alone would not show.")

    out = _HERE / "results" / "lambda_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
