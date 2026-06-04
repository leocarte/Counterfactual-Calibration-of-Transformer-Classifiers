#!/usr/bin/env python3
"""Aggregate multi-seed test metrics into mean ± std per model.

Expects directory structure:
    <metrics_dir>/
        model_a_roberta_base_seed42/test_metrics.json
        model_a_roberta_base_seed43/test_metrics.json
        model_a_roberta_base_seed44/test_metrics.json
        model_b_deberta_v3_base_seed42/test_metrics.json
        ...

Usage:
    python scripts/aggregate_multiseed.py \
        --metrics-dir results/metrics_from_rcp \
        --output results/multiseed_summary.csv
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = [
    "macro_f1",
    "f1_positive",
    "f1_negative",
    "precision",
    "recall",
    "pr_auc",
    "ece",
    "mce",
    "brier_score",
    "nll",
]


def collect(metrics_dir: Path):
    """Return {model_base_name: {seed: test_metrics_dict}}."""
    pat = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")
    out = defaultdict(dict)
    for child in sorted(metrics_dir.iterdir()):
        if not child.is_dir():
            continue
        m = pat.match(child.name)
        if not m:
            continue
        tm_path = child / "test_metrics.json"
        if not tm_path.exists():
            print(f"[warn] missing {tm_path}", file=sys.stderr)
            continue
        with tm_path.open() as f:
            out[m["base"]][int(m["seed"])] = json.load(f)
    return out


def summarize(runs_by_model):
    rows = []
    for model, seed_to_metrics in sorted(runs_by_model.items()):
        seeds = sorted(seed_to_metrics.keys())
        row = {"model": model, "n_seeds": len(seeds), "seeds": ",".join(map(str, seeds))}
        for mk in METRICS:
            vals = [seed_to_metrics[s][mk] for s in seeds if mk in seed_to_metrics[s]]
            if not vals:
                continue
            row[f"{mk}_mean"] = mean(vals)
            row[f"{mk}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def print_table(rows):
    # F1+ first per the post-pivot rule (positive-class F1 is the headline
    # metric for our 4.79:1 imbalanced binary task).
    cols = [
        ("f1_positive", "F1+"),
        ("macro_f1", "macro_f1"),
        ("recall", "recall"),
        ("ece", "ECE"),
        ("brier_score", "Brier"),
    ]
    header = f"{'Model':<38} {'seeds':<10} " + " ".join(f"{label:>18}" for _, label in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(k):
            m = r.get(f"{k}_mean")
            s = r.get(f"{k}_std")
            if m is None:
                return "-"
            return f"{m:.4f} ± {s:.4f}"
        cells = " ".join(f"{fmt(k):>18}" for k, _ in cols)
        print(f"{r['model']:<38} {r['seeds']:<10} {cells}")


def save_csv(rows, path: Path):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved CSV: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default="results/metrics_from_rcp")
    ap.add_argument("--output", default="results/multiseed_summary.csv")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.is_dir():
        sys.exit(f"error: metrics dir not found: {metrics_dir}")

    runs = collect(metrics_dir)
    if not runs:
        sys.exit(f"error: no *_seedNN subfolders found under {metrics_dir}")

    rows = summarize(runs)
    print_table(rows)
    save_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()
