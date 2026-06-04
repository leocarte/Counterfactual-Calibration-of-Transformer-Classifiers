#!/usr/bin/env python3
"""Aggregate cross-dataset (HateXplain-Jewish, ToxiGen-Jewish) metrics across
seeds for every model cell that has a `full_evaluation.json` under a root.

Produces:
  - stdout table: AUROC + Macro-F1 + F1+ + PR-AUC + ECE per (cell, dataset)
  - cross_dataset_aggregate.json: same numbers, machine-readable, consumable
    by the dumbbell figure script.

AUROC: computed approximately via Mann-Whitney U on the 15-bin reliability
histogram stored in each `full_evaluation.json`. With 15 bins, the
approximation is typically accurate to ~0.005 vs the true AUROC computed from
per-example probabilities. Flag this in the figure caption.

Run on a laptop (defaults to `results/metrics_from_rcp/`):
    python scripts/aggregate_cross_dataset.py

Run directly on the RCP scratch volume (no rsync needed):
    python scripts/aggregate_cross_dataset.py \
        --root /scratch/${USER}/antisemitism-results \
        --output ./cross_dataset_aggregate.json

Or with the jumphost-side mount:
    python scripts/aggregate_cross_dataset.py \
        --root /mnt/course-ee-559/rcp-caas-ee-559-g39/scratch-g39/${USER}/antisemitism-results \
        --output ./cross_dataset_aggregate.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CROSS_ROOT = ROOT / "results" / "metrics_from_rcp"
DEFAULT_OUT_JSON = ROOT / "results" / "figures" / "cross_dataset_aggregate.json"

# Pattern: cross_dataset_<exp>_seed<N>/<exp>/full_evaluation.json
#   OR    cross_dataset_seed<N>/<exp>/full_evaluation.json   (older layout)
SEED_RE = re.compile(r"_seed(\d+)$")


def auroc_from_bins(bin_count: list, bin_pos_rate: list) -> float:
    """Approximate AUROC via Mann-Whitney U on a binned score histogram.

    bin_count[i]    : number of examples in bin i (bins sorted ascending by score)
    bin_pos_rate[i] : empirical positive rate among the bin_count[i] examples
                      (may be None if the bin is empty)

    Returns the probability that a random positive scores higher than a random
    negative, with same-bin ties counted as 1/2.
    """
    pos: list = []
    neg: list = []
    for n, p in zip(bin_count, bin_pos_rate):
        if not n or p is None:
            pos.append(0.0)
            neg.append(0.0)
        else:
            pos.append(n * p)
            neg.append(n * (1.0 - p))
    n_pos = sum(pos)
    n_neg = sum(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    wins = 0.0
    for i in range(len(bin_count)):
        for j in range(i):
            wins += pos[i] * neg[j]
    ties = sum(p * q for p, q in zip(pos, neg))
    return (wins + 0.5 * ties) / (n_pos * n_neg)


def discover_cells(cross_root: Path) -> dict:
    """Walk `cross_root` and group every full_evaluation.json by
    (cell_base, seed). Returns: {cell_base: [(seed, path), ...]}"""
    cells: dict = {}
    if not cross_root.exists():
        return cells

    for fpath in sorted(cross_root.rglob("full_evaluation.json")):
        # Find the closest ancestor matching '*_seed<N>' to identify cell + seed.
        seed = None
        cell_base = None
        for parent in fpath.parents:
            m = SEED_RE.search(parent.name)
            if m:
                seed = int(m.group(1))
                # Strip the seed suffix to get the cell label.
                cell_base = parent.name[: m.start()]
                # If we're in the "cross_dataset_seed<N>" layout, the subfolder
                # name (parent of full_evaluation.json) IS the cell label.
                if cell_base == "cross_dataset" and fpath.parent.name != parent.name:
                    cell_base = fpath.parent.name
                break
        if seed is None or cell_base is None:
            # Not a cross_dataset_*/seed structure — skip.
            continue
        # Normalize: strip a leading "cross_dataset_" if present.
        if cell_base.startswith("cross_dataset_"):
            cell_base = cell_base[len("cross_dataset_"):]
        cells.setdefault(cell_base, []).append((seed, fpath))

    # Sort each cell's runs by seed.
    for c in cells:
        cells[c].sort(key=lambda x: x[0])
    return cells


def metrics_per_seed(fpath: Path) -> dict:
    """Extract (HX, TG) metric rows from a single full_evaluation.json."""
    with open(fpath) as f:
        d = json.load(f)
    out = {}
    for tag, key in [("HateXplain", "cross_dataset_hatexplain"),
                     ("ToxiGen", "cross_dataset_toxigen")]:
        blk = d.get(key)
        if blk is None:
            continue
        rel = blk.get("reliability", {})
        auroc = auroc_from_bins(rel.get("bin_count", []),
                                rel.get("bin_accuracy", []))
        out[tag] = {
            "macro_f1": blk.get("macro_f1"),
            "f1_positive": blk.get("f1_positive"),
            "precision": blk.get("precision"),
            "recall": blk.get("recall"),
            "pr_auc": blk.get("pr_auc"),
            "auroc_approx": auroc,
            "ece": blk.get("ece"),
        }
    return out


def aggregate(seed_metrics: list[dict]) -> dict:
    """seed_metrics: list of per-seed metric dicts (one per seed).
    Returns: {dataset: {metric: {"mean": .., "std": .., "n": ..}}}"""
    if not seed_metrics:
        return {}
    out = {}
    datasets = sorted({ds for s in seed_metrics for ds in s.keys()})
    for ds in datasets:
        vals = {m: [] for m in
                ["macro_f1", "f1_positive", "precision", "recall",
                 "pr_auc", "auroc_approx", "ece"]}
        for s in seed_metrics:
            if ds not in s:
                continue
            for m in vals:
                v = s[ds].get(m)
                if v is not None and not (isinstance(v, float) and v != v):  # not NaN
                    vals[m].append(v)
        out[ds] = {}
        for m, lst in vals.items():
            if not lst:
                continue
            mean = statistics.mean(lst)
            std = statistics.stdev(lst) if len(lst) > 1 else 0.0
            out[ds][m] = {"mean": mean, "std": std, "n": len(lst)}
    return out


def fmt_cell(stats: dict) -> str:
    if not stats:
        return "—"
    return f'{stats["mean"]:.3f} ± {stats["std"]:.3f}'


def print_table(agg: dict, cross_root: Path) -> None:
    cells = sorted(agg.keys())
    if not cells:
        print("No cross-dataset cells found under", cross_root)
        return
    sep = "─" * 134
    print(sep)
    print(f"Cross-dataset aggregation — mean ± std (ddof=1), 6 seeds where available")
    print(sep)
    header = f'{"Cell":<42}| {"Dataset":<11}| {"AUROC≈":<14}| {"Macro-F1":<14}| {"F1+":<14}| {"PR-AUC":<14}| {"ECE":<10}| n'
    print(header)
    print(sep)
    for cell in cells:
        for ds in ["HateXplain", "ToxiGen"]:
            stats = agg[cell].get(ds, {})
            if not stats:
                continue
            n = stats.get("macro_f1", {}).get("n", 0)
            line = (f'{cell:<42}| {ds:<11}| '
                    f'{fmt_cell(stats.get("auroc_approx")):<14}| '
                    f'{fmt_cell(stats.get("macro_f1")):<14}| '
                    f'{fmt_cell(stats.get("f1_positive")):<14}| '
                    f'{fmt_cell(stats.get("pr_auc")):<14}| '
                    f'{fmt_cell(stats.get("ece")):<10}| {n}')
            print(line)
    print(sep)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_CROSS_ROOT,
                        help=("Directory to scan recursively for "
                              "cross_dataset_*/.../full_evaluation.json. "
                              "Default: results/metrics_from_rcp/. On RCP, "
                              "use /scratch/${USER}/antisemitism-results "
                              "or the equivalent jumphost mount."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_JSON,
                        help=("Path to write the aggregate JSON. Default: "
                              "results/figures/cross_dataset_aggregate.json."))
    parser.add_argument("--json-only", action="store_true",
                        help="Suppress stdout table; write JSON only.")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    cells = discover_cells(args.root)
    if not cells:
        print("ERROR: No `full_evaluation.json` files found under "
              f"{args.root} matching the cross_dataset_*/seed pattern.",
              file=sys.stderr)
        print("       Hint: pass --root <path> if your data lives elsewhere "
              "(e.g. /scratch/${USER}/antisemitism-results on RCP).",
              file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(cells)} cell(s) under {args.root}:", file=sys.stderr)
    for c, runs in cells.items():
        seeds = ",".join(str(s) for s, _ in runs)
        print(f"  {c:<50} n={len(runs):>2}  seeds=[{seeds}]", file=sys.stderr)
    print(file=sys.stderr)

    agg = {}
    for cell, runs in cells.items():
        per_seed = []
        for seed, fpath in runs:
            try:
                per_seed.append(metrics_per_seed(fpath))
            except Exception as e:
                print(f"  ! failed to parse {fpath}: {e}", file=sys.stderr)
        agg[cell] = aggregate(per_seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)

    if not args.json_only:
        print_table(agg, args.root)


if __name__ == "__main__":
    main()
