#!/usr/bin/env python3
"""Draw a stratified, blinded sample of counterfactual pairs for human validation.

WHY THIS EXISTS
---------------
The WOAH meta-review calls the missing validation "a significant limitation
rather than a minor omission", because CFR_hard is defined as "the prediction
changed" but is *reported* as "the model failed". The gap between those two
readings is exactly the label-preservation assumption, and today it is not
quantified at all. Reviewer #2 asks, verbatim, for "the sample size, annotation
procedure, agreement, and validity rates by replacement identity and label".

This script produces the sheets; 15_score_validation.py produces the numbers.

DESIGN
------
* Stratified by (alternative_group x gold label) -- the exact breakdown the
  reviewer asked for. Uniform allocation, so small cells stay estimable instead
  of being swamped. The overall rate is therefore a stratified estimate, and
  15_score_validation.py reweights back to manifest proportions.
* Blinded: annotators never see the model prediction, whether the pair flipped,
  which symmetry rule admitted it, or which replacement group it is. Anchoring
  on any of those makes the validity rate uninterpretable.
* They DO see the gold label of the original, because the question is "does this
  label still apply after the swap?" -- that is the assumption the metric rests
  on. Blind re-labelling of both sides would instead measure annotator agreement
  with the corpus, a different quantity.
* Same row order for every annotator, randomised against manifest order with a
  fixed seed, so agreement is computable row-wise and the draw is reproducible.

Usage:
    python3 analysis/direction1/14_make_validation_sample.py --per-cell 20
    python3 analysis/direction1/14_make_validation_sample.py --per-cell 25 \
        --annotators annotator_1 annotator_2 annotator_3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from analysis.direction1._utils import cfr_cache_root  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.data.splits import load_splits  # noqa: E402

LABEL_NAMES = {0: "NOT ANTISEMITIC", 1: "ANTISEMITIC"}

PREAMBLE = [
    "COUNTERFACTUAL PAIR VALIDATION",
    "",
    "For each row you see a tweet, the label it carries in GoldStandard2024, and a",
    "rewritten version in which one religious-ethnic identity term was replaced.",
    "",
    "Q1  label_preserved  -- Does the ORIGINAL label still apply to the rewritten text?",
    "      YES      the rewritten text warrants the same label",
    "      NO       the rewrite changes what the text is doing, so the label no longer fits",
    "      UNCLEAR  genuinely undecidable without more context",
    "",
    "Q2  fluency  -- Independently of the label, how does the rewritten text read?",
    "      OK             natural",
    "      AWKWARD        odd phrasing but comprehensible",
    "      UNGRAMMATICAL  broken",
    "",
    "Judge Q1 and Q2 separately: an awkward rewrite can still preserve the label.",
    "Do not discuss rows with the other annotators before all sheets are complete.",
    "Use the notes column where a row was hard -- those cases are the interesting ones.",
]

DV_LABEL = '"YES,NO,UNCLEAR"'
DV_FLUENCY = '"OK,AWKWARD,UNGRAMMATICAL"'


def build_sheet(rows: pd.DataFrame, out_path: Path) -> None:
    """Write one annotator workbook with dropdown validation on the two questions."""
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.styles import Alignment, Font

    wb = Workbook()

    ws0 = wb.active
    ws0.title = "instructions"
    for i, line in enumerate(PREAMBLE, start=1):
        ws0.cell(row=i, column=1, value=line)
    ws0.column_dimensions["A"].width = 100

    ws = wb.create_sheet("pairs")
    headers = ["pair_id", "gold_label", "original_text", "swap_text",
               "label_preserved", "fluency", "notes"]
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        ws.cell(row=1, column=c).font = Font(bold=True)

    for r in rows.itertuples(index=False):
        ws.append([r.pair_id, r.gold_label, r.original_text, r.swap_text, "", "", ""])

    dv1 = DataValidation(type="list", formula1=DV_LABEL, allow_blank=True)
    dv2 = DataValidation(type="list", formula1=DV_FLUENCY, allow_blank=True)
    ws.add_data_validation(dv1)
    ws.add_data_validation(dv2)
    last = len(rows) + 1
    dv1.add("E2:E%d" % last)
    dv2.add("F2:F%d" % last)

    widths = (("A", 10), ("B", 18), ("C", 70), ("D", 70), ("E", 16), ("F", 16), ("G", 30))
    for col, w in widths:
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2, min_col=3, max_col=4):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"

    wb.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build blinded pair-validation sheets.")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--manifest", default=None,
                    help="swap manifest parquet (default: <scratch>/cfr_cache/swap_manifest.parquet)")
    ap.add_argument("--per-cell", type=int, default=20,
                    help="pairs per (replacement identity x label) cell")
    ap.add_argument("--annotators", nargs="+", default=["a1", "a2", "a3"])
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out-dir", default=str(_HERE / "validation"))
    args = ap.parse_args()

    # 01_build_swap_manifest.py writes to cfr_cache_root()/_shared/, not to the
    # cache root itself -- the manifest is shared across every checkpoint.
    manifest_path = (Path(args.manifest) if args.manifest
                     else cfr_cache_root() / "_shared" / "test_swaps_manifest.parquet")
    if not manifest_path.exists():
        raise SystemExit("manifest not found: %s\nRun 01_build_swap_manifest.py first."
                         % manifest_path)

    man = pd.read_parquet(manifest_path)

    cfg = load_config(args.config)
    splits = load_splits(cfg.paths.processed_dir)
    test = splits["test"].reset_index(drop=True)
    labels = test[cfg.data.label_column].astype(int)

    # test_idx indexes the test split positionally -- same construction as 01.
    man = man[man.test_idx < len(labels)].copy()
    man["label"] = man.test_idx.map(labels)

    # ---- stratified draw ----
    rng_seed = args.seed
    picked, report = [], []
    for (alt, lab), grp in man.groupby(["alternative_group", "label"], sort=True):
        take = min(args.per_cell, len(grp))
        picked.append(grp.sample(n=take, random_state=rng_seed))
        report.append({"alternative_group": alt, "label": int(lab),
                       "available": int(len(grp)), "sampled": int(take)})
        rng_seed += 1
    sample = pd.concat(picked, ignore_index=True)

    # randomise against manifest order, then assign ids in presentation order
    sample = sample.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sample["pair_id"] = ["P%04d" % (i + 1) for i in range(len(sample))]
    sample["gold_label"] = sample["label"].map(LABEL_NAMES)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["pair_id", "gold_label", "original_text", "swap_text"]
    for a in args.annotators:
        build_sheet(sample[cols], out / ("pairs_%s.xlsx" % a))

    # Same rows, same order, as CSV: analysis/direction1/validation_app.html
    # loads this directly, so nobody exports from Excel by hand.
    sample[cols].to_csv(out / "pairs.csv", index=False)

    # the key stays OUT of the annotator sheets
    key_cols = ["pair_id", "test_idx", "swap_idx", "source_group", "alternative_group",
                "original_token", "swap_token", "label", "decision_reason"]
    sample[[c for c in key_cols if c in sample.columns]].to_csv(out / "key.csv", index=False)

    props = (man.groupby(["alternative_group", "label"]).size() / len(man)).round(6)
    meta = {
        "manifest": str(manifest_path),
        "manifest_pairs_total": int(len(man)),
        "per_cell": args.per_cell,
        "sampled": int(len(sample)),
        "seed": args.seed,
        "annotators": list(args.annotators),
        "strata": report,
        "manifest_proportions": {"%s|%d" % (k[0], k[1]): float(v) for k, v in props.items()},
    }
    (out / "sample_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("manifest pairs      : %d" % len(man))
    print("sampled             : %d  (%d cells, up to %d each)"
          % (len(sample), len(report), args.per_cell))
    print()
    print("%-14s %6s %10s %8s" % ("replacement", "label", "available", "sampled"))
    for r in report:
        print("%-14s %6d %10d %8d"
              % (r["alternative_group"], r["label"], r["available"], r["sampled"]))
    print()
    print("wrote %d annotator sheets + key.csv + sample_meta.json to %s"
          % (len(args.annotators), out))
    print("Keep key.csv away from the annotators until every sheet is returned.")


if __name__ == "__main__":
    main()
