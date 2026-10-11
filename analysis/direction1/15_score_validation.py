#!/usr/bin/env python3
"""Score the returned pair-validation sheets: agreement + validity rates.

Answers Reviewer #2 verbatim -- "sample size, annotation procedure, agreement,
and validity rates by replacement identity and label" -- and Reviewer #1's Q3,
which asks whether awkward rewrites could themselves bias the judgement (the
fluency x validity cross-tab below).

WHAT IT REPORTS
---------------
* Fleiss' kappa over the three annotators, on Q1 and on Q2 separately.
  Fleiss (not Cohen) because there are three raters on the same items.
* Validity by majority vote. UNCLEAR is NOT silently folded into either side:
  it is reported as its own bucket, and the headline rate is given twice --
  strict (UNCLEAR counts against) and lenient (UNCLEAR excluded). A reviewer
  can then see the assumption instead of inheriting ours.
* Wilson score intervals, not normal-approximation: at rates near 1.0 with
  n ~ 20 per cell the normal interval overshoots past 1 and is indefensible.
* A stratified overall estimate reweighted to the manifest proportions, since
  the draw was uniform per cell and the manifest is not.
* Fluency x validity cross-tab: if invalid pairs concentrate in the
  UNGRAMMATICAL bucket, the swap engine has a fixable bug rather than a
  conceptual one -- a materially different story for the paper.
* A ready-to-paste LaTeX table.

Usage:
    python3 analysis/direction1/15_score_validation.py
    python3 analysis/direction1/15_score_validation.py --dir analysis/direction1/validation
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

Q1_LEVELS = ["YES", "NO", "UNCLEAR"]
Q2_LEVELS = ["OK", "AWKWARD", "UNGRAMMATICAL"]


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval. Degenerate n=0 returns (nan, nan)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fleiss_kappa(counts: list[list[int]]) -> float:
    """Fleiss' kappa. counts[i][j] = raters assigning item i to category j.

    Rows with fewer than 2 ratings are dropped: kappa is undefined there and
    including them silently deflates the statistic.
    """
    rows = [r for r in counts if sum(r) >= 2]
    if not rows:
        return float("nan")
    n_items = len(rows)
    n_raters = sum(rows[0])
    if any(sum(r) != n_raters for r in rows):
        # ragged (a sheet left blanks) -- fall back to per-item normalisation
        p_i = []
        for r in rows:
            n = sum(r)
            p_i.append((sum(c * c for c in r) - n) / (n * (n - 1)))
        p_bar = sum(p_i) / n_items
        tot = sum(sum(r) for r in rows)
        p_j = [sum(r[j] for r in rows) / tot for j in range(len(rows[0]))]
        p_e = sum(p * p for p in p_j)
        return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")
    p_i = [(sum(c * c for c in r) - n_raters) / (n_raters * (n_raters - 1)) for r in rows]
    p_bar = sum(p_i) / n_items
    p_j = [sum(r[j] for r in rows) / (n_items * n_raters) for j in range(len(rows[0]))]
    p_e = sum(p * p for p in p_j)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def majority(votes: list[str], levels: list[str]) -> str:
    """Strict majority; ties resolve to UNCLEAR / AWKWARD (the middle option)."""
    votes = [v for v in votes if v in levels]
    if not votes:
        return "MISSING"
    counts = {lv: votes.count(lv) for lv in levels}
    top = max(counts.values())
    winners = [lv for lv, c in counts.items() if c == top]
    if len(winners) == 1:
        return winners[0]
    return "UNCLEAR" if levels is Q1_LEVELS else "AWKWARD"


def load_sheets(d: Path) -> pd.DataFrame:
    sheets = sorted(d.glob("pairs_*.xlsx"))
    if len(sheets) < 2:
        raise SystemExit("need at least 2 completed sheets in %s, found %d" % (d, len(sheets)))
    frames = []
    for s in sheets:
        who = s.stem.replace("pairs_", "")
        df = pd.read_excel(s, sheet_name="pairs")
        df["annotator"] = who
        frames.append(df[["pair_id", "annotator", "label_preserved", "fluency", "notes"]])
    long = pd.concat(frames, ignore_index=True)
    for col, levels in (("label_preserved", Q1_LEVELS), ("fluency", Q2_LEVELS)):
        long[col] = long[col].astype(str).str.strip().str.upper()
        long.loc[~long[col].isin(levels), col] = pd.NA
    print("loaded %d sheets: %s" % (len(sheets), ", ".join(s.stem[6:] for s in sheets)))
    return long


def rate_block(df: pd.DataFrame, by: str | None) -> pd.DataFrame:
    groups = [("ALL", df)] if by is None else list(df.groupby(by, sort=True))
    rows = []
    for name, g in groups:
        n = len(g)
        yes = int((g.verdict == "YES").sum())
        no = int((g.verdict == "NO").sum())
        unc = int((g.verdict == "UNCLEAR").sum())
        lo_s, hi_s = wilson(yes, n)                       # strict: UNCLEAR counts against
        lo_l, hi_l = wilson(yes, yes + no) if yes + no else (float("nan"),) * 2
        rows.append({
            "group": name, "n": n, "yes": yes, "no": no, "unclear": unc,
            "valid_strict": yes / n if n else float("nan"), "lo_s": lo_s, "hi_s": hi_s,
            "valid_lenient": yes / (yes + no) if yes + no else float("nan"),
            "lo_l": lo_l, "hi_l": hi_l,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score returned pair-validation sheets.")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent / "validation"))
    args = ap.parse_args()
    d = Path(args.dir)

    long = load_sheets(d)
    key = pd.read_csv(d / "key.csv")

    # ---- agreement, before any aggregation ----
    for col, levels, qname in (("label_preserved", Q1_LEVELS, "Q1 label_preserved"),
                               ("fluency", Q2_LEVELS, "Q2 fluency")):
        tab = (long.dropna(subset=[col])
                   .groupby(["pair_id", col]).size().unstack(fill_value=0)
                   .reindex(columns=levels, fill_value=0))
        k = fleiss_kappa(tab.values.tolist())
        print("Fleiss kappa  %-20s = %.3f   (%d items with >=2 ratings)"
              % (qname, k, int((tab.sum(axis=1) >= 2).sum())))

    # ---- majority vote per pair ----
    verdicts = (long.groupby("pair_id")
                    .agg(verdict=("label_preserved", lambda v: majority(list(v.dropna()), Q1_LEVELS)),
                         flu=("fluency", lambda v: majority(list(v.dropna()), Q2_LEVELS)))
                    .reset_index())
    df = key.merge(verdicts, on="pair_id", how="inner")
    df = df[df.verdict != "MISSING"]

    print()
    print("=== OVERALL (unweighted over the stratified draw) ===")
    print(rate_block(df, None).to_string(index=False, float_format=lambda x: "%.3f" % x))

    print()
    print("=== BY REPLACEMENT IDENTITY (Rev #2) ===")
    print(rate_block(df, "alternative_group").to_string(index=False, float_format=lambda x: "%.3f" % x))

    print()
    print("=== BY GOLD LABEL (Rev #2) ===")
    print(rate_block(df, "label").to_string(index=False, float_format=lambda x: "%.3f" % x))

    # ---- reweight to manifest proportions ----
    weighted = None
    meta_path = d / "sample_meta.json"
    if meta_path.exists():
        import json
        props = json.loads(meta_path.read_text(encoding="utf-8")).get("manifest_proportions", {})
        if props:
            cell = df.assign(cell=df.alternative_group.astype(str) + "|" + df.label.astype(str))
            per = cell.groupby("cell").verdict.apply(lambda v: (v == "YES").mean())
            w = {c: props.get(c, 0.0) for c in per.index}
            tot = sum(w.values())
            if tot > 0:
                est = sum(per[c] * w[c] for c in per.index) / tot
                # Stratified-proportion variance: the binomial one does not
                # apply to a weighted estimate over unequal cells.
                n_c = cell.groupby("cell").size()
                var = sum((w[c] / tot) ** 2 * per[c] * (1 - per[c]) / n_c[c]
                          for c in per.index if n_c[c] > 0)
                half = 1.96 * (var ** 0.5)
                weighted = (est, max(0.0, est - half), min(1.0, est + half))
                print()
                print("Manifest-reweighted validity (strict): %.3f  [%.3f, %.3f]"
                      % weighted)
                print("  the draw is uniform per cell; the manifest is not, so this is the")
                print("  figure that applies to the 2,117-pair denominator.")

    # ---- does invalidity concentrate in broken rewrites? (Rev #1 Q3) ----
    print()
    print("=== FLUENCY x VALIDITY (Rev #1 Q3) ===")
    print(pd.crosstab(df.flu, df.verdict).to_string())
    print("If NO concentrates in UNGRAMMATICAL, the swap engine has a fixable bug")
    print("rather than a conceptual problem -- a different story for the paper.")

    # ---- LaTeX ----
    tex = ["\\begin{tabular}{lrrrr}", "\\toprule",
           "Replacement & $n$ & Valid & 95\\% CI & Unclear \\\\", "\\midrule"]
    for r in rate_block(df, "alternative_group").itertuples(index=False):
        tex.append("%s & %d & %.2f & [%.2f, %.2f] & %d \\\\"
                   % (str(r.group).capitalize(), r.n, r.valid_strict, r.lo_s, r.hi_s, r.unclear))
    o = rate_block(df, None).iloc[0]
    tex += ["\\midrule",
            "Balanced audit sample & %d & %.2f & [%.2f, %.2f] & %d \\\\"
            % (o.n, o.valid_strict, o.lo_s, o.hi_s, o.unclear)]
    if weighted is not None:
        tex.append("Manifest-weighted & %d & %.2f & [%.2f, %.2f] & --- \\\\"
                   % (o.n, weighted[0], weighted[1], weighted[2]))
    tex += ["\\bottomrule", "\\end{tabular}"]
    out = d / "validity_table.tex"
    out.write_text("\n".join(tex), encoding="utf-8")
    print()
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
