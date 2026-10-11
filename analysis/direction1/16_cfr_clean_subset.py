#!/usr/bin/env python3
"""Is the reported CFR driven by pairs where the rewrite did not do its job?

THE DEFECT
----------
The swap operator replaces ONE token occurrence per rewrite
(``_swaps_for_match``: "All accepted swap pairs for a single token occurrence",
splicing at that match's offsets). A tweet with several Jewish identity mentions
therefore yields "counterfactuals" that still say Jew:

    "Jews with their You can't say I'm a Jew just because I'm a Jew"
 -> "Buddhists with their You can't say I'm a Jew just because I'm a Jew"

On the test manifest, 710 of 2,117 pairs (33.5%) still contain a Jewish
religious-ethnic token after the swap, 135 of them twice or more, 120 on
ANTISEMITIC-labelled pairs. CFR is defined as "the prediction changed when the
target identity changed", so on those pairs it is not measuring what the paper
says it measures. The behaviour is deliberate in the code and Section 3 describes
it accurately ("a token", singular) -- the defect is in the operationalisation,
not the execution, which is why every test passes. Two smaller residues ride
along: 383 pairs (18.1%) keep a geopolitical proxy the rewrite does not move, and
23 (1.1%) keep a Jewish-coded conspiracy term.

WHY A BARE SUBSET COMPARISON PROVES NOTHING
-------------------------------------------
Dropping 710 of 2,117 pairs changes CFR whatever those pairs contain, because a
smaller sample is a different sample. Three controls, each answering a question
the others cannot:

  PLACEBO NULL. For every slice, draw random subsets of the SAME SIZE and
  recompute the same statistic. One mask per replicate, shared by both arms,
  mirroring the fact that all cells are evaluated on one manifest -- an unpaired
  null would ignore that and come out far too wide. Observed inside the band =
  the slice did nothing a random subset of that size would not have done.

  SHAM SLICES. Content-defined but identity-irrelevant: no URL, no @USER, shorter
  half. These must land INSIDE. If a sham lands outside, any semantic split of
  this manifest looks special and a lone "outside" on the real slice means
  nothing.

  POWER CHECK. Positive-label pairs only -- a slice where CFR genuinely differs,
  since positives sit near the boundary. This must land OUTSIDE. If it does not,
  the test cannot detect a real difference and "inside" everywhere is
  uninformative rather than reassuring.

Read them together. Real inside + shams inside + power outside is evidence the
defect did not drive the headline. Real outside with shams inside is evidence it
did, and the operator has to be fixed -- which changes every CFR number in an
accepted paper, so it is worth knowing before that conversation rather than
during it.

The ``all`` slice must reproduce cfr_summary.csv and the paper's 49/58/73. It is
the self-check; if it fails, nothing below it counts.

This does NOT estimate CFR under a corrected operator: that needs new swap texts,
hence new forward passes, and arguably a retrain since CCI was trained on
one-occurrence swaps. Cheap question first.

CPU-only. Usage:
    python3 analysis/direction1/16_cfr_clean_subset.py
    python3 analysis/direction1/16_cfr_clean_subset.py --n-placebo 5000
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from analysis.direction1._utils import (  # noqa: E402
    cfr_cache_root,
    positive_class_z,
    seeds_for,
    sigmoid,
)

JEWISH = re.compile(r"\b(jew|jews|jewish|hebrew|hebrews|judaism)\b", re.I)
GEO = re.compile(r"\b(israel|israeli|israelis|zionist|zionists|zionism)\b", re.I)
CODED = re.compile(r"\b(khazar|khazars|rothschild|rothschilds|soros|globalist|globalists)\b", re.I)

HEADLINE_PAIRS = [
    ("model_c_deberta_focal_mask", "cci_v2_fixed_js", "DeBERTa-base"),
    ("model_c_deberta_v3_large", "cci_v2_fixed_js_deberta_v3_large", "DeBERTa-large"),
    ("model_c_roberta_large", "cci_v2_fixed_js_roberta_large", "RoBERTa-large"),
]
METHODS = [m for p in HEADLINE_PAIRS for m in p[:2]] + ["cda_deberta_base", "davani_lambda05"]


def load(experiment: str, seed: int):
    d = cfr_cache_root() / ("%s_seed%d" % (experiment, seed))
    o_path, s_path = d / "test_originals.parquet", d / "test_swap_logits.parquet"
    if not (o_path.exists() and s_path.exists()):
        return None
    o, s = pd.read_parquet(o_path), pd.read_parquet(s_path)
    if "swap_text" not in s.columns:
        raise SystemExit("cache at %s has no swap_text column; rebuild with 02." % d)
    idx = s.test_idx.to_numpy(np.int64)
    p_orig = sigmoid(positive_class_z(
        np.stack([o.logit_0.to_numpy(np.float32), o.logit_1.to_numpy(np.float32)], axis=1)))
    p_swap = sigmoid(positive_class_z(
        np.stack([s.logit_0.to_numpy(np.float32), s.logit_1.to_numpy(np.float32)], axis=1)))
    lab = o["true_label"].to_numpy(np.int64)[idx] if "true_label" in o.columns else None
    return p_orig[idx], p_swap, s.swap_text.astype(str), lab


def main() -> None:
    ap = argparse.ArgumentParser(description="CFR on defect-free subsets, with placebo controls.")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--n-placebo", type=int, default=1000)
    ap.add_argument("--rng-seed", type=int, default=12345)
    args = ap.parse_args()

    flips, swap_text, labels = {}, None, None
    for m in dict.fromkeys(args.methods):
        for sd in seeds_for(m):
            got = load(m, sd)
            if got is None:
                print("[skip] no cache for %s seed %d" % (m, sd))
                continue
            p_o, p_s, txt, lab = got
            if swap_text is None:
                swap_text, labels = txt, lab
            elif len(txt) != len(swap_text):
                raise SystemExit("pair count differs across cells (%d vs %d)"
                                 % (len(txt), len(swap_text)))
            flips[(m, sd)] = ((p_o >= 0.5) != (p_s >= 0.5)).astype(np.float64)
    if not flips:
        raise SystemExit("no cached checkpoints found")

    n = len(swap_text)
    jew = swap_text.str.contains(JEWISH, regex=True).to_numpy()
    geo = swap_text.str.contains(GEO, regex=True).to_numpy()
    cod = swap_text.str.contains(CODED, regex=True).to_numpy()
    lens = swap_text.str.len().to_numpy()

    slices = [
        ("all", "REFERENCE", np.ones(n, bool), "every pair; must match the paper"),
        ("no_jewish", "REAL", ~jew, "no Jewish token survives the rewrite"),
        ("no_anchor", "REAL", ~(jew | geo | cod), "no Jewish token, proxy or coded term"),
        ("sham_no_url", "SHAM", ~swap_text.str.contains(r"\[URL\]", regex=True).to_numpy(),
         "content-defined, identity-irrelevant"),
        ("sham_no_user", "SHAM", ~swap_text.str.contains("@USER", regex=False).to_numpy(),
         "content-defined, identity-irrelevant"),
        ("sham_shorter", "SHAM", lens <= np.percentile(lens, 66.5),
         "shortest 66.5%, matching no_jewish retention"),
    ]
    if labels is not None:
        slices.append(("positives_only", "POWER", labels == 1,
                       "must land OUTSIDE or the test has no power"))

    print()
    print("%-16s %-10s %6s %7s  %s" % ("slice", "role", "kept", "of", "rationale"))
    print("-" * 92)
    for name, role, mask, why in slices:
        print("%-16s %-10s %6d %6.1f%%  %s" % (name, role, mask.sum(), 100*mask.mean(), why))

    def reduction(base, cci, mask):
        b = np.mean([flips[(base, s)][mask].mean() for s in seeds_for(base) if (base, s) in flips])
        c = np.mean([flips[(cci, s)][mask].mean() for s in seeds_for(cci) if (cci, s) in flips])
        return 100.0 * (b - c) / b if b else np.nan

    print()
    print("%-34s %s" % ("cell", "  ".join("%-13s" % s[0] for s in slices)))
    print("-" * (36 + 15 * len(slices)))
    for m in dict.fromkeys(args.methods):
        sds = [s for s in seeds_for(m) if (m, s) in flips]
        if not sds:
            continue
        cells = ["%-13s" % ("%.4f" % np.mean([flips[(m, s)][mask].mean() for s in sds]))
                 for _, _, mask, _ in slices]
        print("%-34s %s" % (m, "  ".join(cells)))

    rng = np.random.default_rng(args.rng_seed)
    print()
    print("CCI reduction vs same-architecture Model C, each slice against a")
    print("size-matched placebo null (%d replicates, one mask shared by both arms)."
          % args.n_placebo)
    print("Expect: REAL inside, SHAM inside, POWER outside.")
    print()
    print("%-15s %-16s %-10s %8s %19s %8s  %s"
          % ("backbone", "slice", "role", "observed", "placebo 95%", "p", "verdict"))
    print("-" * 100)

    out_rows = []
    for base, cci, lab in HEADLINE_PAIRS:
        if not any((base, s) in flips for s in seeds_for(base)):
            continue
        for name, role, mask, _ in slices:
            obs = reduction(base, cci, mask)
            if name == "all":
                print("%-15s %-16s %-10s %7.1f%% %19s %8s  %s"
                      % (lab, name, role, obs, "--", "--", "reference"))
                out_rows.append({"backbone": lab, "slice": name, "role": role,
                                 "observed": obs, "lo": np.nan, "hi": np.nan, "p": np.nan})
                continue
            k = int(mask.sum())
            null = np.empty(args.n_placebo)
            for i in range(args.n_placebo):
                mm = np.zeros(n, bool)
                mm[rng.choice(n, size=k, replace=False)] = True
                null[i] = reduction(base, cci, mm)
            lo, hi = np.percentile(null, [2.5, 97.5])
            centre = null.mean()
            p = float(np.mean(np.abs(null - centre) >= abs(obs - centre)))
            inside = lo <= obs <= hi
            expected_inside = role in ("REAL", "SHAM")
            flagv = "" if inside == expected_inside else "   <-- UNEXPECTED"
            print("%-15s %-16s %-10s %7.1f%% %19s %8.3f  %s%s"
                  % (lab, name, role, obs, "[%.1f%%, %.1f%%]" % (lo, hi), p,
                     "inside" if inside else "OUTSIDE", flagv))
            out_rows.append({"backbone": lab, "slice": name, "role": role, "observed": obs,
                             "lo": lo, "hi": hi, "p": p, "inside": inside})
        print()

    print("How to read this. The 'all' rows must reproduce 49 / 58 / 73 from the")
    print("paper; if they do not, the script is wrong and nothing else counts.")
    print("Then: REAL inside + SHAM inside + POWER outside means the one-occurrence")
    print("defect did not drive the headline. REAL outside while SHAM stays inside")
    print("means it did. SHAM outside, or POWER inside, means the control battery")
    print("itself is not trustworthy here and the verdicts should not be quoted.")

    out = _HERE / "results" / "cfr_clean_subset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print()
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
