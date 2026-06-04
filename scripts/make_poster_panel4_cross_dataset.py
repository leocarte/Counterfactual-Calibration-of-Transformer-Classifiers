#!/usr/bin/env python3
"""Poster Panel 4 (NeurIPS-style v3) - cross-dataset slope chart.

Two-panel slope chart of F1+ at the default 0.5 cutoff vs F1+ at each
model's PR-curve-optimal threshold, on HateXplain (Jewish-target) and
ToxiGen (Jewish-target). The right column is the "real transfer result":
once each model uses its own threshold, CCI v2 wins or ties Model C on
5 of 6 iso-architecture cells.

v3 restyle (from v2):
  * Inline endpoint labels (DBa / DLa / RLa) moved outside the marker
    column with thin horizontal lead-lines so the markers stay clean.
  * @opt highlight column made noticeably more visible (denser blue,
    border + interior star annotation).
  * "5 of 6 cells" rounded callout placed inside the HateXplain panel
    so the headline finding reads from 1 m without scanning captions.
  * CCI v2 lines thicker (2.4) and more saturated (alpha 0.95); Model C
    thinner (1.8) and softer (alpha 0.75) -> eye follows CCI v2 first.
  * Title and subtitle tightened; PR-AUC sub-claim folded into callout.
  * Bottom legend split into two rows: models on top, arch markers below.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch  # noqa: F401  (kept for future tweaks)

from _poster_style import (
    COLOR_BASELINE,
    COLOR_CCI,
    COLOR_HIGHLIGHT_BG,
    setup_neurips_style,
)


HERE = Path(__file__).resolve().parent.parent
CSV = HERE / "results" / "metrics_from_rcp" / "threshold_sweep_cross_dataset.csv"
OUT_DIR = HERE / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


ARCH_LABEL = {
    "deberta-base":  "DBa",
    "deberta-large": "DLa",
    "roberta-large": "RLa",
}
ARCH_MARKER = {
    "deberta-base":  "o",
    "deberta-large": "s",
    "roberta-large": "^",
}
ARCH_MARKERSIZE = {
    "deberta-base":  9,
    "deberta-large": 9,
    "roberta-large": 10,
}

DATASETS = [
    ("hatexplain", "HateXplain  (Jewish-target, n=2,543)"),
    ("toxigen",    "ToxiGen  (Jewish-target, n=684)"),
]


def main() -> None:
    setup_neurips_style()
    df = pd.read_csv(CSV)
    agg = df.groupby(["model", "arch", "dataset"], sort=False).agg(
        F1_05_mean=("F1+_at_0.5", "mean"),
        F1_05_std=("F1+_at_0.5", "std"),
        F1_opt_mean=("F1+_at_optimal", "mean"),
        F1_opt_std=("F1+_at_optimal", "std"),
    ).reset_index()
    for c in ("F1_05_std", "F1_opt_std"):
        agg[c] = agg[c].fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
    x = [0, 1]
    # x-position of the label column (well outside the @opt marker at x=1).
    LABEL_X = 1.18
    # x-limits leave room for both the highlight column and the label column.
    XLIM = (-0.25, 1.42)

    for ax_idx, (ax, (ds_tag, ds_label)) in enumerate(zip(axes, DATASETS)):
        sub = agg[agg["dataset"] == ds_tag].copy()

        # --- @opt column highlight ---------------------------------------
        # v2 used alpha=0.65 on a very pale colour and washed out; v3 uses
        # a slightly denser fill (alpha=0.35 over the same #EAF1F8 base
        # but plotted twice for opacity) plus a darker border + star.
        ax.axvspan(
            0.72, 1.30,
            color=COLOR_HIGHLIGHT_BG, alpha=1.0, zorder=-3,
        )
        ax.axvspan(
            0.72, 1.30,
            color=COLOR_CCI, alpha=0.10, zorder=-2,
        )
        # Subtle vertical anchor line on the @opt tick.
        ax.axvline(1.0, color=COLOR_CCI, alpha=0.35, linewidth=1.0, zorder=-1)

        # Compute y-bounds first so we can position the star + callout.
        all_y = list(sub["F1_05_mean"]) + list(sub["F1_opt_mean"])
        y_lo = min(all_y) - 0.04
        y_hi = max(all_y) + 0.10  # extra headroom for star + callout

        # Star annotation inside the @opt column. On the HateXplain panel
        # the rounded callout occupies the top-right, so we anchor the
        # "ranking ceiling" label to the BOTTOM of that panel instead;
        # on ToxiGen there is no callout, so we keep it at the top.
        if ax_idx == 0:
            star_y = y_lo + (y_hi - y_lo) * 0.055
            ax.text(
                1.0, star_y, r"$\bigstar$  ranking ceiling",
                ha="center", va="bottom",
                fontsize=9.5, fontstyle="italic", color=COLOR_CCI,
                zorder=4,
            )
        else:
            star_y = y_hi - 0.012
            ax.text(
                1.0, star_y, "ranking ceiling",
                ha="center", va="top",
                fontsize=9.0, fontstyle="italic", color=COLOR_CCI,
                zorder=4,
            )
            ax.text(
                1.0, star_y - (y_hi - y_lo) * 0.055, r"$\bigstar$",
                ha="center", va="top",
                fontsize=11, color=COLOR_CCI, zorder=4,
            )

        # --- slope lines + endpoint labels -------------------------------
        # Pre-compute CCI endpoint y's so we can de-overlap labels.
        cci_rows = sub[sub["model"] == "cci_v2"].copy()
        cci_rows = cci_rows.sort_values("F1_opt_mean").reset_index(drop=True)

        for _, row in sub.iterrows():
            is_cci = row["model"] == "cci_v2"
            color = COLOR_CCI if is_cci else COLOR_BASELINE
            lw = 2.4 if is_cci else 1.8
            alpha = 0.95 if is_cci else 0.75
            marker = ARCH_MARKER[row["arch"]]
            msize = ARCH_MARKERSIZE[row["arch"]]
            y = [row["F1_05_mean"], row["F1_opt_mean"]]
            yerr = [row["F1_05_std"], row["F1_opt_std"]]
            ax.errorbar(
                x, y, yerr=yerr,
                color=color, alpha=alpha,
                marker=marker, markersize=msize,
                linewidth=lw,
                capsize=3, capthick=1.0,
                zorder=4 if is_cci else 3,
            )

        # --- Inline DBa / DLa / RLa labels w/ lead-lines (CCI only) ------
        # Spread labels vertically if they bunch (>1 row within 0.012 of
        # F1+ space at the @opt endpoint). Keeps the lead-lines tidy.
        cci_y = cci_rows["F1_opt_mean"].tolist()
        cci_arch = cci_rows["arch"].tolist()
        spread = _spread_labels(cci_y, min_gap=(y_hi - y_lo) * 0.045)

        for arch, y_raw, y_label in zip(cci_arch, cci_y, spread):
            # Thin lead-line from marker (x=1) -> label (x=LABEL_X).
            ax.plot(
                [1.005, LABEL_X - 0.015],
                [y_raw, y_label],
                color=COLOR_CCI, alpha=0.55, linewidth=0.9, zorder=2,
            )
            ax.text(
                LABEL_X, y_label, ARCH_LABEL[arch],
                fontsize=10.5, fontweight="semibold",
                color=COLOR_CCI,
                ha="left", va="center", zorder=5,
            )

        ax.set_title(ds_label, fontsize=12, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([
            "F$_1^+$ @ 0.5\n(default cutoff)",
            "F$_1^+$ @ opt\n(PR-curve optimal)",
        ], fontsize=11)
        ax.set_xlim(*XLIM)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(axis="y", labelsize=10)
        ax.yaxis.grid(True, alpha=0.5, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        # --- Headline callout INSIDE the HateXplain panel ----------------
        if ax_idx == 0:
            callout_text = (
                "At F$_1^+$@opt:\n"
                "5 / 6 cells\n"
                "CCI v2 $\\geq$ Model C\n"
                "(PR-AUC $\\Delta \\in [-0.007, +0.021]$)"
            )
            ax.text(
                0.97, 0.97, callout_text,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=10.5, color=COLOR_CCI, fontweight="semibold",
                linespacing=1.35,
                bbox=dict(
                    boxstyle="round,pad=0.45,rounding_size=0.35",
                    facecolor="white",
                    edgecolor=COLOR_CCI,
                    linewidth=1.4,
                    alpha=0.96,
                ),
                zorder=6,
            )

    axes[0].set_ylabel("F$_1^+$ (positive class)", fontsize=12)

    # Title: tighter than v2.
    fig.suptitle(
        "Cross-dataset transfer: ranking transfers, operating point shifts with prevalence",
        fontsize=14, y=1.01, fontweight="semibold",
    )
    # Subtitle: single line, punchier; PR-AUC moved to the callout.
    fig.text(
        0.5, -0.04,
        r"$\bf{5\ of\ 6\ iso\text{-}architecture\ cells:\ CCI\ v2\ \geq\ Model\ C\ at\ F_1^+@opt.}$",
        ha="center", fontsize=11, style="italic", color="#374151",
    )

    # --- Legend: two rows (models above, arch markers below) -------------
    model_handles = [
        Line2D([0], [0], color=COLOR_CCI, lw=2.4, marker="o", markersize=8,
               alpha=0.95, label="CCI v2 (ours)"),
        Line2D([0], [0], color=COLOR_BASELINE, lw=1.8, marker="o", markersize=8,
               alpha=0.75, label="Model C baseline"),
    ]
    arch_handles = [
        Line2D([0], [0], color="#374151", marker="o", linestyle="None",
               markersize=8, label="DBa: DeBERTa-v3-base"),
        Line2D([0], [0], color="#374151", marker="s", linestyle="None",
               markersize=8, label="DLa: DeBERTa-v3-large"),
        Line2D([0], [0], color="#374151", marker="^", linestyle="None",
               markersize=9, label="RLa: RoBERTa-large"),
    ]
    leg_models = fig.legend(
        handles=model_handles, loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.10),
        frameon=False, fontsize=10.5,
    )
    fig.add_artist(leg_models)  # keep this legend when we add the second
    fig.legend(
        handles=arch_handles, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.15),
        frameon=False, fontsize=10.5,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    png = OUT_DIR / "poster_panel4_cross_dataset_slope.png"
    pdf = OUT_DIR / "poster_panel4_cross_dataset_slope.pdf"
    plt.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def _spread_labels(y_values, min_gap):
    """Return label y-positions that are at least min_gap apart while
    staying close to the original y_values. Inputs are assumed sorted
    ascending. A simple greedy pass from bottom to top.
    """
    out = []
    last = -float("inf")
    for y in y_values:
        y_new = max(y, last + min_gap)
        out.append(y_new)
        last = y_new
    # Re-center the spread around the original mean so we do not drift up.
    if out:
        drift = (sum(out) - sum(y_values)) / len(out)
        out = [v - drift for v in out]
    return out


if __name__ == "__main__":
    main()
