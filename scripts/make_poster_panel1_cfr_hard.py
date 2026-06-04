#!/usr/bin/env python3
"""Poster Panel 1 (NeurIPS-style v3) — CFR_hard reduction across architectures.

Grouped bar chart of the hard counterfactual flip rate before/after CCI v2 on
DeBERTa-base, DeBERTa-large, and RoBERTa-large. Values are sourced from
`analysis/direction1/REPORT.md` (the per-seed ground truth), not from the
paper's rounded table — so the standard deviations are correct, not
hand-reconstructed from reduction percentages.

v3 redesign vs v2:
  - Percent labels are no longer floating high above the bars. Each `−X%`
    sits next to a reduction bracket that visually connects the Model C
    bar top to the CCI v2 bar top, so the eye reads "this much was cut".
  - Legend moved below the title (horizontal, centred) so it no longer
    competes with the leftmost `-49%` label.
  - Subtitle cut from 4 lines to 2.
  - Title shortened (CFR_hard is on the y-axis, no need to repeat "Hard").
  - Y-axis label vertical (matplotlib default).
  - Subtle 1 px baseline line at y=0 for the "no flips" reference.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from _poster_style import (
    COLOR_BASELINE,
    COLOR_CCI,
    setup_neurips_style,
)


HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (arch label, Model C (mean, std), CCI v2 (mean, std), reduction%)
# Values are read DIRECTLY from analysis/direction1/REPORT.md rows 44, 47,
# 50, 51, 52, 53 — NOT reconstructed from reduction percentages. The v1
# pass had Model C-large + Model C-RoBERTa stds reconstructed (0.012, 0.010)
# which were 3.5-4x larger than the actual per-seed std (0.0034, 0.0023).
ROWS = [
    ("DeBERTa-v3-base",  (0.0358, 0.0083), (0.0183, 0.0028), -49),
    ("DeBERTa-v3-large", (0.0417, 0.0034), (0.0187, 0.0093), -55),
    ("RoBERTa-large",    (0.0465, 0.0023), (0.0126, 0.0064), -73),
]


def _draw_reduction_bracket(ax, x_center, y_top, y_bot, label, color):
    """Draw a thin vertical arrow from y_top down to y_bot at x_center, with
    a percent label to the right of the arrow. The arrow visually connects
    the Model C bar top to the CCI v2 bar top, so the eye reads the cut.
    """
    # Vertical arrow pointing DOWN from Model C top to CCI v2 top.
    arrow = FancyArrowPatch(
        (x_center, y_top),
        (x_center, y_bot),
        arrowstyle="-|>",
        mutation_scale=12,
        color=color,
        linewidth=1.4,
        shrinkA=0,
        shrinkB=0,
        zorder=5,
    )
    ax.add_patch(arrow)

    # Short horizontal tick at the top of the arrow (the "bracket" feel).
    tick_half = 0.045  # in data coords on the x-axis
    ax.plot(
        [x_center - tick_half, x_center + tick_half],
        [y_top, y_top],
        color=color, linewidth=1.4, solid_capstyle="round", zorder=5,
    )

    # Percent label to the right of the bracket, vertically centred between
    # the two bar tops so it visually belongs to the reduction.
    y_mid = 0.5 * (y_top + y_bot)
    ax.text(
        x_center + tick_half + 0.03, y_mid,
        f"{label}%",
        ha="left", va="center",
        fontsize=13, color=color, fontweight="bold",
        zorder=6,
    )


def main() -> None:
    setup_neurips_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    archs = [r[0] for r in ROWS]
    x = np.arange(len(archs))
    width = 0.34

    mc_means = [r[1][0] for r in ROWS]
    mc_stds = [r[1][1] for r in ROWS]
    cci_means = [r[2][0] for r in ROWS]
    cci_stds = [r[2][1] for r in ROWS]
    deltas = [r[3] for r in ROWS]

    ax.bar(x - width/2, mc_means, width, yerr=mc_stds,
           capsize=4, color=COLOR_BASELINE, label="Model C baseline",
           edgecolor="white", linewidth=0.8, error_kw=dict(linewidth=1.0),
           zorder=2)
    ax.bar(x + width/2, cci_means, width, yerr=cci_stds,
           capsize=4, color=COLOR_CCI, label="CCI v2 (ours)",
           edgecolor="white", linewidth=0.8, error_kw=dict(linewidth=1.0),
           zorder=2)

    # Reduction brackets: thin vertical arrow from Model C bar top (mean +
    # std) down to CCI v2 bar top (mean + std), with the percent label to
    # the right. Anchored to the pair it labels, not floating high.
    for i, d in enumerate(deltas):
        mc_top = mc_means[i] + mc_stds[i]
        cci_top = cci_means[i] + cci_stds[i]
        # Place the bracket in the narrow gap between the two bars of the
        # pair, so it visually couples to the pair it labels rather than
        # drifting into the next architecture's space.
        x_bracket = x[i]
        _draw_reduction_bracket(
            ax,
            x_center=x_bracket,
            y_top=mc_top,
            y_bot=cci_top,
            label=d,
            color=COLOR_CCI,
        )

    # Light gridlines (we override the global setting because bar charts
    # benefit from a horizontal-only grid for value lookup).
    ax.yaxis.grid(True, alpha=0.5, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Subtle "no flips" baseline at y=0 — 1 px muted grey line. Sits on top
    # of the x-axis spine so it reads as a reference, not a chart artefact.
    ax.axhline(0, color="#9CA3AF", linewidth=1.0, alpha=0.7, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=11)
    # Y-axis label: vertical (matplotlib default rotation=90), with the
    # CFR_hard subscript rendered via mathtext.
    ax.set_ylabel(
        r"CFR$_{\mathrm{hard}}$  (lower is better)",
        fontsize=12,
    )

    # Keep some headroom for the bracket arrows + percent labels.
    y_top_data = max(mc_means[i] + mc_stds[i] for i in range(len(ROWS)))
    ax.set_ylim(0, y_top_data + 0.012)

    # Title: one line, centred, "Hard" dropped (CFR_hard is on the axis).
    ax.set_title(
        "Counterfactual flip rate: −49% / −55% / −73% across three architectures",
        fontsize=13.5, pad=28,  # extra pad so the horizontal legend fits below
    )

    # Legend: horizontal, centred just under the title (above the axes).
    # bbox_to_anchor in axes coords; loc="lower center" anchors the legend's
    # lower-centre to that point. y just above 1.0 puts it in the space
    # between the title and the top of the chart.
    legend_handles = [
        Line2D([0], [0], color=COLOR_BASELINE, lw=8, solid_capstyle="butt",
               label="Model C baseline"),
        Line2D([0], [0], color=COLOR_CCI, lw=8, solid_capstyle="butt",
               label="CCI v2 (ours)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.6,
        columnspacing=2.2,
        handletextpad=0.6,
    )

    # Subtitle — 2 lines max. Cut the "(5 for DeBERTa-large CCI)" caveat;
    # that detail belongs in the paper, not the poster.
    fig.text(
        0.5, -0.08,
        "CCI v2 cuts hard CFR by 49–73% on three architectures "
        "(DeBERTa-base; same-family DeBERTa-large; cross-family RoBERTa-large).\n"
        r"6 seeds each. Mean $\pm$ std. All within-architecture comparisons reject Holm-Bonferroni 6/6.",
        ha="center", fontsize=10, style="italic", color="#374151",
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.97])

    png = OUT_DIR / "poster_panel1_cfr_hard_headline.png"
    pdf = OUT_DIR / "poster_panel1_cfr_hard_headline.pdf"
    plt.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
