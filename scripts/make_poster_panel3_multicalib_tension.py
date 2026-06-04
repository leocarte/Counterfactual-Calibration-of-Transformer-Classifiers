#!/usr/bin/env python3
"""Poster Panel 3 (NeurIPS-style v3) — multicalibration ↔ counterfactual tension.

3-row dumbbell of ECE, CFR_hard, CFR_soft_L1. The +96% red arrow on
CFR_soft_L1 is the headline 5/5-novelty visual; the −54% green arrow on ECE
shows multicalibration doing what it's supposed to on the calibration axis.

v3 redesign (from review feedback):
- Legend moved below the chart, horizontal, with round dot markers.
- CFR_hard endpoint labels no longer overlap (extra horizontal padding,
  both labels on the side of their dot).
- "TENSION" bracket added on the LEFT outside the y-axis labels,
  linking the ECE and CFR_soft_L1 rows.
- Subtitle cut from 4 lines to 2, retaining the counterfactual-pairs
  sentence and dropping the F1+ aside and the "first quantitative
  report" claim.
- +96% arrow drawn thicker (lw=4) than −54% (lw=2.6), with a larger
  bold-italic percent label, to pull the eye to the regression.
- Inline value labels switched from "under the dot" to "left of raw dot,
  right of patched dot" — the row's vertical band is now arrow-only.
- Title shortened by one line.

Sourced from `analysis/direction1/REPORT.md` directly (not from rounded
paper tables); per-row percentages are computed from the unrounded values
so the side labels match the title to within rounding.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch  # noqa: F401 (kept for future)

from _poster_style import (
    COLOR_ACCENT,
    COLOR_BASELINE,
    COLOR_CCI,
    COLOR_GOOD,
    setup_neurips_style,
)


HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (metric_label, raw_value, patched_value, lower_is_better)
# Values from analysis/direction1/REPORT.md rows 47 (Model C raw) and 49
# (Model C + multicalibration). Unrounded so the side-label percentages
# round consistently with paper §3.3 tab:tension.
ROWS = [
    (r"ECE",                       0.0772, 0.0354, True),
    (r"CFR$_{\mathrm{hard}}$",     0.0358, 0.0330, True),
    (r"CFR$_{\mathrm{soft},L_1}$", 0.0322, 0.0630, True),
]

# Subtle accent grey for the TENSION bracket (not red, per brief).
COLOR_BRACKET = "#6B7280"


def _classify(raw: float, patched: float, lower_is_better: bool):
    """Return (arrow_color, percent_string, is_flat, is_regression)."""
    improved = (patched < raw) if lower_is_better else (patched > raw)
    flat = abs(raw - patched) < (raw * 0.10)  # less than 10% change → flat band
    if flat:
        return "#9CA3AF", r"$\approx$ 0", True, False
    arrow_color = COLOR_GOOD if improved else COLOR_ACCENT
    pct_change = 100.0 * (patched - raw) / raw
    sign = "+" if pct_change > 0 else "−"
    pct_str = f"{sign}{abs(pct_change):.0f}%"
    return arrow_color, pct_str, False, (not improved)


def main() -> None:
    setup_neurips_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))

    n = len(ROWS)
    y = np.arange(n)[::-1]

    # Padding for inline value labels (in data-x units). The "close-pair"
    # extra padding applies only when raw and patched are nearly coincident
    # (i.e. CFR_hard) so the two labels don't collide.
    PAD_DEFAULT = 0.0028
    PAD_CLOSE   = 0.0070
    CLOSE_THRESHOLD = 0.010  # |raw - patched| below this triggers extra padding

    # Per-row arrow / label styling — the regression row gets visual emphasis.
    for i, (label, raw, patched, lower_is_better) in enumerate(ROWS):
        yi = y[i]
        arrow_color, pct_str, is_flat, is_regression = _classify(
            raw, patched, lower_is_better
        )

        # Visual emphasis on the regression row (CFR_soft_L1, +96%).
        if is_regression:
            arrow_lw = 4.0
            arrow_mut = 24
            pct_fontsize = 18
            pct_fontweight = "bold"
            pct_fontstyle = "italic"
        else:
            arrow_lw = 2.6
            arrow_mut = 18
            pct_fontsize = 14
            pct_fontweight = "bold"
            pct_fontstyle = "normal"

        # Arrow body.
        ax.annotate(
            "", xy=(patched, yi), xytext=(raw, yi),
            arrowprops=dict(
                arrowstyle="-|>",
                linewidth=arrow_lw,
                color=arrow_color,
                shrinkA=7, shrinkB=7,
                mutation_scale=arrow_mut,
            ),
            zorder=3,
        )

        # Endpoint dots.
        ax.scatter([raw], [yi], s=140, color=COLOR_BASELINE, zorder=5,
                   edgecolor="white", linewidth=1.3)
        ax.scatter([patched], [yi], s=140, color=COLOR_CCI, zorder=5,
                   edgecolor="white", linewidth=1.3)

        # Inline value labels: LEFT of raw dot, RIGHT of patched dot.
        # When raw and patched are nearly coincident (CFR_hard), use
        # extra padding so the two labels don't overlap each other.
        close = abs(raw - patched) < CLOSE_THRESHOLD
        pad = PAD_CLOSE if close else PAD_DEFAULT

        # Pick which dot is the LEFT one and which is the RIGHT one,
        # so the raw value is always to the LEFT of its own dot and the
        # patched value is always to the RIGHT of its own dot.
        if raw <= patched:
            x_raw_label = raw - pad
            ha_raw = "right"
            x_patched_label = patched + pad
            ha_patched = "left"
        else:
            x_raw_label = raw + pad
            ha_raw = "left"
            x_patched_label = patched - pad
            ha_patched = "right"

        ax.text(x_raw_label, yi, f"{raw:.3f}",
                color=COLOR_BASELINE, fontsize=10.5,
                ha=ha_raw, va="center", fontweight="semibold")
        ax.text(x_patched_label, yi, f"{patched:.3f}",
                color=COLOR_CCI, fontsize=10.5,
                ha=ha_patched, va="center", fontweight="semibold")

        # Percent change label — placed past the rightmost endpoint label.
        rightmost = max(raw, patched) + pad
        # Extra room for the longer "0.063"-style label before the percent.
        x_pct = rightmost + 0.011
        ax.text(x_pct, yi, pct_str,
                fontsize=pct_fontsize, color=arrow_color,
                fontweight=pct_fontweight, fontstyle=pct_fontstyle,
                ha="left", va="center")

    # --- Axes setup ---
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=12)
    ax.set_xlabel("Metric value (mean across 6 Model C seeds)", fontsize=12)
    ax.set_xlim(0.0, 0.105)
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.grid(True, alpha=0.5, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Tick only out to 0.08 — past that we have inline % labels, not data.
    ax.set_xticks(np.arange(0.0, 0.085, 0.02))

    ax.set_title(
        r"Multicalibration: ECE $-54\%$, "
        r"CFR$_{\mathrm{soft},L_1}$ $+96\%$  "
        r"(a tension on Model C)",
        fontsize=14, pad=14,
    )

    # --- TENSION bracket on the LEFT, outside y-axis tick labels ---
    # Use axes coordinates so the bracket sits in the figure margin and
    # is robust to the data x-range. Connects the ECE row (top, y[0]=2)
    # and the CFR_soft_L1 row (bottom, y[2]=0).
    y_ece = y[0]
    y_soft = y[2]
    # Axes fraction for x: a small negative value sits outside the y-axis
    # spine, in the margin between the spine and the y-tick labels.
    # Instead of trying to compute the exact label width, we just push
    # well past it using a blended transform.
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    x_bracket = -0.14   # axes-fraction; well left of the y-tick labels
    x_tick_in = -0.125  # short inward tick at each end of the bracket
    x_text    = -0.155  # text label, slightly further left of the bracket spine

    # Vertical spine of the bracket
    ax.plot(
        [x_bracket, x_bracket], [y_ece, y_soft],
        transform=trans, color=COLOR_BRACKET, linewidth=1.3,
        solid_capstyle="butt", clip_on=False, zorder=4,
    )
    # Short horizontal tick at top (ECE row)
    ax.plot(
        [x_bracket, x_tick_in], [y_ece, y_ece],
        transform=trans, color=COLOR_BRACKET, linewidth=1.3,
        solid_capstyle="butt", clip_on=False, zorder=4,
    )
    # Short horizontal tick at bottom (CFR_soft_L1 row)
    ax.plot(
        [x_bracket, x_tick_in], [y_soft, y_soft],
        transform=trans, color=COLOR_BRACKET, linewidth=1.3,
        solid_capstyle="butt", clip_on=False, zorder=4,
    )
    # Rotated "TENSION" label
    y_mid = 0.5 * (y_ece + y_soft)
    ax.text(
        x_text, y_mid, r"$\updownarrow$  TENSION",
        transform=trans, color=COLOR_BRACKET,
        fontsize=11, fontweight="bold",
        ha="center", va="center", rotation=90,
        clip_on=False,
    )

    # --- Subtitle (2 lines, trimmed) ---
    fig.text(
        0.5, -0.06,
        r"Post-hoc multicalibration patcher fit on Model C's dev-set predictions, applied at inference." "\n"
        r"Counterfactual pairs $(x, x')$ live in different input-group cells, so the multicalibration patcher can drag them apart.",
        ha="center", fontsize=10.5, style="italic", color="#374151",
    )

    # --- Legend below chart, horizontal, round dot markers ---
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLOR_BASELINE, markeredgecolor="white",
               markeredgewidth=1.0, markersize=11,
               label="Model C raw"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLOR_CCI, markeredgecolor="white",
               markeredgewidth=1.0, markersize=11,
               label="Model C + multicalibration"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.005),
        frameon=False, fontsize=11.5,
        handletextpad=0.6, columnspacing=2.5,
    )

    plt.tight_layout(rect=[0.04, 0.10, 1, 1])

    png = OUT_DIR / "poster_panel3_multicalib_tension.png"
    pdf = OUT_DIR / "poster_panel3_multicalib_tension.pdf"
    plt.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
