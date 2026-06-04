#!/usr/bin/env python3
"""Poster Panel 2 (NeurIPS-style v3) — score compression KDE.

Refinements from v2:
  - Median shift now visualised as an explicit horizontal bracket at the
    top of the chart ("+0.08 → median shift"), not as two separate boxed
    labels that overlapped the KDE curves.
  - Legend moved to upper-right with a thin frame (so it has visual
    separation from the KDE peak without obscuring it).
  - "Default cutoff" marker is a small down-pointing triangle at the top
    of the x-axis instead of a bottom label that runs into the area fill.
  - Subtitle compressed to 2 lines.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from _poster_style import (
    COLOR_BASELINE,
    COLOR_CCI,
    setup_neurips_style,
)


HERE = Path(__file__).resolve().parent.parent
METRICS_ROOT = HERE / "results" / "metrics_from_rcp"
OUT_DIR = HERE / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46, 47]

SOURCES = [
    ("Model C baseline", "model_c_deberta_focal_mask", "model_c_deberta_focal_mask", COLOR_BASELINE),
    ("CCI v2 (ours)",    "cci_v2_fixed_js",            "cci_v2_fixed_js",            COLOR_CCI),
]


def load_probs(ckpt_prefix: str, experiment_name: str) -> np.ndarray:
    probs = []
    for seed in SEEDS:
        candidates = [
            METRICS_ROOT / f"cross_dataset_{ckpt_prefix}_seed{seed}" / experiment_name / "predictions_test.csv",
            METRICS_ROOT / f"cross_dataset_{ckpt_prefix}_seed{seed}" / "predictions_test.csv",
            METRICS_ROOT / f"cross_dataset_seed{seed}" / ckpt_prefix / "predictions_test.csv",
        ]
        for c in candidates:
            if c.exists():
                df = pd.read_csv(c)
                probs.append(df["prob_positive"].to_numpy(dtype=np.float64))
                break
    if not probs:
        raise FileNotFoundError(f"No predictions_test.csv for {ckpt_prefix}")
    return np.concatenate(probs)


def main() -> None:
    setup_neurips_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.linspace(0, 1, 600)
    medians = []
    for label, prefix, exp, color in SOURCES:
        probs = load_probs(prefix, exp)
        reflected = np.concatenate([-probs, probs, 2 - probs])
        kde = gaussian_kde(reflected, bw_method=0.06)
        density = kde(x) * 3.0
        ax.fill_between(x, density, color=color, alpha=0.18, zorder=2,
                        linewidth=0)
        ax.plot(x, density, color=color, linewidth=2.4, label=label, zorder=3)
        medians.append((label, color, float(np.median(probs))))

    # Median lines — thin, dashed.
    for label, color, median in medians:
        ax.axvline(median, color=color, linestyle="--", linewidth=1.4,
                   alpha=0.85, zorder=4)

    # Horizontal "shift bracket" connecting the two medians at top.
    m_mc = medians[0][2]
    m_cci = medians[1][2]
    shift = m_cci - m_mc
    y_bracket = 0.93   # in axes coords
    bracket_color = "#374151"
    ax.annotate(
        "", xy=(m_cci, y_bracket), xytext=(m_mc, y_bracket),
        xycoords=ax.get_xaxis_transform(),
        arrowprops=dict(
            arrowstyle="<|-|>", linewidth=1.4,
            color=bracket_color, mutation_scale=14,
        ),
    )
    # Center label on the bracket
    mid_x = 0.5 * (m_mc + m_cci)
    ax.text(
        mid_x, y_bracket + 0.04,
        f"median shift  +{shift:.2f}",
        transform=ax.get_xaxis_transform(),
        fontsize=11, fontweight="semibold", color=bracket_color,
        ha="center", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", pad=2),
    )

    # Tiny value labels at the foot of each median line, just above x-axis.
    for label, color, median in medians:
        ax.text(median, -0.05, f"{median:.2f}",
                transform=ax.get_xaxis_transform(),
                fontsize=10, fontweight="semibold", color=color,
                ha="center", va="top")

    # Default-cutoff marker: small down-triangle at the top of the chart
    # (clearer than a bottom italic label that fights the KDE area fill).
    ax.scatter([0.5], [y_bracket - 0.08],
               transform=ax.get_xaxis_transform(),
               marker="v", s=90, color="#4B5563", zorder=5)
    ax.text(0.5, y_bracket - 0.13, "default\ncutoff",
            transform=ax.get_xaxis_transform(),
            fontsize=9, color="#4B5563", ha="center", va="top",
            style="italic")

    ax.yaxis.grid(True, alpha=0.5, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Predicted probability  $p(\\mathrm{antisemitic} \\mid x)$",
                  fontsize=12)
    ax.set_ylabel("Density (in-distribution test set)", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title(
        "Score compression: the visible mechanism behind the threshold shift",
        fontsize=13.5, pad=18,
    )

    fig.text(
        0.5, -0.08,
        r"DeBERTa-v3-base, 6 seeds pooled ($n{\approx}6{,}360$ test predictions per model)." "\n"
        r"JS pair-invariance pulls predictions toward the swap-pair mean — $\bf{raising}$"
        r" the median from $0.09$ to $0.17$, flattening the negative-confident peak.",
        ha="center", fontsize=10, style="italic", color="#374151",
    )

    # Move legend up-right with a thin separator border, so it has its own
    # space and doesn't fight the KDE peak.
    leg = ax.legend(loc="upper right", fontsize=10.5, frameon=True,
                    framealpha=0.95, edgecolor="#D1D5DB")
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    png = OUT_DIR / "poster_panel2_score_compression.png"
    pdf = OUT_DIR / "poster_panel2_score_compression.pdf"
    plt.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
