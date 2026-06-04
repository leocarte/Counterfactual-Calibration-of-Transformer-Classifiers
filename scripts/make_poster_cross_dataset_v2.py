#!/usr/bin/env python3
"""Poster cross-dataset figure v2 — AUROC + Macro-F1 dumbbells.

Reads analysis/direction1/cross_dataset_aggregate.json (committed, full
precision) and renders a 2-panel dumbbell figure:

  Left panel : AUROC     (prevalence-invariant ranking quality)
  Right panel: Macro-F1  (hate-speech subfield convention)

Each panel has 6 rows = 3 architectures x 2 datasets, grouped by dataset
(HateXplain on top, ToxiGen below a divider). Each row is a dumbbell from
Model C (grey) to CCI v2 (blue); arrow colour encodes win (green) / loss
(red) / tie (grey).

Honest framing per the verified numbers:
  - HateXplain: CCI v2 (DeBERTa-v3-large) wins on BOTH metrics -> starred.
  - ToxiGen: Model C wins AUROC/Macro-F1 on the large backbones; the two
    methods are within seed noise. No star.

AUROC is the 15-bin Mann-Whitney approximation (see aggregate_cross_dataset.py).

Run:
    POSTER_MODE=1 python scripts/make_poster_cross_dataset_v2.py
    POSTER_MODE=0 python scripts/make_poster_cross_dataset_v2.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

POSTER_MODE = os.environ.get("POSTER_MODE", "1") == "1"

OKABE_GREY  = "#6B7280"
OKABE_BLUE  = "#0072B2"
OKABE_GREEN = "#009E73"
OKABE_VERMIL = "#D55E00"
GOLD = "#F0C419"

HERE = Path(__file__).resolve().parent.parent
AGG_JSON = HERE / "analysis" / "direction1" / "cross_dataset_aggregate.json"
OUT_DIR = HERE / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (display label, Model C cell key, CCI v2 cell key)
ARCHS = [
    ("base",             "model_c_deberta_focal_mask",  "cci_v2_fixed_js"),
    ("DeBERTa-v3-large", "model_c_deberta_v3_large",    "cci_v2_fixed_js_deberta_v3_large"),
    ("RoBERTa-large",    "model_c_roberta_large",       "cci_v2_fixed_js_roberta_large"),
]
DATASETS = [
    ("HateXplain", "HateXplain-Jewish  (n=2,543, 68% +)"),
    ("ToxiGen",    "ToxiGen-Jewish  (n=684, 43% +)"),
]
METRICS = [("auroc_approx", "AUROC  (prevalence-invariant)"),
           ("macro_f1",     "Macro-F$_1$  (subfield standard)")]

TIE_EPS = 0.0015  # |delta| below this is a tie


def setup_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        "font.size": 16 if POSTER_MODE else 8,
        "axes.titlesize": 19 if POSTER_MODE else 9,
        "axes.labelsize": 17 if POSTER_MODE else 9,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 15 if POSTER_MODE else 8,
        "ytick.labelsize": 15 if POSTER_MODE else 8,
        "legend.fontsize": 14 if POSTER_MODE else 8,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.linewidth": 1.2, "axes.edgecolor": "#1F2937",
        "axes.facecolor": "white", "figure.facecolor": "white",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": False,
        "xtick.major.size": 6, "xtick.minor.size": 3,
        "xtick.major.width": 1.0, "xtick.minor.width": 0.7,
        "xtick.minor.visible": True,
        "savefig.dpi": 240, "savefig.bbox": "tight",
    })


def win_color(delta: float) -> str:
    if abs(delta) <= TIE_EPS:
        return OKABE_GREY
    return OKABE_GREEN if delta > 0 else OKABE_VERMIL


def main():
    setup_style()
    with open(AGG_JSON) as f:
        agg = json.load(f)

    # Row y-positions: 3 HateXplain (top), gap, 3 ToxiGen (bottom).
    y_hx = [6.5, 5.5, 4.5]
    y_tg = [2.5, 1.5, 0.5]
    rows = []  # (y, dataset_key, arch_label, mc_key, cci_key)
    for y, (label, mc, cci) in zip(y_hx, ARCHS):
        rows.append((y, "HateXplain", label, mc, cci))
    for y, (label, mc, cci) in zip(y_tg, ARCHS):
        rows.append((y, "ToxiGen", label, mc, cci))

    if POSTER_MODE:
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        # Determine x-range across this metric's values for tight zoom.
        vals = []
        for _, ds, _, mc, cci in rows:
            vals.append(agg[mc][ds][metric_key]["mean"])
            vals.append(agg[cci][ds][metric_key]["mean"])
        xmin = min(vals) - 0.04
        xmax = max(vals) + 0.06

        for y, ds, arch, mc, cci in rows:
            v_mc = agg[mc][ds][metric_key]["mean"]
            v_cci = agg[cci][ds][metric_key]["mean"]
            delta = v_cci - v_mc
            col = win_color(delta)
            is_star = (ds == "HateXplain" and arch == "DeBERTa-v3-large")

            # Connector line (Model C -> CCI v2)
            ax.plot([v_mc, v_cci], [y, y], color=col, lw=3.0, alpha=0.55,
                    solid_capstyle="round", zorder=2)
            # Model C dot
            ax.scatter(v_mc, y, s=150, c="white", edgecolors=OKABE_GREY,
                       linewidths=2.2, zorder=3)
            ax.scatter(v_mc, y, s=42, c=OKABE_GREY, zorder=4)
            # CCI v2 dot
            ax.scatter(v_cci, y, s=190 if is_star else 150, c="white",
                       edgecolors=OKABE_BLUE, linewidths=2.4, zorder=3)
            ax.scatter(v_cci, y, s=54 if is_star else 42, c=OKABE_BLUE, zorder=4)
            if is_star:
                ax.scatter(v_cci, y, marker="*", s=420, c=GOLD,
                           edgecolors="black", linewidths=0.8, zorder=6)

            # Value labels: Model C on its outer side, CCI v2 on its outer side.
            lo, hi = (v_mc, v_cci) if v_mc <= v_cci else (v_cci, v_mc)
            lo_is_mc = v_mc <= v_cci
            ax.text(lo - 0.006, y, f"{lo:.3f}", ha="right", va="center",
                    fontsize=12 if POSTER_MODE else 7,
                    color=OKABE_GREY if lo_is_mc else OKABE_BLUE, zorder=7)
            ax.text(hi + 0.010, y, f"{hi:.3f}", ha="left", va="center",
                    fontsize=12 if POSTER_MODE else 7,
                    fontweight="bold" if (not lo_is_mc) else "normal",
                    color=OKABE_BLUE if not lo_is_mc else OKABE_GREY, zorder=7)

        # Dataset divider + group labels
        ax.axhline(3.5, color="#9CA3AF", lw=0.8, ls=(0, (4, 3)), alpha=0.7)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.3, 7.3)
        ax.set_yticks([r[0] for r in rows])
        ax.set_yticklabels([r[2] for r in rows])
        ax.set_xlabel(metric_title, labelpad=7)
        ax.grid(True, axis="x", alpha=0.22)

    # Dataset band labels on the far left of the left axis
    axes[0].text(-0.40, 5.5, "HateXplain", transform=axes[0].get_yaxis_transform(),
                 rotation=90, ha="center", va="center",
                 fontsize=15 if POSTER_MODE else 8, fontweight="bold",
                 color="#1E3A8A")
    axes[0].text(-0.40, 1.5, "ToxiGen", transform=axes[0].get_yaxis_transform(),
                 rotation=90, ha="center", va="center",
                 fontsize=15 if POSTER_MODE else 8, fontweight="bold",
                 color="#92400E")

    # Figure-level legend
    handles = [
        Line2D([], [], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=OKABE_GREY, markeredgewidth=2.2, markersize=12,
               label="Model C (focal + kw-mask)", linewidth=0),
        Line2D([], [], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=OKABE_BLUE, markeredgewidth=2.4, markersize=12,
               label="CCI v2 (proposed)", linewidth=0),
        Line2D([], [], color=OKABE_GREEN, lw=3, label="CCI v2 wins"),
        Line2D([], [], color=OKABE_VERMIL, lw=3, label="CCI v2 loses"),
        Line2D([], [], color=OKABE_GREY, lw=3, label=f"tie (|Δ| ≤ {TIE_EPS})"),
        Line2D([], [], marker="*", color="w", markerfacecolor=GOLD,
               markeredgecolor="black", markeredgewidth=0.8, markersize=18,
               label="CCI v2 wins both metrics", linewidth=0),
    ]

    if POSTER_MODE:
        fig.suptitle("Cross-dataset transfer — Model C → CCI v2 across two robust metrics",
                     fontsize=21, fontweight="semibold", y=0.99)
        fig.legend(handles=handles, loc="lower center", ncol=6,
                   bbox_to_anchor=(0.5, 0.005), framealpha=0.97,
                   edgecolor="#1F2937", handletextpad=0.5, columnspacing=1.4)
        fig.text(0.5, 0.075,
                 "CCI v2 (DeBERTa-v3-large) beats its same-architecture baseline on HateXplain-Jewish on BOTH metrics (starred). "
                 "On ToxiGen-Jewish the two methods are within seed noise — Model C edges AUROC/Macro-F$_1$ on the large backbones. "
                 "Dots = seed-mean (5–6 seeds); AUROC is the 15-bin Mann-Whitney approximation.",
                 ha="center", fontsize=12.5, color="#374151", style="italic")
        plt.subplots_adjust(top=0.91, bottom=0.20, left=0.13, right=0.97, wspace=0.42)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=6,
                   bbox_to_anchor=(0.5, -0.02), fontsize=7)
        plt.tight_layout()

    for ext in ("png", "svg", "pdf"):
        out = OUT_DIR / f"xds_v2.{ext}"
        try:
            fig.savefig(out, dpi=240, bbox_inches="tight")
            print(f"wrote {out}")
        except PermissionError:
            alt = OUT_DIR / f"xds_v2_new.{ext}"
            fig.savefig(alt, dpi=240, bbox_inches="tight")
            print(f"wrote {alt}  (original locked)")
    plt.close(fig)


if __name__ == "__main__":
    main()
