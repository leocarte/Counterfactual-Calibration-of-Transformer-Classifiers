"""Shared NeurIPS-style figure defaults for the 4 poster panels.

Goals: muted palette (no flashy reds/golds), thin lines, light gridlines,
top/right spines off, sans-serif at moderate size. Readable from 1 m
without looking like a marketing slide.
"""
from __future__ import annotations

import matplotlib as mpl


# Muted, professional palette — chosen to read on white at distance without
# the "internal slide deck" feel of the v1 crimson + gold scheme.
COLOR_BASELINE = "#6B7280"   # cool neutral grey — Model C baseline
COLOR_CCI      = "#1F4E79"   # deep blue — CCI v2 (our method)
COLOR_ACCENT   = "#9E2A2B"   # subdued crimson — regressions only
COLOR_GOOD     = "#2E7D32"   # subdued green — improvements
COLOR_GRID     = "#D1D5DB"
COLOR_HIGHLIGHT_BG = "#EAF1F8"  # very pale blue for column highlights


def setup_neurips_style() -> None:
    """Apply NeurIPS-friendly matplotlib defaults. Idempotent — safe to call
    multiple times. Each panel script calls this at the top of `main()`.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#374151",
        "axes.facecolor": "white",
        "axes.grid": False,
        "figure.facecolor": "white",
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "lines.markersize": 7,
        "savefig.dpi": 220,
        "savefig.facecolor": "white",
    })
