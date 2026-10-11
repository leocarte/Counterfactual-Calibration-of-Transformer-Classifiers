#!/usr/bin/env python3
"""Composes the 4 finished poster panels into a 2x2 preview grid.

Layout (reading clockwise from top-left, matching the story arc proposed
in the design discussion):

  ┌─────────────────────┬─────────────────────┐
  │ Panel 1 (headline)  │ Panel 3 (novelty)   │
  │ CFR_hard reduction  │ Multicalib tension  │
  ├─────────────────────┼─────────────────────┤
  │ Panel 2 (mechanism) │ Panel 4 (punchline) │
  │ Score compression   │ Cross-dataset       │
  └─────────────────────┴─────────────────────┘

Uses PIL/Pillow to compose the existing PNGs side-by-side, padded to a
uniform per-cell size. Not for print (the print-quality version embeds
the four PDFs in poster.tex one cell at a time); just for at-a-glance
review of how they read together.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent.parent
FIG_DIR = HERE / "results" / "figures"
OUT = FIG_DIR / "poster_2x2_preview.png"

PANELS = [
    ["poster_panel1_cfr_hard_headline.png",   "poster_panel3_multicalib_tension.png"],
    ["poster_panel2_score_compression.png",   "poster_panel4_cross_dataset_slope.png"],
]

# White margin around the whole grid + between cells.
OUTER_MARGIN = 32
INNER_MARGIN = 24
# Target per-cell width — each panel image gets scaled to fit while
# preserving aspect ratio; cells are then padded to a uniform size.
CELL_WIDTH = 1400


def _load_scaled(path: Path, target_w: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    target_h = round(h * target_w / w)
    return img.resize((target_w, target_h), Image.LANCZOS)


def main() -> None:
    rows = []
    for row in PANELS:
        rows.append([_load_scaled(FIG_DIR / p, CELL_WIDTH) for p in row])

    # Each row's cells get the same height — the max height of the two.
    row_heights = [max(c.size[1] for c in r) for r in rows]
    grid_w = 2 * CELL_WIDTH + INNER_MARGIN + 2 * OUTER_MARGIN
    grid_h = sum(row_heights) + INNER_MARGIN + 2 * OUTER_MARGIN

    canvas = Image.new("RGB", (grid_w, grid_h), "white")

    y = OUTER_MARGIN
    for ri, row in enumerate(rows):
        x = OUTER_MARGIN
        for ci, cell in enumerate(row):
            # Vertically centre each cell in its row's reserved height,
            # so shorter panels don't anchor to the top.
            cell_y = y + (row_heights[ri] - cell.size[1]) // 2
            canvas.paste(cell, (x, cell_y))
            x += CELL_WIDTH + INNER_MARGIN
        y += row_heights[ri] + INNER_MARGIN

    canvas.save(OUT, "PNG", optimize=True)
    print(f"Wrote {OUT}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
