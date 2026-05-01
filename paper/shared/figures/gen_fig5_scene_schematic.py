"""Compose the three Genesis renders (sand / snow / elastoplastic) into a
single 3-panel figure for the paper, with publication-grade typography and
separate metadata footers so labels do not obscure the renders.

Inputs:  figures/genesis_renders/scene_{sand,snow,elastoplastic}.png
Output:  figures/fig5_scene_schematic.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Match the rest of the paper figures (serif Times, stix mathtext, 300 dpi).
sys.path.insert(0, str(Path(__file__).parent))
import paper_plot_style  # noqa: F401  (loads rcParams)

import matplotlib.pyplot as plt
from matplotlib.image import imread


PANELS = [
    {
        "file":     "scene_sand.png",
        "label":    "Sand",
        "subtitle": "Granular, non-cohesive",
        "params":   r"$E=5\times10^{4}$, $\nu=0.30$, $\rho=2000$",
        "scripted": "Scripted transfer: 32%",
    },
    {
        "file":     "scene_snow.png",
        "label":    "Snow",
        "subtitle": "Cohesive aggregate",
        "params":   r"$E=1\times10^{5}$, $\nu=0.20$, $\rho=400$",
        "scripted": "Scripted transfer: 87%",
    },
    {
        "file":     "scene_elastoplastic.png",
        "label":    "Elastoplastic",
        "subtitle": "Recoverable rebound",
        "params":   r"$E=5\times10^{4}$, $\nu=0.40$, $\rho=1000$",
        "scripted": "Scripted transfer: 70%",
    },
]

SUB_LABELS = ["(a)", "(b)", "(c)"]


def main() -> None:
    here = Path(__file__).parent
    src_dir = here / "genesis_renders"
    out_path = here / "fig5_scene_schematic.pdf"

    # Native size is close to the LaTeX include width.  Keeping metadata in a
    # dedicated footer prevents labels from fighting the image content.
    fig = plt.figure(figsize=(5.4, 2.05))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 0.26],
        hspace=0.05,
        wspace=0.035,
    )

    for idx, (panel, sub) in enumerate(zip(PANELS, SUB_LABELS)):
        ax = fig.add_subplot(gs[0, idx])
        img = imread(src_dir / panel["file"])
        h, w = img.shape[:2]
        # Aggressive crop: cut sky (top ~22%) and trim sides so the
        # platform/pile/arm fill more of the panel.
        img = img[int(0.22 * h):int(0.98 * h), int(0.08 * w):int(0.92 * w)]
        ax.imshow(img, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#1c1c1c")
            spine.set_linewidth(0.5)

        # Consistent panel label and material name, kept away from the crop.
        ax.text(
            0.025, 1.055, sub, transform=ax.transAxes,
            fontsize=8.0, fontweight="bold", color="#1c1c1c",
            family="serif", va="bottom", ha="left", clip_on=False,
        )
        ax.text(
            0.5, 1.055, panel["label"], transform=ax.transAxes,
            fontsize=8.2, fontweight="bold", color="#1c1c1c",
            family="serif", va="bottom", ha="center", clip_on=False,
        )

        footer = fig.add_subplot(gs[1, idx])
        footer.set_facecolor("#F6F7F4")
        footer.set_xticks([])
        footer.set_yticks([])
        for spine in footer.spines.values():
            spine.set_edgecolor("#D4D9D2")
            spine.set_linewidth(0.45)

        footer.text(
            0.5, 0.73, panel["scripted"], transform=footer.transAxes,
            fontsize=6.6, color="#2F5948", fontweight="bold",
            family="serif", va="center", ha="center",
            bbox=dict(facecolor="#E4EEE7", edgecolor="#B9C9BE",
                      linewidth=0.35, boxstyle="round,pad=0.18"),
        )
        footer.text(
            0.5, 0.32, panel["params"], transform=footer.transAxes,
            fontsize=5.8, color="#333333",
            family="serif", va="center", ha="center",
        )

    fig.subplots_adjust(top=0.88, bottom=0.04, left=0.01, right=0.99)
    fig.savefig(out_path, format="pdf", bbox_inches="tight",
                pad_inches=0.02, dpi=300)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
