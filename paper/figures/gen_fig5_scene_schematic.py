"""Compose the three Genesis renders (sand / snow / elastoplastic) into a
single 3-panel figure for the paper, with publication-grade typography
and on-panel annotations (subfigure labels, material titles, MPM
parameters, scripted-baseline transfer).

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
        "params":   r"$E\!=\!5\!\times\!10^{4}$  $\nu\!=\!0.30$  $\rho\!=\!2000$",
        "scripted": "Scripted: 32\\%",
    },
    {
        "file":     "scene_snow.png",
        "label":    "Snow",
        "subtitle": "Cohesive aggregate",
        "params":   r"$E\!=\!1\!\times\!10^{5}$  $\nu\!=\!0.20$  $\rho\!=\!400$",
        "scripted": "Scripted: 87\\%",
    },
    {
        "file":     "scene_elastoplastic.png",
        "label":    "Elastoplastic",
        "subtitle": "Recoverable rebound",
        "params":   r"$E\!=\!5\!\times\!10^{4}$  $\nu\!=\!0.40$  $\rho\!=\!1000$",
        "scripted": "Scripted: 70\\%",
    },
]

SUB_LABELS = ["(a)", "(b)", "(c)"]


def main() -> None:
    here = Path(__file__).parent
    src_dir = here / "genesis_renders"
    out_path = here / "fig5_scene_schematic.pdf"

    # Native size = single-column textwidth so embedded text renders at
    # paper-target point sizes without LaTeX rescaling.
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.30),
                             gridspec_kw={"wspace": 0.035})

    for ax, panel, sub in zip(axes, PANELS, SUB_LABELS):
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

        # (a)/(b)/(c) sub-label, top-left corner.
        ax.text(
            0.04, 0.93, sub, transform=ax.transAxes,
            fontsize=8.0, fontweight="bold", color="white",
            family="serif", va="top", ha="left",
            bbox=dict(facecolor="#1c1c1c", edgecolor="none",
                      boxstyle="round,pad=0.18", alpha=0.92),
        )

        # Material name, top-center.
        ax.text(
            0.5, 0.95, panel["label"], transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color="white",
            family="serif", va="top", ha="center",
            bbox=dict(facecolor="#1c1c1c", edgecolor="none",
                      boxstyle="round,pad=0.20", alpha=0.92),
        )

        # MPM parameters, bottom-right.
        ax.text(
            0.96, 0.05, panel["params"], transform=ax.transAxes,
            fontsize=5.6, color="white",
            family="serif", va="bottom", ha="right",
            bbox=dict(facecolor="#1c1c1c", edgecolor="none",
                      boxstyle="round,pad=0.14", alpha=0.88),
        )

        # Scripted-baseline tag, bottom-left.
        ax.text(
            0.04, 0.05, panel["scripted"], transform=ax.transAxes,
            fontsize=6.0, color="white", fontweight="bold",
            family="serif", va="bottom", ha="left",
            bbox=dict(facecolor="#1A5F28", edgecolor="none",
                      boxstyle="round,pad=0.16", alpha=0.92),
        )

    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.005, right=0.995,
                        wspace=0.04)
    fig.savefig(out_path, format="pdf", bbox_inches="tight",
                pad_inches=0.02, dpi=300)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
