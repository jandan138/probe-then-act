"""Compose the three Genesis renders (sand / snow / elastoplastic) into a
single 3-panel figure for the paper.

Inputs:  figures/genesis_renders/scene_{sand,snow,elastoplastic}.png
Output:  figures/fig5_scene_schematic.pdf
         (same filename as the previous schematic so the LaTeX include
         path stays unchanged)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread


PANELS = [
    ("scene_sand.png",          "Sand (granular)"),
    ("scene_snow.png",          "Snow (cohesive)"),
    ("scene_elastoplastic.png", "Elastoplastic\n(recoverable rebound)"),
]


def main() -> None:
    here = Path(__file__).parent
    src_dir = here / "genesis_renders"
    out_path = here / "fig5_scene_schematic.pdf"

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for ax, (fname, label) in zip(axes, PANELS):
        img = imread(src_dir / fname)
        # Crop to the interesting lower-left region (where the platform +
        # pile + arm sit).  Genesis renders the scene into the full
        # 1280x960 canvas but the action is in the lower-left quadrant.
        h, w = img.shape[:2]
        # Keep the full width but trim ~12% from the top, as the upper
        # area is sky.
        img = img[int(0.12 * h):, :]
        ax.imshow(img)
        ax.set_title(label, fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    fig.suptitle(
        "Edge-push benchmark: Genesis renders across material families",
        fontsize=10, y=1.00,
    )
    fig.subplots_adjust(top=0.86, bottom=0.02, left=0.01, right=0.99,
                         wspace=0.04)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=200)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
