"""Schematic of the Genesis edge-push scene used in the paper.

Top row: 3 top-down views (one per material family) showing source
platform, particle pile, edge, and target bin. Bottom row: a single
side view shared across materials, showing the source pedestal, pile,
scoop end-effector, and edge.

Geometry follows ``pta/envs/builders/scene_builder.py::_DEFAULT_CONFIG``:
- Source platform at (0.50, 0.00) with 0.15 x 0.15 base, 0.08 m wall height
- Target bin at (0.50, 0.35) with 0.12 x 0.12 base, 0.10 m wall height
- Particle pile centred at (0.55, 0.02) with 0.12 x 0.06 x 0.03 extent
- Franka base at the origin
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Circle
from matplotlib.lines import Line2D


# --- Scene geometry --------------------------------------------------------
SOURCE_POS = (0.50, 0.00)
SOURCE_BASE = (0.15, 0.15)
SOURCE_WALL_H = 0.08
TARGET_POS = (0.50, 0.35)
TARGET_BASE = (0.12, 0.12)
TARGET_WALL_H = 0.10
PILE_POS = (0.55, 0.02)
PILE_EXTENT = (0.12, 0.06, 0.03)
PLATFORM_TOP_Z = 0.05
ROBOT_BASE = (0.0, 0.0)


MATERIAL_PALETTE = {
    "Sand":          {"face": "#D8B07A", "edge": "#7A5A2C", "scatter": "#A8783E",
                      "subtitle": "granular"},
    "Snow":          {"face": "#E9F0F5", "edge": "#3F6B91", "scatter": "#9DB7CC",
                      "subtitle": "cohesive"},
    "Elastoplastic": {"face": "#C8C4DD", "edge": "#3C2F77", "scatter": "#5B4DAA",
                      "subtitle": "recoverable rebound"},
}


def _draw_pile_top(ax, palette, n_grains: int = 70, rng_seed: int = 0,
                   label: bool = True):
    rng = np.random.default_rng(rng_seed)
    px, py = PILE_POS
    pw, ph = PILE_EXTENT[0], PILE_EXTENT[1]
    ax.add_patch(Rectangle(
        (px - pw / 2, py - ph / 2), pw, ph,
        facecolor=palette["face"], edgecolor=palette["edge"],
        linewidth=1.0, zorder=4,
    ))
    xs = rng.uniform(px - pw / 2 + 0.005, px + pw / 2 - 0.005, size=n_grains)
    ys = rng.uniform(py - ph / 2 + 0.005, py + ph / 2 - 0.005, size=n_grains)
    ax.scatter(xs, ys, s=3.5, c=palette["scatter"],
               alpha=0.85, linewidths=0, zorder=5)
    if label:
        ax.annotate(
            "pile",
            xy=(px, py),
            xytext=(px + 0.13, py + 0.10),
            ha="left", va="center", fontsize=7, color="#333333",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#666666"),
            zorder=6,
        )


def _draw_top_down(ax, material_name: str):
    palette = MATERIAL_PALETTE[material_name]

    # Robot base — small filled circle with label below the panel
    ax.add_patch(Circle(ROBOT_BASE, 0.035, facecolor="#777777",
                        edgecolor="black", linewidth=0.8, zorder=2))
    ax.text(0.0, -0.10, "Franka\nbase", ha="center", va="top",
            fontsize=7, color="#333333")

    # Source platform (top view) — drawn first so the pile sits on top
    sx, sy = SOURCE_POS
    sw, sh = SOURCE_BASE
    ax.add_patch(Rectangle(
        (sx - sw / 2, sy - sh / 2), sw, sh,
        facecolor="#F2F2F2", edgecolor="black", linewidth=1.0, zorder=1,
    ))
    ax.text(sx, sy - sh / 2 - 0.018, "source platform",
            ha="center", va="top", fontsize=7, color="#333333")

    # Particle pile
    _draw_pile_top(ax, palette, rng_seed=42)

    # Target bin
    tx, ty = TARGET_POS
    tw, th = TARGET_BASE
    ax.add_patch(Rectangle(
        (tx - tw / 2, ty - th / 2), tw, th,
        facecolor="#FFFAEC", edgecolor="black", linewidth=1.0, zorder=1,
    ))
    ax.text(tx, ty + th / 2 + 0.012, "target bin",
            ha="center", fontsize=7, color="#333333")

    # Edge (top of source platform on +y side)
    edge_y = sy + sh / 2
    ax.plot([sx - sw / 2, sx + sw / 2], [edge_y, edge_y],
            color="#B22222", linewidth=2.0, zorder=6)
    ax.text(sx + sw / 2 + 0.012, edge_y, "edge", color="#B22222",
            va="center", fontsize=7)

    # Push arrow from pile to target
    px, py = PILE_POS
    pw, ph = PILE_EXTENT[0], PILE_EXTENT[1]
    ax.add_patch(FancyArrowPatch(
        (px, py + ph / 2 + 0.005), (tx, ty - th / 2 - 0.005),
        arrowstyle="->", mutation_scale=11,
        color="#222222", linewidth=1.0, zorder=7,
    ))

    ax.set_aspect("equal")
    ax.set_xlim(-0.10, 0.78)
    ax.set_ylim(-0.20, 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title(f"{material_name}\n({palette['subtitle']})",
                 fontsize=9, pad=2)


def _draw_side_view(ax):
    """Side view (X-Z) shared across materials."""
    # Floor
    ax.axhline(y=0.0, color="black", linewidth=0.8)

    sx, _ = SOURCE_POS
    sw = SOURCE_BASE[0]

    # Source platform pedestal (X-Z slab)
    ax.add_patch(Rectangle(
        (sx - sw / 2, 0.0), sw, PLATFORM_TOP_Z,
        facecolor="#DDDDDD", edgecolor="black", linewidth=0.8,
    ))
    ax.text(sx, -0.018, "source platform pedestal",
            ha="center", va="top", fontsize=7, color="#333333")

    # Source wall on the +x edge (the "edge" the scoop pushes the pile over)
    wall_t = 0.005
    ax.add_patch(Rectangle(
        (sx + sw / 2 - wall_t, PLATFORM_TOP_Z), wall_t, SOURCE_WALL_H,
        facecolor="#BBBBBB", edgecolor="black", linewidth=0.6,
    ))
    # Edge: top of the +x source wall
    edge_z = PLATFORM_TOP_Z + SOURCE_WALL_H
    ax.plot([sx + sw / 2 - wall_t, sx + sw / 2],
            [edge_z, edge_z], color="#B22222", linewidth=2.0, zorder=6)
    ax.text(sx + sw / 2 + 0.005, edge_z + 0.005, "edge",
            color="#B22222", fontsize=7)

    # Particle pile sitting on the platform.  In the real Genesis scene the
    # pile is initialised at x = 0.55 (config "Config D") so a small portion
    # extends past the +x platform wall, but for visual clarity here we draw
    # only the in-platform footprint.
    pw, _, pz = PILE_EXTENT
    pile_x_center = PILE_POS[0]
    pile_z_center = PLATFORM_TOP_Z + 0.005 + pz / 2
    platform_x_max = sx + sw / 2
    pile_x_left = pile_x_center - pw / 2
    pile_x_right = min(pile_x_center + pw / 2, platform_x_max - 0.001)
    pile_visible_w = pile_x_right - pile_x_left
    ax.add_patch(Rectangle(
        (pile_x_left, PLATFORM_TOP_Z + 0.005),
        pile_visible_w, pz,
        facecolor="#D8B07A", edgecolor="#7A5A2C", linewidth=0.8, zorder=4,
    ))
    ax.annotate(
        "particle pile",
        xy=(pile_x_left + pile_visible_w / 2, pile_z_center + pz / 2),
        xytext=(pile_x_left - 0.06, pile_z_center + 0.06),
        ha="right", fontsize=7,
        arrowprops=dict(arrowstyle="-", lw=0.6, color="#666666"),
    )

    # Scoop end-effector silhouette approaching from the -x side
    scoop_tip_x = pile_x_left - 0.005
    scoop_z = pile_z_center
    triangle = Polygon([
        (scoop_tip_x, scoop_z),
        (scoop_tip_x - 0.07, scoop_z - 0.012),
        (scoop_tip_x - 0.07, scoop_z + 0.012),
    ], closed=True, facecolor="#777777", edgecolor="black", linewidth=0.7, zorder=5)
    ax.add_patch(triangle)
    ax.annotate(
        "scoop EE",
        xy=(scoop_tip_x - 0.05, scoop_z),
        xytext=(scoop_tip_x - 0.05, scoop_z - 0.055),
        ha="center", fontsize=7,
        arrowprops=dict(arrowstyle="-", lw=0.6, color="#666666"),
    )

    # Push direction arrow
    ax.add_patch(FancyArrowPatch(
        (pile_x_right, pile_z_center),
        (pile_x_right + 0.05, pile_z_center),
        arrowstyle="->", mutation_scale=12,
        color="#222222", linewidth=1.0,
    ))

    ax.set_aspect("equal")
    ax.set_xlim(0.30, 0.78)
    ax.set_ylim(-0.09, 0.24)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title("Side view (X-Z, identical across materials)",
                 fontsize=9, pad=2)


def main() -> None:
    out_path = Path(__file__).parent / "fig5_scene_schematic.pdf"

    fig = plt.figure(figsize=(7.0, 5.0))
    fig.suptitle(
        "Edge-push benchmark: scene geometry across material families",
        fontsize=10, y=0.98,
    )

    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.35, 1.0],
        hspace=0.30, wspace=0.05,
        top=0.90, bottom=0.10, left=0.04, right=0.98,
    )

    materials = list(MATERIAL_PALETTE.keys())
    for col, name in enumerate(materials):
        ax = fig.add_subplot(gs[0, col])
        _draw_top_down(ax, name)

    ax_side = fig.add_subplot(gs[1, :])
    _draw_side_view(ax_side)

    handles = []
    for name, palette in MATERIAL_PALETTE.items():
        handles.append(Line2D([0], [0], marker="s", linestyle="",
                              markerfacecolor=palette["face"],
                              markeredgecolor=palette["edge"],
                              markersize=10,
                              label=f"{name} ({palette['subtitle']})"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.01))

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
