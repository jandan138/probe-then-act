"""Render a single representative edge-push scene frame for the paper.

Builds a fresh scene with the given material family and parameters, lets
particles settle for ``--settle`` simulation steps, then renders one
high-resolution RGB frame from a fixed three-quarter camera angle and
saves it as a PNG.

Run **once per material** (Genesis state is best treated as
process-scoped).  The companion wrapper drives this for sand / snow /
elastoplastic and composites the results.

Example::

    python -m pta.scripts.render_paper_scene \\
        --material sand --E 5e4 --nu 0.3 --rho 2000 \\
        --out figures/genesis_renders/scene_sand.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


def _coerce_rgb(rgb: Any) -> np.ndarray:
    """Convert a Genesis camera RGB return into a uint8 numpy image."""
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    arr = np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    return arr


def render_one(material: str, params: Dict[str, float], out_path: Path,
               settle_steps: int, res: tuple[int, int],
               surface_color: tuple[float, float, float, float] | None = None) -> None:
    import os, genesis as gs
    # Force the OSMesa platform buffer to be at least as large as the requested
    # render resolution.  OffscreenRenderer reads this env var on init.
    os.environ.setdefault("GS_OFFSCREEN_INIT_RES", f"{res[0]}x{res[1]}")
    if not gs._initialized:
        gs.init(
            backend=gs.cpu,
            precision="32",
            logging_level="warning",
            performance_mode=False,
        )

    # Optional per-material override of the particle surface colour.  The
    # SceneBuilder hard-codes a golden-tan colour for all materials so a
    # paper figure cannot tell sand / snow / elastoplastic apart visually
    # without this override.  We monkey-patch ``_add_particles`` instead of
    # threading a new config key through the env so existing training code
    # is unchanged.
    if surface_color is not None:
        from pta.envs.builders import scene_builder as _sb
        _orig_add_particles = _sb.SceneBuilder._add_particles

        def _patched(self, scene, config):
            family = config.get("particle_material", "sand")
            params = config.get("particle_params", {})
            mat = self._material_builder.create_material(family, params)
            return scene.add_entity(
                material=mat,
                morph=gs.morphs.Box(
                    pos=config["particle_pos"], size=config["particle_size"],
                ),
                surface=gs.surfaces.Default(
                    color=surface_color, vis_mode="particle",
                ),
            )
        _sb.SceneBuilder._add_particles = _patched

    from pta.envs.builders.scene_builder import SceneBuilder

    cfg: Dict[str, Any] = {
        "n_envs": 1,
        "task_layout": "edge_push",
        "particle_material": material,
        "particle_params": params,
        "particle_pos": (0.55, 0.02, 0.20),
        "camera_res": res,
        "camera_pos": (1.20, -0.45, 0.85),
        "camera_lookat": (0.50, 0.18, 0.10),
        "camera_fov": 50,
    }

    builder = SceneBuilder()
    components = builder.build_scene(cfg)

    for _ in range(settle_steps):
        components.scene.step()

    rgb, _depth, _seg, _normal = components.camera.render(rgb=True)
    img = _coerce_rgb(rgb)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"saved {out_path} (shape={img.shape}, dtype={img.dtype})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", required=True,
                        choices=["sand", "snow", "elastoplastic"])
    parser.add_argument("--E", type=float, required=True,
                        help="Young's modulus (Pa)")
    parser.add_argument("--nu", type=float, required=True,
                        help="Poisson's ratio")
    parser.add_argument("--rho", type=float, required=True,
                        help="Density (kg/m^3)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output PNG path")
    parser.add_argument("--settle", type=int, default=120,
                        help="Number of physics steps before rendering")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--surface-color", type=str, default=None,
        help="Optional 'r,g,b,a' override for particle render colour "
             "(0..1 floats).  E.g. '0.85,0.70,0.40,1.0' for sand.",
    )
    args = parser.parse_args()

    color = None
    if args.surface_color is not None:
        try:
            parts = [float(s) for s in args.surface_color.split(",")]
            assert len(parts) == 4
            color = tuple(parts)
        except Exception:
            raise ValueError(
                f"--surface-color must be 'r,g,b,a' floats, got {args.surface_color!r}"
            )

    params = {"E": args.E, "nu": args.nu, "rho": args.rho}
    render_one(args.material, params, args.out, args.settle,
               (args.width, args.height), surface_color=color)


if __name__ == "__main__":
    main()
