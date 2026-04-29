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
               settle_steps: int, res: tuple[int, int]) -> None:
    import genesis as gs
    if not gs._initialized:
        gs.init(
            backend=gs.cpu,
            precision="32",
            logging_level="warning",
            performance_mode=False,
        )

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
    args = parser.parse_args()

    params = {"E": args.E, "nu": args.nu, "rho": args.rho}
    render_one(args.material, params, args.out, args.settle,
               (args.width, args.height))


if __name__ == "__main__":
    main()
