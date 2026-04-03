"""Tool library -- Registry of available scoop/tool geometries.

Each tool is identified by a stable string key.  The registry maps
tool names to config dicts containing mesh path, default scale,
attach-link name, and meta-data used for train/test splitting.
"""

from __future__ import annotations

from typing import Any, Dict


#: Master registry of tool configurations.
#: Keys are stable tool identifiers; values are config dicts.
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "flat_scoop": {
        "mesh_path": "assets/tool_meshes/flat_scoop.obj",
        "scale": 1.0,
        "attach_link": "panda_hand",
        "split": "train",
    },
    "deep_scoop": {
        "mesh_path": "assets/tool_meshes/deep_scoop.obj",
        "scale": 1.0,
        "attach_link": "panda_hand",
        "split": "train",
    },
    "wide_scoop": {
        "mesh_path": "assets/tool_meshes/wide_scoop.obj",
        "scale": 1.0,
        "attach_link": "panda_hand",
        "split": "ood",
    },
    "narrow_scoop": {
        "mesh_path": "assets/tool_meshes/narrow_scoop.obj",
        "scale": 1.0,
        "attach_link": "panda_hand",
        "split": "ood",
    },
}


def get_tool_config(name: str) -> Dict[str, Any]:
    """Look up tool configuration by name.

    Parameters
    ----------
    name:
        A key in :data:`TOOL_REGISTRY`.

    Returns
    -------
    dict
        Tool configuration dict.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    raise NotImplementedError
