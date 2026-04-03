"""File I/O helpers — YAML, JSON, pickle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : Path
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.
    """
    raise NotImplementedError


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """Serialise a dictionary to a YAML file.

    Parameters
    ----------
    data : dict[str, Any]
        Data to serialise.
    path : Path
        Destination file path.
    """
    raise NotImplementedError


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dictionary.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict[str, Any]
        Parsed JSON content.
    """
    raise NotImplementedError


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Serialise a dictionary to a JSON file.

    Parameters
    ----------
    data : dict[str, Any]
        Data to serialise.
    path : Path
        Destination file path.
    """
    raise NotImplementedError


def load_pickle(path: Path) -> Any:
    """Load a pickled Python object from disk.

    Parameters
    ----------
    path : Path
        Path to the pickle file.

    Returns
    -------
    Any
        The deserialised object.
    """
    raise NotImplementedError


def save_pickle(obj: Any, path: Path) -> None:
    """Serialise a Python object to a pickle file.

    Parameters
    ----------
    obj : Any
        Object to serialise.
    path : Path
        Destination file path.
    """
    raise NotImplementedError
