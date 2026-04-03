"""Launch a training run on the local machine."""

from __future__ import annotations

from typing import Any, Dict


def launch_local(config: Dict[str, Any]) -> None:
    """Start a training job on the local machine.

    Sets up logging directories, resolves device placement (CPU / GPU),
    and invokes the appropriate training entry point.

    Parameters
    ----------
    config : dict
        Full training configuration (environment, policy, optimiser,
        and launch-specific keys such as ``num_gpus``, ``log_dir``).
    """
    raise NotImplementedError
