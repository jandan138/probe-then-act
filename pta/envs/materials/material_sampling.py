"""Material parameter sampling for ID and OOD evaluation splits.

In-distribution (ID) parameters are sampled from the training range.
Out-of-distribution (OOD) parameters are sampled from a held-out
range that is strictly outside the training range, ensuring a clean
generalisation test.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pta.envs.materials.material_family import MaterialFamily, MaterialParams


def sample_material_params(
    family: MaterialFamily,
    split: Literal["train", "id_test", "ood_parameter", "ood_family"],
    rng: np.random.Generator | None = None,
) -> MaterialParams:
    """Sample material parameters for a given family and data split.

    Parameters
    ----------
    family:
        The material family to sample from.
    split:
        Which data split to sample for:
        - ``"train"`` / ``"id_test"``: in-distribution ranges.
        - ``"ood_parameter"``: same family, parameters outside
          training range.
        - ``"ood_family"``: reserved for novel families not seen
          during training.
    rng:
        Numpy random generator for reproducibility.

    Returns
    -------
    MaterialParams
        Sampled parameter set ready for :func:`create_mpm_material`.
    """
    raise NotImplementedError
