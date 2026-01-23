"""
Small numeric helpers.

The functions in this module are internal utilities used by multiple components.
They are prefixed with ``_`` to signal that they are not part of the stable public
API, but they are still documented to support Sphinx builds.
"""

from __future__ import annotations

import numpy as np
from .params.bounds import ParameterBoundsSpace


def _softmax(u: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute a numerically stable softmax of utilities.

    Parameters
    ----------
    u : numpy.ndarray
        Utility vector of shape ``(A,)``.
    beta : float
        Inverse temperature. Larger values produce more deterministic choices.

    Returns
    -------
    numpy.ndarray
        Probability vector of shape ``(A,)`` that sums to 1.

    Notes
    -----
    The implementation subtracts ``max(beta*u)`` before exponentiating for numerical
    stability.
    """
    z = beta * u
    z -= float(np.max(z))
    expz = np.exp(z)
    return expz / float(np.sum(expz))


def _as_scipy_bounds(space: ParameterBoundsSpace) -> list[tuple[float, float]]:
    """
    Convert a :class:`~comp_model_core.params.bounds.ParameterBoundsSpace` to SciPy bounds.

    Parameters
    ----------
    space : ParameterBoundsSpace
        Bounds space describing parameter order and box constraints.

    Returns
    -------
    list[tuple[float, float]]
        Bounds in the format expected by ``scipy.optimize.minimize(..., bounds=...)``.
    """
    return [(space.bounds[name].lo, space.bounds[name].hi) for name in space.names]
