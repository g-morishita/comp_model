"""
Random number generator helpers.

The library uses :class:`numpy.random.Generator` consistently. This module provides
a tiny wrapper to standardize seeding and creation of generators across components.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class RNG:
    """
    Simple wrapper for reproducible RNG creation.

    Parameters
    ----------
    seed : int or None, optional
        Seed passed to :func:`numpy.random.default_rng`. If ``None``, NumPy chooses
        a random seed.

    Attributes
    ----------
    seed : int or None
        Seed used to initialize the generator.
    """

    seed: int | None = None

    def numpy(self) -> np.random.Generator:
        """
        Create a NumPy RNG.

        Returns
        -------
        numpy.random.Generator
            A generator seeded with ``self.seed``.
        """
        return np.random.default_rng(self.seed)
