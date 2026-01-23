"""
Box bounds and bounded parameter spaces.

This module provides:

- :class:`Bound`: a simple lower/upper bound container.
- :class:`ParameterBoundsSpace`: a named vector space with per-parameter bounds.

These are typically used with deterministic optimizers (e.g., SciPy) that support
box constraints directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np


@dataclass(frozen=True, slots=True)
class Bound:
    """
    Closed interval bound ``[lo, hi]``.

    Parameters
    ----------
    lo : float
        Lower bound.
    hi : float
        Upper bound.

    Attributes
    ----------
    lo : float
    hi : float
    """

    lo: float
    hi: float

    def clip(self, x: float) -> float:
        """
        Clip a scalar into the bound interval.

        Parameters
        ----------
        x : float
            Value to clip.

        Returns
        -------
        float
            ``x`` clipped to ``[lo, hi]``.
        """
        if x < self.lo:
            return self.lo
        if x > self.hi:
            return self.hi
        return x


@dataclass(frozen=True, slots=True)
class ParameterBoundsSpace:
    """
    Direct parameterization with box constraints.

    Parameters
    ----------
    names : Sequence[str]
        Parameter names in the optimization vector order.
    bounds : Mapping[str, Bound]
        Mapping from parameter name to its bound.

    Attributes
    ----------
    names : Sequence[str]
    bounds : Mapping[str, Bound]
    """

    names: Sequence[str]
    bounds: Mapping[str, Bound]

    @property
    def dim(self) -> int:
        """
        Dimensionality of the optimization vector.

        Returns
        -------
        int
            Number of parameters.
        """
        return len(self.names)

    def clip_vec(self, x: np.ndarray) -> np.ndarray:
        """
        Clip a vector into the parameter bounds.

        Parameters
        ----------
        x : numpy.ndarray
            Vector of shape ``(dim,)``.

        Returns
        -------
        numpy.ndarray
            Clipped vector of shape ``(dim,)``.

        Raises
        ------
        ValueError
            If the shape of ``x`` is not ``(dim,)``.
        """
        if x.shape != (self.dim,):
            raise ValueError(f"Expected x.shape == ({self.dim},), got {x.shape}.")
        y = x.copy()
        for i, name in enumerate(self.names):
            y[i] = self.bounds[name].clip(float(y[i]))
        return y

    def to_params(self, x: np.ndarray) -> dict[str, float]:
        """
        Convert a bounded vector to a parameter dict.

        Parameters
        ----------
        x : numpy.ndarray
            Vector of shape ``(dim,)``.

        Returns
        -------
        dict[str, float]
            Mapping from parameter name to value.
        """
        y = self.clip_vec(x)
        return {name: float(y[i]) for i, name in enumerate(self.names)}

    def sample_init(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample a random initialization uniformly within bounds.

        Parameters
        ----------
        rng : numpy.random.Generator
            RNG used for sampling.

        Returns
        -------
        numpy.ndarray
            Random vector of shape ``(dim,)`` with each component sampled from
            ``Uniform(lo, hi)``.
        """
        x = np.empty((self.dim,), dtype=float)
        for i, name in enumerate(self.names):
            b = self.bounds[name]
            x[i] = float(rng.uniform(b.lo, b.hi))
        return x
