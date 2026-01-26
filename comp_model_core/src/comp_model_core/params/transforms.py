r"""
Transforms between constrained parameters and unconstrained real space.

Optimization and some inference methods (e.g., unconstrained optimizers, HMC)
operate on parameters in :math:`\mathbb{R}^D`. Many psychological model parameters
are naturally constrained (e.g., learning rates in (0, 1)).

This module defines simple, differentiable transforms with forward/inverse maps.

Notes
-----
Transforms are used by :class:`~comp_model_core.params.schema.ParameterSchema` to map
between:

- **Constrained parameters** ``x`` (what your model expects), and
- **Unconstrained variables** ``z`` (what an optimizer or sampler expects).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.special import expit, logit, logsumexp

from ..errors import ParameterValidationError


class Transform(Protocol):
    """
    Protocol for parameter transforms.

    Any transform must define both a forward and inverse mapping.
    """

    def forward(self, z: float) -> float:
        """
        Map an unconstrained real value to a constrained value.

        Parameters
        ----------
        z : float
            Unconstrained value.

        Returns
        -------
        float
            Constrained value.
        """
        ...

    def inverse(self, x: float) -> float:
        """
        Map a constrained value to an unconstrained real value.

        Parameters
        ----------
        x : float
            Constrained value.

        Returns
        -------
        float
            Unconstrained value.
        """
        ...


@dataclass(frozen=True, slots=True)
class Identity:
    """
    Identity transform for unconstrained parameters.
    """

    def forward(self, z: float) -> float:
        """
        Forward map (identity).

        Parameters
        ----------
        z : float
            Unconstrained value.

        Returns
        -------
        float
            Same as ``z``.
        """
        return float(z)

    def inverse(self, x: float) -> float:
        """
        Inverse map (identity).

        Parameters
        ----------
        x : float
            Constrained value.

        Returns
        -------
        float
            Same as ``x``.
        """
        return float(x)


@dataclass(frozen=True, slots=True)
class Sigmoid:
    r"""
    Logistic transform mapping :math:`\mathbb{R} \to (0, 1)`.

    Parameters
    ----------
    eps : float, optional
        Small epsilon used to clip values when computing the inverse.
    """

    eps: float = 1e-12

    def forward(self, z: float) -> float:
        r"""
        Map :math:`z \in \mathbb{R}` to :math:`x \in (0, 1)`.

        Parameters
        ----------
        z : float
            Unconstrained value.

        Returns
        -------
        float
            Value in (0, 1).
        """
        return float(expit(float(z)))

    def inverse(self, x: float) -> float:
        r"""
        Map :math:`x \in (0, 1)` to :math:`z \in \mathbb{R}`.

        Parameters
        ----------
        x : float
            Value in (0, 1).

        Returns
        -------
        float
            Unconstrained value.

        Notes
        -----
        The input is clipped into ``[eps, 1-eps]`` to avoid infinities.
        """
        x = float(x)
        x = min(max(x, self.eps), 1.0 - self.eps)
        return float(logit(x))


@dataclass(frozen=True, slots=True)
class Softplus:
    r"""
    Softplus transform mapping :math:`\mathbb{R} \to (0, \infty)`.

    Parameters
    ----------
    eps : float, optional
        Small epsilon used in the inverse for stability.
    """

    eps: float = 1e-12

    def forward(self, z: float) -> float:
        r"""
        Map :math:`z \in \mathbb{R}` to :math:`x \in (0, \infty)`.

        Parameters
        ----------
        z : float
            Unconstrained value.

        Returns
        -------
        float
            Positive value.
        """
        z = float(z)
        # softplus(z) = log(1 + exp(z)) = logsumexp([0, z])
        return float(logsumexp([0.0, z]))

    def inverse(self, x: float) -> float:
        """
        Inverse of softplus.

        Parameters
        ----------
        x : float
            Positive value.

        Returns
        -------
        float
            Unconstrained value.

        Raises
        ------
        ParameterValidationError
            If ``x <= 0``.

        Notes
        -----
        Uses a stable computation:

        - For small ``x``: ``log(expm1(x))``
        - For large ``x``: approximately ``x``.
        """
        x = float(x)
        if x <= 0:
            raise ParameterValidationError(f"Softplus inverse requires x>0, got {x}.")
        if x > 30:
            return float(x)
        return float(np.log(np.expm1(max(x, self.eps))))


@dataclass(frozen=True, slots=True)
class BoundedTanh:
    r"""
    Smooth transform mapping :math:`\mathbb{R} \to (lo, hi)` using ``tanh``.

    Parameters
    ----------
    lo : float
        Lower bound.
    hi : float
        Upper bound.
    eps : float, optional
        Epsilon used for clipping in the inverse.
    """

    lo: float
    hi: float
    eps: float = 1e-12

    def forward(self, z: float) -> float:
        r"""
        Map :math:`z \in \mathbb{R}` to :math:`x \in (lo, hi)`.

        Parameters
        ----------
        z : float
            Unconstrained value.

        Returns
        -------
        float
            Value in (lo, hi).
        """
        z = float(z)
        t = float(np.tanh(z))  # (-1,1)
        return float(self.lo + (self.hi - self.lo) * (t + 1.0) * 0.5)

    def inverse(self, x: float) -> float:
        r"""
        Map :math:`x \in (lo, hi)` to :math:`z \in \mathbb{R}`.

        Parameters
        ----------
        x : float
            Value in (lo, hi).

        Returns
        -------
        float
            Unconstrained value.

        Notes
        -----
        The input is clipped into ``(lo+eps, hi-eps)`` to keep ``arctanh`` finite.
        """
        x = float(x)
        # Clip into (lo, hi) for invertibility.
        x = min(max(x, self.lo + self.eps), self.hi - self.eps)
        y = (2.0 * (x - self.lo) / (self.hi - self.lo)) - 1.0
        y = min(max(y, -1.0 + self.eps), 1.0 - self.eps)
        return float(np.arctanh(y))
