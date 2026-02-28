"""Parameter transform primitives for optimizer-based inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ParameterTransform:
    """Bidirectional parameter transform.

    Parameters
    ----------
    forward : Callable[[float], float]
        Maps unconstrained optimizer-space value ``z`` to constrained parameter
        value ``theta``.
    inverse : Callable[[float], float]
        Maps constrained parameter value ``theta`` back to optimizer-space
        value ``z``.
    """

    forward: Callable[[float], float]
    inverse: Callable[[float], float]


def identity_transform() -> ParameterTransform:
    """Return identity transform for unconstrained parameters."""

    return ParameterTransform(forward=lambda value: float(value), inverse=lambda value: float(value))


def unit_interval_logit_transform(*, eps: float = 1e-9) -> ParameterTransform:
    """Return logit/sigmoid transform for parameters constrained to ``(0, 1)``.

    Parameters
    ----------
    eps : float, optional
        Numerical clamp used in inverse transform to avoid infinities.

    Returns
    -------
    ParameterTransform
        Transform mapping unconstrained values to ``(0, 1)`` via sigmoid.
    """

    if eps <= 0.0 or eps >= 0.5:
        raise ValueError("eps must be in (0, 0.5)")

    def forward(z_value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-float(z_value))))

    def inverse(theta_value: float) -> float:
        theta = float(np.clip(float(theta_value), eps, 1.0 - eps))
        return float(np.log(theta / (1.0 - theta)))

    return ParameterTransform(forward=forward, inverse=inverse)


def positive_log_transform(*, eps: float = 1e-12) -> ParameterTransform:
    """Return exponential/log transform for positive parameters.

    Parameters
    ----------
    eps : float, optional
        Lower clamp for inverse transform.

    Returns
    -------
    ParameterTransform
        Transform mapping unconstrained values to ``(0, +inf)``.
    """

    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    def forward(z_value: float) -> float:
        return float(np.exp(float(z_value)))

    def inverse(theta_value: float) -> float:
        return float(np.log(max(float(theta_value), eps)))

    return ParameterTransform(forward=forward, inverse=inverse)


__all__ = [
    "ParameterTransform",
    "identity_transform",
    "positive_log_transform",
    "unit_interval_logit_transform",
]
