"""Analysis utilities for model comparison and diagnostics."""

from .information_criteria import WAICResult, aic, bic, waic
from .profile_likelihood import (
    ProfileLikelihood1DResult,
    ProfileLikelihood2DResult,
    ProfilePoint1D,
    ProfilePoint2D,
    profile_likelihood_1d,
    profile_likelihood_2d,
)

__all__ = [
    "ProfileLikelihood1DResult",
    "ProfileLikelihood2DResult",
    "ProfilePoint1D",
    "ProfilePoint2D",
    "WAICResult",
    "aic",
    "bic",
    "profile_likelihood_1d",
    "profile_likelihood_2d",
    "waic",
]
