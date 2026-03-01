"""Analysis utilities for model comparison and diagnostics."""

from .information_criteria import PSISLOOResult, WAICResult, aic, bic, psis_loo, waic
from .profile_likelihood import (
    ProfileLikelihood1DResult,
    ProfileLikelihood2DResult,
    ProfilePoint1D,
    ProfilePoint2D,
    profile_likelihood_1d,
    profile_likelihood_2d,
)

__all__ = [
    "PSISLOOResult",
    "ProfileLikelihood1DResult",
    "ProfileLikelihood2DResult",
    "ProfilePoint1D",
    "ProfilePoint2D",
    "WAICResult",
    "aic",
    "bic",
    "psis_loo",
    "profile_likelihood_1d",
    "profile_likelihood_2d",
    "waic",
]
