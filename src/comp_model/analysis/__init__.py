"""Analysis utilities for model comparison and diagnostics."""

from .information_criteria import aic, bic
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
    "aic",
    "bic",
    "profile_likelihood_1d",
    "profile_likelihood_2d",
]
