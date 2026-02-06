"""Analysis utilities for computational models."""

from .profile_likelihood import profile_likelihood_1d, profile_likelihood_2d
from .parameter_recovery import plot_parameter_recovery

__all__ = [
    "profile_likelihood_1d",
    "profile_likelihood_2d",
    "plot_parameter_recovery",
]
