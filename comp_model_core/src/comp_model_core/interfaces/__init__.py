"""
Abstract interfaces (ABCs) for tasks, models, generators, and estimators.

The :mod:`comp_model_core.interfaces` subpackage defines the minimal contracts that
implementations must satisfy in downstream packages (e.g., ``comp_model_impl``).

Notes
-----
These interfaces are designed to be small, explicit, and easy to mock for testing.
They intentionally avoid opinionated training frameworks or heavy dependencies.
"""

from .bandit import BanditEnv, SocialBanditEnv, EnvStep, SocialObservation
from .model import ComputationalModel, SocialComputationalModel
from .generator import Generator
from .estimator import Estimator, FitResult
from .demonstrator import Demonstrator

__all__ = [
    "BanditEnv",
    "SocialBanditEnv",
    "EnvStep",
    "SocialObservation",
    "ComputationalModel",
    "SocialComputationalModel",
    "Generator",
    "Estimator",
    "FitResult",
    "Demonstrator",
]
