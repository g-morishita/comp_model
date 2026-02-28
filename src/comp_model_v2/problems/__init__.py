"""Problem implementations.

This package contains concrete decision problems under the generic
``DecisionProblem`` protocol.
"""

from .stationary_bandit import (
    BanditOutcome,
    StationaryBanditProblem,
    create_stationary_bandit_problem,
)

__all__ = ["BanditOutcome", "StationaryBanditProblem", "create_stationary_bandit_problem"]
