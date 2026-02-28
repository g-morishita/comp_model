"""Problem implementations.

This package contains concrete decision problems under the generic
``DecisionProblem`` protocol.
"""

from .stationary_bandit import BanditOutcome, StationaryBanditProblem

__all__ = ["BanditOutcome", "StationaryBanditProblem"]
