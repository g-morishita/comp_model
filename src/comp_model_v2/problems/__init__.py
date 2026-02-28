"""Problem implementations.

This package contains concrete decision problems under the generic
``DecisionProblem`` protocol.
"""

from .stationary_bandit import (
    BanditOutcome,
    StationaryBanditProblem,
    create_stationary_bandit_problem,
)
from .social_two_stage_bandit import (
    SocialBanditOutcome,
    TwoStageSocialBanditProgram,
    create_two_stage_social_bandit_program,
)

__all__ = [
    "BanditOutcome",
    "SocialBanditOutcome",
    "StationaryBanditProblem",
    "TwoStageSocialBanditProgram",
    "create_stationary_bandit_problem",
    "create_two_stage_social_bandit_program",
]
