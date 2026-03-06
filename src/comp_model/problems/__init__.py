"""Problem implementations.

This package contains concrete decision problems under the generic
``DecisionProblem`` protocol and ordered multi-phase social trial programs.
"""

from .social_learning_bandits import (
    DemonstratorThenSubjectActionOnlyProgram,
    DemonstratorThenSubjectActionOnlySelfOutcomeProgram,
    DemonstratorThenSubjectObservedOutcomeProgram,
    DemonstratorThenSubjectObservedOutcomeSelfOutcomeProgram,
    SubjectThenDemonstratorActionOnlyProgram,
    SubjectThenDemonstratorActionOnlySelfOutcomeProgram,
    SubjectThenDemonstratorObservedOutcomeProgram,
    SubjectThenDemonstratorObservedOutcomeSelfOutcomeProgram,
    TwoActorSocialBanditOutcome,
    create_demonstrator_then_subject_action_only_program,
    create_demonstrator_then_subject_action_only_self_outcome_program,
    create_demonstrator_then_subject_observed_outcome_program,
    create_demonstrator_then_subject_observed_outcome_self_outcome_program,
    create_subject_then_demonstrator_action_only_program,
    create_subject_then_demonstrator_action_only_self_outcome_program,
    create_subject_then_demonstrator_observed_outcome_program,
    create_subject_then_demonstrator_observed_outcome_self_outcome_program,
)
from .stationary_bandit import (
    BanditOutcome,
    StationaryBanditProblem,
    create_stationary_bandit_problem,
)

__all__ = [
    "BanditOutcome",
    "DemonstratorThenSubjectActionOnlyProgram",
    "DemonstratorThenSubjectActionOnlySelfOutcomeProgram",
    "DemonstratorThenSubjectObservedOutcomeProgram",
    "DemonstratorThenSubjectObservedOutcomeSelfOutcomeProgram",
    "StationaryBanditProblem",
    "SubjectThenDemonstratorActionOnlyProgram",
    "SubjectThenDemonstratorActionOnlySelfOutcomeProgram",
    "SubjectThenDemonstratorObservedOutcomeProgram",
    "SubjectThenDemonstratorObservedOutcomeSelfOutcomeProgram",
    "TwoActorSocialBanditOutcome",
    "create_demonstrator_then_subject_action_only_program",
    "create_demonstrator_then_subject_action_only_self_outcome_program",
    "create_demonstrator_then_subject_observed_outcome_program",
    "create_demonstrator_then_subject_observed_outcome_self_outcome_program",
    "create_stationary_bandit_problem",
    "create_subject_then_demonstrator_action_only_program",
    "create_subject_then_demonstrator_action_only_self_outcome_program",
    "create_subject_then_demonstrator_observed_outcome_program",
    "create_subject_then_demonstrator_observed_outcome_self_outcome_program",
]
