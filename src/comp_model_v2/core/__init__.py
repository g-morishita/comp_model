"""Core contracts and event schema for generic decision modeling."""

from .contracts import AgentModel, DecisionContext, DecisionProblem
from .events import (
    DEFAULT_TRIAL_PHASE_SEQUENCE,
    EpisodeTrace,
    EventPhase,
    SimulationEvent,
    group_events_by_trial,
    validate_trace,
)
from .requirements import ComponentRequirements

__all__ = [
    "AgentModel",
    "DecisionContext",
    "DecisionProblem",
    "ComponentRequirements",
    "DEFAULT_TRIAL_PHASE_SEQUENCE",
    "EpisodeTrace",
    "EventPhase",
    "SimulationEvent",
    "group_events_by_trial",
    "validate_trace",
]
