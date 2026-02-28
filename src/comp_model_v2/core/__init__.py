"""Core contracts and event schema for generic decision modeling."""

from .contracts import AgentModel, DecisionContext, DecisionProblem
from .data import (
    BlockData,
    StudyData,
    SubjectData,
    TrialDecision,
    attach_missing_event_traces,
    get_block_trace,
    trace_from_trial_decisions,
    trial_decisions_from_trace,
)
from .events import (
    DEFAULT_TRIAL_PHASE_SEQUENCE,
    EpisodeTrace,
    EventPhase,
    SimulationEvent,
    group_events_by_trial,
    split_trial_events_into_phase_blocks,
    validate_trace,
)
from .requirements import ComponentRequirements

__all__ = [
    "AgentModel",
    "BlockData",
    "DecisionContext",
    "DecisionProblem",
    "ComponentRequirements",
    "DEFAULT_TRIAL_PHASE_SEQUENCE",
    "EpisodeTrace",
    "EventPhase",
    "SimulationEvent",
    "StudyData",
    "SubjectData",
    "TrialDecision",
    "attach_missing_event_traces",
    "get_block_trace",
    "group_events_by_trial",
    "split_trial_events_into_phase_blocks",
    "trace_from_trial_decisions",
    "trial_decisions_from_trace",
    "validate_trace",
]
