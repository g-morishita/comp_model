"""Core contracts and event schema for generic decision modeling."""

from .config_loading import SUPPORTED_CONFIG_SUFFIXES, load_config_mapping
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
    DecisionEventRecord,
    EpisodeTrace,
    EventPhase,
    SimulationEvent,
    decision_records_from_trace,
    decision_records_from_trial_events,
    group_events_by_trial,
    validate_trace,
)
from .requirements import ComponentRequirements

__all__ = [
    "AgentModel",
    "BlockData",
    "DecisionContext",
    "DecisionProblem",
    "ComponentRequirements",
    "SUPPORTED_CONFIG_SUFFIXES",
    "DecisionEventRecord",
    "EpisodeTrace",
    "EventPhase",
    "SimulationEvent",
    "StudyData",
    "SubjectData",
    "TrialDecision",
    "attach_missing_event_traces",
    "get_block_trace",
    "decision_records_from_trace",
    "decision_records_from_trial_events",
    "group_events_by_trial",
    "trace_from_trial_decisions",
    "trial_decisions_from_trace",
    "validate_trace",
    "load_config_mapping",
]
