"""Canonical event schema for simulation and replay.

The event schema is intentionally explicit. Every trial is represented by an
ordered event stream. Trials may interleave observations, decisions, outcomes,
and updates in program-defined order as long as node-level dependencies remain
valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class EventPhase(str, Enum):
    """Runtime event phases used in canonical traces.

    Attributes
    ----------
    OBSERVATION
        Program emitted an observation and available actions for a node.
    DECISION
        Model produced a policy and one action was sampled for a node.
    OUTCOME
        Program transition executed and emitted an outcome for a node.
    UPDATE
        One learner update callback completed for a node.
    """

    OBSERVATION = "observation"
    DECISION = "decision"
    OUTCOME = "outcome"
    UPDATE = "update"


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    """One event emitted by the runtime.

    Parameters
    ----------
    trial_index : int
        Zero-based trial index.
    phase : EventPhase
        Event phase within the trial.
    payload : Mapping[str, Any]
        Structured event data.
    """

    trial_index: int
    phase: EventPhase
    payload: Mapping[str, Any]


@dataclass(slots=True)
class EpisodeTrace:
    """Container for an episode's event stream.

    Parameters
    ----------
    events : list[SimulationEvent]
        Ordered events emitted by the runtime.
    """

    events: list[SimulationEvent]

    def by_trial(self, trial_index: int) -> list[SimulationEvent]:
        """Return all events for one trial in emission order."""

        return [event for event in self.events if event.trial_index == trial_index]


@dataclass(frozen=True, slots=True)
class DecisionEventRecord:
    """Normalized per-decision record parsed from an event trace.

    Parameters
    ----------
    trial_index : int
        Zero-based trial index.
    node_id : str
        Stable decision-node identifier.
    actor_id : str
        Actor that produced the decision.
    decision_index : int
        Zero-based decision-step index within the trial.
    observation_event : SimulationEvent
        Observation event for the node.
    decision_event : SimulationEvent
        Decision event for the node.
    outcome_event : SimulationEvent | None
        Outcome event for the node when present.
    update_events : tuple[SimulationEvent, ...]
        All update events emitted for the node in chronological order.
    """

    trial_index: int
    node_id: str
    actor_id: str
    decision_index: int
    observation_event: SimulationEvent
    decision_event: SimulationEvent
    outcome_event: SimulationEvent | None
    update_events: tuple[SimulationEvent, ...]


@dataclass(slots=True)
class _DecisionRecordBuilder:
    """Mutable builder used while parsing one trial's event stream."""

    node_id: str
    observation_event: SimulationEvent | None = None
    decision_event: SimulationEvent | None = None
    outcome_event: SimulationEvent | None = None
    update_events: list[SimulationEvent] = field(default_factory=list)
    actor_id: str | None = None
    decision_index: int | None = None


def group_events_by_trial(trace: EpisodeTrace) -> dict[int, list[SimulationEvent]]:
    """Group events by trial index while preserving event order."""

    grouped: dict[int, list[SimulationEvent]] = {}
    for event in trace.events:
        grouped.setdefault(event.trial_index, []).append(event)
    return grouped


def validate_trace(trace: EpisodeTrace) -> None:
    """Validate that an episode trace follows canonical node dependencies.

    Raises
    ------
    ValueError
        If trial indices are not monotonic/contiguous or a trial contains an
        invalid ordered event sequence.
    """

    last_trial = -1
    for event in trace.events:
        if event.trial_index < last_trial:
            raise ValueError("trace trial indices must be non-decreasing")
        last_trial = event.trial_index

    grouped = group_events_by_trial(trace)
    if not grouped:
        return

    actual_indices = sorted(grouped)
    expected_indices = list(range(len(actual_indices)))
    if actual_indices != expected_indices:
        raise ValueError(
            "trace trial indices must be contiguous starting at 0; "
            f"got {actual_indices!r}"
        )

    for trial_index in expected_indices:
        decision_records_from_trial_events(
            grouped[trial_index],
            trial_index=trial_index,
        )


def decision_records_from_trace(trace: EpisodeTrace) -> tuple[DecisionEventRecord, ...]:
    """Parse a validated trace into normalized per-decision records."""

    validate_trace(trace)
    grouped = group_events_by_trial(trace)
    records: list[DecisionEventRecord] = []
    for trial_index in sorted(grouped):
        records.extend(
            decision_records_from_trial_events(
                grouped[trial_index],
                trial_index=trial_index,
            )
        )
    return tuple(records)


def decision_records_from_trial_events(
    trial_events: list[SimulationEvent],
    *,
    trial_index: int | None = None,
) -> tuple[DecisionEventRecord, ...]:
    """Parse one trial's event stream into normalized per-decision records.

    Parameters
    ----------
    trial_events : list[SimulationEvent]
        Events for one trial in chronological order.
    trial_index : int | None, optional
        Optional trial index used to improve error messages.

    Returns
    -------
    tuple[DecisionEventRecord, ...]
        Decision records in decision order.

    Raises
    ------
    ValueError
        If trial events violate canonical node-level dependency rules.
    """

    builders: dict[str, _DecisionRecordBuilder] = {}
    decision_order: list[str] = []

    for event in trial_events:
        payload = _payload_mapping(event.payload, trial_index=trial_index)
        node_id = _coerce_non_empty_str(payload.get("node_id"), field_name="payload.node_id", trial_index=trial_index)
        builder = builders.setdefault(node_id, _DecisionRecordBuilder(node_id=node_id))

        if event.phase is EventPhase.OBSERVATION:
            if builder.observation_event is not None:
                raise ValueError(_trial_prefix(trial_index) + f"duplicate observation for node {node_id!r}")

            actor_id = _coerce_non_empty_str(payload.get("actor_id"), field_name="payload.actor_id", trial_index=trial_index)
            _coerce_available_actions(payload, trial_index=trial_index)
            builder.observation_event = event
            builder.actor_id = actor_id
            if "decision_index" in payload:
                builder.decision_index = _coerce_non_negative_int(
                    payload["decision_index"],
                    field_name="payload.decision_index",
                    trial_index=trial_index,
                )
            continue

        if builder.observation_event is None:
            raise ValueError(
                _trial_prefix(trial_index)
                + f"{event.phase.value!r} event for node {node_id!r} requires a prior observation"
            )

        actor_id = _coerce_non_empty_str(
            payload.get("actor_id", builder.actor_id),
            field_name="payload.actor_id",
            trial_index=trial_index,
        )
        if builder.actor_id is not None and actor_id != builder.actor_id:
            raise ValueError(
                _trial_prefix(trial_index)
                + f"actor_id mismatch for node {node_id!r}: "
                f"{actor_id!r} != {builder.actor_id!r}"
            )

        if "decision_index" in payload:
            decision_index = _coerce_non_negative_int(
                payload["decision_index"],
                field_name="payload.decision_index",
                trial_index=trial_index,
            )
            if builder.decision_index is not None and decision_index != builder.decision_index:
                raise ValueError(
                    _trial_prefix(trial_index)
                    + f"decision_index mismatch for node {node_id!r}: "
                    f"{decision_index} != {builder.decision_index}"
                )
            builder.decision_index = decision_index

        if event.phase is EventPhase.DECISION:
            if builder.decision_event is not None:
                raise ValueError(_trial_prefix(trial_index) + f"duplicate decision for node {node_id!r}")
            if "action" not in payload:
                raise ValueError(_trial_prefix(trial_index) + f"decision payload for node {node_id!r} is missing 'action'")
            builder.decision_event = event
            builder.actor_id = actor_id
            decision_order.append(node_id)
            if builder.decision_index is None:
                builder.decision_index = len(decision_order) - 1
            continue

        if builder.decision_event is None:
            raise ValueError(
                _trial_prefix(trial_index)
                + f"{event.phase.value!r} event for node {node_id!r} requires a prior decision"
            )

        if event.phase is EventPhase.OUTCOME:
            if builder.outcome_event is not None:
                raise ValueError(_trial_prefix(trial_index) + f"duplicate outcome for node {node_id!r}")
            builder.outcome_event = event
            continue

        if event.phase is EventPhase.UPDATE:
            builder.update_events.append(event)
            continue

        raise ValueError(_trial_prefix(trial_index) + f"unsupported event phase {event.phase!r}")

    for node_id, builder in builders.items():
        if builder.observation_event is None:
            raise ValueError(_trial_prefix(trial_index) + f"node {node_id!r} is missing an observation event")
        if builder.decision_event is None:
            raise ValueError(_trial_prefix(trial_index) + f"node {node_id!r} is missing a decision event")

    records: list[DecisionEventRecord] = []
    for expected_index, node_id in enumerate(decision_order):
        builder = builders[node_id]
        assert builder.observation_event is not None
        assert builder.decision_event is not None
        decision_index = builder.decision_index if builder.decision_index is not None else expected_index
        if decision_index != expected_index:
            raise ValueError(
                _trial_prefix(trial_index)
                + f"decision_index sequence must be contiguous starting at 0; "
                f"node {node_id!r} has {decision_index}, expected {expected_index}"
            )
        records.append(
            DecisionEventRecord(
                trial_index=trial_index if trial_index is not None else builder.observation_event.trial_index,
                node_id=node_id,
                actor_id=builder.actor_id if builder.actor_id is not None else "subject",
                decision_index=decision_index,
                observation_event=builder.observation_event,
                decision_event=builder.decision_event,
                outcome_event=builder.outcome_event,
                update_events=tuple(builder.update_events),
            )
        )

    return tuple(records)


def _payload_mapping(payload: Mapping[str, Any] | Any, *, trial_index: int | None) -> Mapping[str, Any]:
    """Validate payload is mapping-like and return it."""

    if not isinstance(payload, Mapping):
        prefix = _trial_prefix(trial_index)
        raise ValueError(f"{prefix}payload must be a mapping")
    return payload


def _coerce_available_actions(payload: Mapping[str, Any], *, trial_index: int | None) -> tuple[Any, ...]:
    """Read and validate available actions from observation payload."""

    if "available_actions" not in payload:
        raise ValueError(_trial_prefix(trial_index) + "observation payload is missing 'available_actions'")
    raw_actions = tuple(payload["available_actions"])
    if len(raw_actions) == 0:
        raise ValueError(_trial_prefix(trial_index) + "available_actions must contain at least one action")
    return raw_actions


def _coerce_non_empty_str(value: Any, *, field_name: str, trial_index: int | None) -> str:
    """Read a non-empty string field or raise a clear error."""

    text = str(value).strip() if value is not None else ""
    if text == "":
        raise ValueError(_trial_prefix(trial_index) + f"{field_name} must be a non-empty string")
    return text


def _coerce_non_negative_int(value: Any, *, field_name: str, trial_index: int | None) -> int:
    """Read a non-negative integer field or raise a clear error."""

    result = int(value)
    if result < 0:
        raise ValueError(_trial_prefix(trial_index) + f"{field_name} must be >= 0")
    return result


def _trial_prefix(trial_index: int | None) -> str:
    """Return standard trial prefix used in validation errors."""

    return f"trial {trial_index}: " if trial_index is not None else ""


__all__ = [
    "DecisionEventRecord",
    "EpisodeTrace",
    "EventPhase",
    "SimulationEvent",
    "decision_records_from_trace",
    "decision_records_from_trial_events",
    "group_events_by_trial",
    "validate_trace",
]
