"""Study-level data containers and conversion helpers.

This module provides generic data structures analogous to prior internal
``Trial``/``Block``/``SubjectData``/``StudyData`` layers while keeping
:class:`comp_model.core.events.EpisodeTrace` as the canonical runtime format.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .events import (
    EpisodeTrace,
    EventPhase,
    SimulationEvent,
    decision_records_from_trace,
    validate_trace,
)


@dataclass(frozen=True, slots=True)
class TrialDecision:
    """One decision record inside a trial.

    Parameters
    ----------
    trial_index : int
        Zero-based trial index.
    decision_index : int, optional
        Zero-based decision-node index within the trial.
    actor_id : str, optional
        Actor that produced the decision.
    learner_ids : tuple[str, ...] | None, optional
        Actors that receive update callbacks for this decision in chronological
        order. ``None`` implies one update for ``actor_id``. ``()`` means this
        decision emits no update events.
    node_id : str | None, optional
        Optional decision-node identifier.
    available_actions : tuple[Any, ...] | None, optional
        Legal actions at decision time. If ``None``, conversion requires
        ``action`` and infers a singleton action set.
    action : Any | None, optional
        Observed action.
    observation : Any, optional
        Observation payload.
    outcome : Any, optional
        Outcome payload. If missing and ``reward`` is provided, conversion uses
        ``{"reward": reward}``.
    reward : float | None, optional
        Convenience scalar reward field for tabular datasets.
    metadata : Mapping[str, Any], optional
        Free-form row metadata.

    Raises
    ------
    ValueError
        If indices are negative or action is incompatible with available actions.
    """

    trial_index: int
    decision_index: int = 0
    actor_id: str = "subject"
    learner_ids: tuple[str, ...] | None = None
    node_id: str | None = None
    available_actions: tuple[Any, ...] | None = None
    action: Any | None = None
    observation: Any = None
    outcome: Any = None
    reward: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trial_index < 0:
            raise ValueError("trial_index must be >= 0")
        if self.decision_index < 0:
            raise ValueError("decision_index must be >= 0")
        if self.available_actions is not None and len(self.available_actions) == 0:
            raise ValueError("available_actions must not be empty when provided")
        if self.available_actions is not None and self.action is not None and self.action not in self.available_actions:
            raise ValueError("action must be one of available_actions")
        if self.learner_ids is not None:
            for learner_id in self.learner_ids:
                if not str(learner_id).strip():
                    raise ValueError("learner_ids must contain only non-empty strings")


@dataclass(frozen=True, slots=True)
class BlockData:
    """One experimental block for a subject.

    Parameters
    ----------
    block_id : str | int | None, optional
        Optional block identifier.
    trials : tuple[TrialDecision, ...], optional
        Tabular decisions for the block.
    event_trace : EpisodeTrace | None, optional
        Canonical block trace.
    metadata : Mapping[str, Any], optional
        Free-form metadata.

    Raises
    ------
    ValueError
        If both ``trials`` and ``event_trace`` are missing, or trial ordering is
        invalid.
    """

    block_id: str | int | None = None
    trials: tuple[TrialDecision, ...] = ()
    event_trace: EpisodeTrace | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.trials) == 0 and self.event_trace is None:
            raise ValueError("block must include trials or event_trace")

        if self.event_trace is not None:
            validate_trace(self.event_trace)

        if self.trials:
            _validate_trial_decision_order(self.trials)

    @property
    def n_trials(self) -> int:
        """Return number of unique trial indices in this block."""

        if self.event_trace is not None:
            trial_indices = {event.trial_index for event in self.event_trace.events}
            return len(trial_indices)

        return len({trial.trial_index for trial in self.trials})

    def with_event_trace(self, trace: EpisodeTrace) -> "BlockData":
        """Return a new block with the provided event trace."""

        validate_trace(trace)
        return replace(self, event_trace=trace)


@dataclass(frozen=True, slots=True)
class SubjectData:
    """All blocks for one subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    blocks : tuple[BlockData, ...]
        Subject blocks.
    metadata : Mapping[str, Any], optional
        Free-form metadata.

    Raises
    ------
    ValueError
        If ``subject_id`` is empty or no blocks are provided.
    """

    subject_id: str
    blocks: tuple[BlockData, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id must be non-empty")
        if len(self.blocks) == 0:
            raise ValueError("subject must include at least one block")


@dataclass(frozen=True, slots=True)
class StudyData:
    """Container for a complete study.

    Parameters
    ----------
    subjects : tuple[SubjectData, ...]
        Subject list.
    metadata : Mapping[str, Any], optional
        Free-form study metadata.

    Raises
    ------
    ValueError
        If no subjects exist or subject IDs are not unique.
    """

    subjects: tuple[SubjectData, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.subjects) == 0:
            raise ValueError("study must include at least one subject")

        ids = [subject.subject_id for subject in self.subjects]
        if len(set(ids)) != len(ids):
            raise ValueError("subject_id values must be unique")

    @property
    def n_subjects(self) -> int:
        """Return number of subjects in the study."""

        return len(self.subjects)


def trace_from_trial_decisions(decisions: Sequence[TrialDecision]) -> EpisodeTrace:
    """Convert tabular trial decisions into a canonical trace.

    Parameters
    ----------
    decisions : Sequence[TrialDecision]
        Input decision rows. Rows must be sorted by ``(trial_index,
        decision_index)`` and cover contiguous trial indices.

    Returns
    -------
    EpisodeTrace
        Canonical event trace with one phase block per decision row.

    Raises
    ------
    ValueError
        If decision rows are empty or inconsistent.
    """

    if len(decisions) == 0:
        raise ValueError("decisions must include at least one row")

    rows = tuple(decisions)
    _validate_trial_decision_order(rows)

    events: list[SimulationEvent] = []
    for row in rows:
        available_actions = _coerce_available_actions(row)
        action = row.action
        if action is None:
            raise ValueError(
                f"decision row trial={row.trial_index} index={row.decision_index} is missing action"
            )

        node_id = row.node_id if row.node_id is not None else f"decision_{row.decision_index}"

        observation = row.observation if row.observation is not None else {"trial_index": row.trial_index}
        outcome = _coerce_outcome(row)
        learner_ids = row.learner_ids if row.learner_ids is not None else (row.actor_id,)

        distribution = _one_hot_distribution(available_actions, action)
        events.extend(
            (
                SimulationEvent(
                    trial_index=row.trial_index,
                    phase=EventPhase.OBSERVATION,
                    payload={
                        "observation": observation,
                        "available_actions": available_actions,
                        "actor_id": row.actor_id,
                        "decision_index": row.decision_index,
                        "node_id": node_id,
                    },
                ),
                SimulationEvent(
                    trial_index=row.trial_index,
                    phase=EventPhase.DECISION,
                    payload={
                        "distribution": distribution,
                        "action": action,
                        "actor_id": row.actor_id,
                        "decision_index": row.decision_index,
                        "node_id": node_id,
                    },
                ),
            )
        )
        if outcome is not None:
            events.append(
                SimulationEvent(
                    trial_index=row.trial_index,
                    phase=EventPhase.OUTCOME,
                    payload={
                        "outcome": outcome,
                        "actor_id": row.actor_id,
                        "decision_index": row.decision_index,
                        "node_id": node_id,
                    },
                )
            )
        for learner_id in learner_ids:
            events.append(
                SimulationEvent(
                    trial_index=row.trial_index,
                    phase=EventPhase.UPDATE,
                    payload={
                        "update_called": True,
                        "action": action,
                        "actor_id": row.actor_id,
                        "learner_id": learner_id,
                        "decision_index": row.decision_index,
                        "node_id": node_id,
                    },
                )
            )

    trace = EpisodeTrace(events=events)
    validate_trace(trace)
    return trace


def trial_decisions_from_trace(trace: EpisodeTrace) -> tuple[TrialDecision, ...]:
    """Convert canonical trace into tabular trial decision rows.

    Parameters
    ----------
    trace : EpisodeTrace
        Canonical event trace.

    Returns
    -------
    tuple[TrialDecision, ...]
        Decision rows in chronological order.
    """

    validate_trace(trace)

    rows: list[TrialDecision] = []
    for record in decision_records_from_trace(trace):
        observation_payload = _payload_mapping(record.observation_event.payload, record.trial_index)
        decision_payload = _payload_mapping(record.decision_event.payload, record.trial_index)
        outcome_payload = (
            _payload_mapping(record.outcome_event.payload, record.trial_index)
            if record.outcome_event is not None
            else {}
        )
        learner_ids = tuple(
            str(_payload_mapping(update_event.payload, record.trial_index).get("learner_id", record.actor_id))
            for update_event in record.update_events
        )
        outcome = outcome_payload.get("outcome")
        reward = _extract_reward(outcome)

        rows.append(
            TrialDecision(
                trial_index=record.trial_index,
                decision_index=record.decision_index,
                actor_id=record.actor_id,
                learner_ids=learner_ids,
                node_id=record.node_id,
                available_actions=tuple(observation_payload["available_actions"]),
                action=decision_payload.get("action"),
                observation=observation_payload.get("observation"),
                outcome=outcome,
                reward=reward,
            )
        )

    _validate_trial_decision_order(tuple(rows))
    return tuple(rows)


def get_block_trace(block: BlockData) -> EpisodeTrace:
    """Return block trace, converting from trial rows if needed.

    Parameters
    ----------
    block : BlockData
        Block input.

    Returns
    -------
    EpisodeTrace
        Canonical event trace for the block.

    Raises
    ------
    ValueError
        If block has no event trace and no trial rows.
    """

    if block.event_trace is not None:
        return block.event_trace

    if not block.trials:
        raise ValueError("block has neither event_trace nor trial decisions")

    return trace_from_trial_decisions(block.trials)


def attach_missing_event_traces(study: StudyData, *, overwrite: bool = False) -> StudyData:
    """Return a study with block traces attached where missing.

    Parameters
    ----------
    study : StudyData
        Input study.
    overwrite : bool, optional
        If ``True``, regenerate traces from trial rows even when traces already
        exist.

    Returns
    -------
    StudyData
        New study with event traces attached on each block.
    """

    subjects_out: list[SubjectData] = []
    for subject in study.subjects:
        blocks_out: list[BlockData] = []
        for block in subject.blocks:
            if block.event_trace is not None and not overwrite:
                blocks_out.append(block)
                continue

            trace = get_block_trace(block)
            blocks_out.append(block.with_event_trace(trace))

        subjects_out.append(
            SubjectData(
                subject_id=subject.subject_id,
                blocks=tuple(blocks_out),
                metadata=dict(subject.metadata),
            )
        )

    return StudyData(subjects=tuple(subjects_out), metadata=dict(study.metadata))


def _validate_trial_decision_order(rows: tuple[TrialDecision, ...]) -> None:
    """Validate trial/decision ordering and contiguity constraints."""

    sorted_rows = tuple(sorted(rows, key=lambda item: (item.trial_index, item.decision_index)))
    if rows != sorted_rows:
        raise ValueError("trial decisions must be sorted by (trial_index, decision_index)")

    trial_indices = sorted({row.trial_index for row in rows})
    if trial_indices != list(range(len(trial_indices))):
        raise ValueError("trial_index values must be contiguous starting at 0")

    current_trial = -1
    seen_decision_indices: set[int] = set()
    for row in rows:
        if row.trial_index != current_trial:
            current_trial = row.trial_index
            seen_decision_indices = set()

        if row.decision_index in seen_decision_indices:
            raise ValueError(
                "duplicate decision_index within trial: "
                f"trial={row.trial_index} decision_index={row.decision_index}"
            )
        seen_decision_indices.add(row.decision_index)


def _coerce_available_actions(row: TrialDecision) -> tuple[Any, ...]:
    """Normalize available actions for conversion."""

    if row.available_actions is not None:
        return tuple(row.available_actions)

    if row.action is None:
        raise ValueError(
            f"decision row trial={row.trial_index} index={row.decision_index} "
            "must include available_actions or action"
        )

    return (row.action,)


def _coerce_outcome(row: TrialDecision) -> Any:
    """Build outcome payload from row fields."""

    if row.outcome is not None:
        return row.outcome

    if row.reward is not None:
        return {"reward": float(row.reward)}

    return None


def _one_hot_distribution(available_actions: tuple[Any, ...], action: Any) -> dict[Any, float]:
    """Build one-hot decision distribution from observed action."""

    if action not in available_actions:
        raise ValueError(f"action {action!r} is not one of available_actions {available_actions!r}")

    distribution = {candidate: 0.0 for candidate in available_actions}
    distribution[action] = 1.0
    return distribution


def _payload_mapping(payload: Mapping[str, Any] | Any, trial_index: int) -> Mapping[str, Any]:
    """Validate payload is mapping-like and return it."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"trial {trial_index} payload must be a mapping")
    return payload


def _extract_reward(outcome: Any) -> float | None:
    """Extract reward when available from outcome payload."""

    if isinstance(outcome, Mapping) and "reward" in outcome:
        return float(outcome["reward"])

    if outcome is not None and hasattr(outcome, "reward"):
        return float(getattr(outcome, "reward"))

    return None
