"""Canonical event schema for simulation and replay.

The event schema is intentionally explicit. Every trial is represented by an
ordered phase sequence so generated traces can be validated and replayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class EventPhase(str, Enum):
    """Ordered phases in one decision trial.

    Attributes
    ----------
    OBSERVATION
        Problem emitted an observation for the model.
    DECISION
        Model produced a policy and one action was sampled.
    OUTCOME
        Problem transition executed and emitted an outcome.
    UPDATE
        Model update callback completed.
    """

    OBSERVATION = "observation"
    DECISION = "decision"
    OUTCOME = "outcome"
    UPDATE = "update"


DEFAULT_TRIAL_PHASE_SEQUENCE: tuple[EventPhase, ...] = (
    EventPhase.OBSERVATION,
    EventPhase.DECISION,
    EventPhase.OUTCOME,
    EventPhase.UPDATE,
)


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

    Notes
    -----
    The runtime appends events in chronological order. Consumers can query
    per-trial slices with :meth:`by_trial`.
    """

    events: list[SimulationEvent]

    def by_trial(self, trial_index: int) -> list[SimulationEvent]:
        """Return all events for one trial.

        Parameters
        ----------
        trial_index : int
            Trial index to filter.

        Returns
        -------
        list[SimulationEvent]
            Events in emission order for the requested trial.
        """

        return [event for event in self.events if event.trial_index == trial_index]


def group_events_by_trial(trace: EpisodeTrace) -> dict[int, list[SimulationEvent]]:
    """Group events by trial index while preserving event order.

    Parameters
    ----------
    trace : EpisodeTrace
        Episode trace to group.

    Returns
    -------
    dict[int, list[SimulationEvent]]
        Mapping from trial index to event list.
    """

    grouped: dict[int, list[SimulationEvent]] = {}
    for event in trace.events:
        grouped.setdefault(event.trial_index, []).append(event)
    return grouped


def validate_trace(
    trace: EpisodeTrace,
    *,
    expected_phase_sequence: tuple[EventPhase, ...] = DEFAULT_TRIAL_PHASE_SEQUENCE,
) -> None:
    """Validate that an episode trace follows canonical trial semantics.

    Parameters
    ----------
    trace : EpisodeTrace
        Trace to validate.
    expected_phase_sequence : tuple[EventPhase, ...], optional
        Canonical phase block sequence. Trials may repeat this block multiple
        times (for multi-decision programs), but each block must match exactly.

    Raises
    ------
    ValueError
        If trial indices are not monotonic/contiguous, or phase order does not
        follow one-or-more repetitions of ``expected_phase_sequence``.
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
        split_trial_events_into_phase_blocks(
            grouped[trial_index],
            expected_phase_sequence=expected_phase_sequence,
            trial_index=trial_index,
        )


def split_trial_events_into_phase_blocks(
    trial_events: list[SimulationEvent],
    *,
    expected_phase_sequence: tuple[EventPhase, ...] = DEFAULT_TRIAL_PHASE_SEQUENCE,
    trial_index: int | None = None,
) -> tuple[tuple[SimulationEvent, ...], ...]:
    """Split a trial's events into canonical phase blocks.

    Parameters
    ----------
    trial_events : list[SimulationEvent]
        Events for one trial in chronological order.
    expected_phase_sequence : tuple[EventPhase, ...], optional
        Phase order expected in each block.
    trial_index : int | None, optional
        Optional trial index used to improve error messages.

    Returns
    -------
    tuple[tuple[SimulationEvent, ...], ...]
        Sequence of fixed-size phase blocks.

    Raises
    ------
    ValueError
        If event count is incompatible with block size or block phases mismatch.
    """

    block_size = len(expected_phase_sequence)
    if block_size == 0:
        raise ValueError("expected_phase_sequence must contain at least one phase")

    if len(trial_events) == 0 or len(trial_events) % block_size != 0:
        prefix = f"trial {trial_index}: " if trial_index is not None else ""
        raise ValueError(
            f"{prefix}event count {len(trial_events)} does not match "
            f"phase block size {block_size}"
        )

    blocks: list[tuple[SimulationEvent, ...]] = []
    for offset in range(0, len(trial_events), block_size):
        block = tuple(trial_events[offset : offset + block_size])
        phases = tuple(event.phase for event in block)
        if phases != expected_phase_sequence:
            prefix = f"trial {trial_index}: " if trial_index is not None else ""
            raise ValueError(
                f"{prefix}invalid phase block {phases!r}, "
                f"expected {expected_phase_sequence!r}"
            )
        blocks.append(block)

    return tuple(blocks)
