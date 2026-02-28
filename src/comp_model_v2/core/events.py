"""Canonical event schema for simulation and replay.

The event schema is intentionally small and explicit. Every trial emits a
fixed phase sequence so generated traces are easy to validate and replay.
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
