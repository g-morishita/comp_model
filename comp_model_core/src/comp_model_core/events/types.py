"""
Event log schema.

This module defines the event types and data structures for representing an
event-log "experience trace" that can be replayed by different estimators
(e.g., MLE vs Stan) without re-implementing timing logic.

The key idea is that **the ordering of events is the source of truth** for when
social observation, choice, and outcome updates occur.

Notes
-----
Generators typically store serialized event logs inside :attr:`comp_model_core.data.types.Block.metadata`
under the key :data:`EVENT_LOG_KEY`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence


#: Key under which generators store the block event log in ``Block.metadata``.
EVENT_LOG_KEY = "event_log"


class EventType(IntEnum):
    """
    Discrete event types for an event stream.

    The integer codes are part of the event-log contract. If you change them, you
    must update any downstream code that relies on the codes (e.g., Stan programs).

    Attributes
    ----------
    BLOCK_START : int
        Marks the beginning of a block. Estimators should reset latent state here.
    SOCIAL_OBSERVED : int
        A social observation event (e.g., demonstrator choice/outcome).
    CHOICE : int
        A subject choice event.
    OUTCOME : int
        An outcome event for a subject choice.
    """

    BLOCK_START = 1
    SOCIAL_OBSERVED = 2
    CHOICE = 3
    OUTCOME = 4


@dataclass(frozen=True, slots=True)
class Event:
    """
    A single event in an event stream.

    Parameters
    ----------
    idx : int
        Zero-based index within the (block-level) event stream.
    type : EventType
        Event type code.
    t : int or None
        Trial index (typically 0-based) within the block, if applicable.
    state : Any or None
        State/context identifier used for this event (often an ``int``).
    payload : Mapping[str, Any]
        Event-specific fields in a JSON-friendly mapping.

    Notes
    -----
    The event log is designed to be serializable. ``payload`` should contain only
    JSON-compatible values (numbers, strings, lists, dicts, ``None``).
    """

    idx: int
    type: EventType
    t: int | None
    state: Any | None
    payload: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        """
        Serialize the event to a JSON-friendly dict.

        Returns
        -------
        dict[str, Any]
            JSON-friendly representation of this event.
        """
        return {
            "idx": int(self.idx),
            "type": int(self.type),
            "t": None if self.t is None else int(self.t),
            "state": self.state,
            "payload": dict(self.payload),
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "Event":
        """
        Deserialize an event from a mapping.

        Parameters
        ----------
        d : Mapping[str, Any]
            Mapping produced by :meth:`to_json`.

        Returns
        -------
        Event
            Parsed event.

        Raises
        ------
        KeyError
            If required keys are missing.
        ValueError
            If fields cannot be coerced to expected types.
        """
        return Event(
            idx=int(d["idx"]),
            type=EventType(int(d["type"])),
            t=None if d.get("t") is None else int(d["t"]),
            state=d.get("state"),
            payload=dict(d.get("payload", {})),
        )


@dataclass(frozen=True, slots=True)
class EventLog:
    """
    Block-level event stream.

    Parameters
    ----------
    events : Sequence[Event]
        Ordered list of events for a block.
    schema_version : int, optional
        Version of the event-log schema. Increment when a backward-incompatible
        change is made.
    metadata : Mapping[str, Any] or None, optional
        Additional information about the log (e.g., generator settings).

    Attributes
    ----------
    events : Sequence[Event]
    schema_version : int
    metadata : Mapping[str, Any] or None
    """

    events: Sequence[Event]
    schema_version: int = 1
    metadata: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        """
        Serialize the event log to a JSON-friendly dict.

        Returns
        -------
        dict[str, Any]
            JSON-friendly representation.
        """
        return {
            "schema_version": int(self.schema_version),
            "metadata": dict(self.metadata or {}),
            "events": [e.to_json() for e in self.events],
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "EventLog":
        """
        Deserialize an event log from a mapping.

        Parameters
        ----------
        d : Mapping[str, Any]
            Mapping produced by :meth:`to_json`.

        Returns
        -------
        EventLog
            Parsed event log.
        """
        return EventLog(
            schema_version=int(d.get("schema_version", 1)),
            metadata=dict(d.get("metadata", {})),
            events=[Event.from_json(x) for x in d.get("events", [])],
        )


def validate_event_log(log: EventLog) -> None:
    """
    Validate minimal invariants of an event log.

    This function performs lightweight checks intended to catch common data-shape
    errors early (e.g., corrupted logs or generator/estimator mismatches).

    Parameters
    ----------
    log : EventLog
        Event log to validate.

    Raises
    ------
    ValueError
        If the log violates required invariants.
    """
    if not log.events:
        raise ValueError("EventLog.events is empty")

    if log.events[0].type is not EventType.BLOCK_START:
        raise ValueError("First event must be BLOCK_START")

    for i, e in enumerate(log.events):
        if int(e.idx) != i:
            raise ValueError(f"Bad idx at position {i}: event.idx={e.idx}")

        # Event-type specific minimum payload checks
        if e.type is EventType.CHOICE:
            if "choice" not in e.payload:
                raise ValueError(f"CHOICE event at idx={i} missing payload['choice']")
        elif e.type is EventType.OUTCOME:
            if "action" not in e.payload:
                raise ValueError(f"OUTCOME event at idx={i} missing payload['action']")
            if "observed_outcome" not in e.payload:
                raise ValueError(f"OUTCOME event at idx={i} missing payload['observed_outcome']")
