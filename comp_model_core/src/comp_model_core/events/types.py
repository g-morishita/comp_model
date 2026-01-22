from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence


# Where generators store the block event log inside Block.metadata
EVENT_LOG_KEY = "event_log"


class EventType(IntEnum):
    """
    Discrete event types for an event-log "experience trace".

    Keep this mapping stable. Stan and other estimators will rely on the integer codes.
    """
    BLOCK_START = 1
    SOCIAL_OBSERVED = 2
    CHOICE = 3
    OUTCOME = 4


@dataclass(frozen=True, slots=True)
class Event:
    """
    A single event in time.

    idx:   0-based index within the block event stream
    type:  EventType
    t:     trial index (0-based) within block, or None if not applicable
    state: model state identifier used for this event (often int)
    payload: event-specific fields (JSON-ish)
    """
    idx: int
    type: EventType
    t: int | None
    state: Any | None
    payload: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "idx": int(self.idx),
            "type": int(self.type),
            "t": None if self.t is None else int(self.t),
            "state": self.state,
            "payload": dict(self.payload),
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "Event":
        return Event(
            idx=int(d["idx"]),
            type=EventType(int(d["type"])),
            t=None if d.get("t") is None else int(d["t"]),
            state=d.get("state"),
            payload=dict(d.get("payload", {})),
        )


@dataclass(frozen=True, slots=True)
class EventLog:
    """Block-level event stream."""
    events: Sequence[Event]
    schema_version: int = 1
    metadata: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "metadata": dict(self.metadata or {}),
            "events": [e.to_json() for e in self.events],
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "EventLog":
        return EventLog(
            schema_version=int(d.get("schema_version", 1)),
            metadata=dict(d.get("metadata", {})),
            events=[Event.from_json(x) for x in d.get("events", [])],
        )


def validate_event_log(log: EventLog) -> None:
    """
    Minimal invariants:
    - non-empty
    - first event is BLOCK_START
    - idx is 0..n-1
    - CHOICE has payload['choice']
    - OUTCOME has payload['action'] and payload['observed_outcome']
    """
    if not log.events:
        raise ValueError("EventLog.events is empty")

    if log.events[0].type is not EventType.BLOCK_START:
        raise ValueError("First event must be BLOCK_START")

    for i, e in enumerate(log.events):
        if int(e.idx) != i:
            raise ValueError(f"Bad idx at position {i}: event.idx={e.idx}")

        if e.type is EventType.CHOICE and "choice" not in e.payload:
            raise ValueError(f"CHOICE event idx={i} missing payload['choice']")

        if e.type is EventType.OUTCOME:
            if "action" not in e.payload or "observed_outcome" not in e.payload:
                raise ValueError(
                    f"OUTCOME event idx={i} missing payload['action'] or payload['observed_outcome']"
                )
