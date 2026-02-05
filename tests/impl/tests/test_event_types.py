"""Tests for event log schema types."""

from __future__ import annotations

import pytest

from comp_model_core.events.types import Event, EventLog, EventType, validate_event_log


def test_event_type_codes():
    """EventType integer codes are stable."""
    assert int(EventType.BLOCK_START) == 1
    assert int(EventType.SOCIAL_OBSERVED) == 2
    assert int(EventType.CHOICE) == 3
    assert int(EventType.OUTCOME) == 4


def test_event_round_trip_json():
    """Event serialization round-trips through JSON mapping."""
    e = Event(idx=1, type=EventType.CHOICE, t=0, state=1, payload={"choice": 1})
    j = e.to_json()
    e2 = Event.from_json(j)
    assert e2.idx == e.idx
    assert e2.type == e.type
    assert e2.t == e.t
    assert e2.state == e.state
    assert dict(e2.payload) == dict(e.payload)


def test_event_log_round_trip_json():
    """EventLog serialization round-trips through JSON mapping."""
    log = EventLog(events=[Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": "c1"})])
    j = log.to_json()
    log2 = EventLog.from_json(j)
    assert log2.schema_version == log.schema_version
    assert dict(log2.metadata or {}) == dict(log.metadata or {})
    assert len(log2.events) == 1
    assert log2.events[0].type is EventType.BLOCK_START


def test_validate_event_log_minimal_ok():
    """validate_event_log accepts a minimal valid sequence."""
    log = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": "c1"}),
            Event(idx=1, type=EventType.CHOICE, t=0, state=0, payload={"choice": 1, "available_actions": [0, 1]}),
            Event(idx=2, type=EventType.OUTCOME, t=0, state=0, payload={"action": 1, "observed_outcome": 1.0}),
        ]
    )
    validate_event_log(log)


def test_validate_event_log_rejects_empty_and_bad_start():
    """validate_event_log rejects empty logs and non-BLOCK_START first events."""
    with pytest.raises(ValueError):
        validate_event_log(EventLog(events=[]))

    log = EventLog(events=[Event(idx=0, type=EventType.CHOICE, t=0, state=0, payload={"choice": 1})])
    with pytest.raises(ValueError):
        validate_event_log(log)


def test_validate_event_log_rejects_bad_idx_and_payloads():
    """validate_event_log catches index mismatches and missing payload keys."""
    log = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": "c1"}),
            Event(idx=2, type=EventType.CHOICE, t=0, state=0, payload={"choice": 1}),
        ]
    )
    with pytest.raises(ValueError):
        validate_event_log(log)

    log2 = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": "c1"}),
            Event(idx=1, type=EventType.CHOICE, t=0, state=0, payload={}),
        ]
    )
    with pytest.raises(ValueError):
        validate_event_log(log2)

    log3 = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"condition": "c1"}),
            Event(idx=1, type=EventType.OUTCOME, t=0, state=0, payload={"action": 0}),
        ]
    )
    with pytest.raises(ValueError):
        validate_event_log(log3)
