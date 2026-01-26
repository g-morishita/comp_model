import pytest

from comp_model_core.data.types import Block, Trial
from comp_model_core.events.accessors import get_event_log
from comp_model_core.events.types import (
    Event,
    EventLog,
    EventType,
    validate_event_log,
)


def test_event_to_from_json_roundtrip():
    e = Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": "b"})
    d = e.to_json()
    e2 = Event.from_json(d)
    assert e2.idx == 0
    assert e2.type is EventType.BLOCK_START
    assert e2.payload["block_id"] == "b"


def test_event_log_validate_happy_path_and_roundtrip():
    events = [
        Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": "b"}),
        Event(idx=1, type=EventType.CHOICE, t=0, state=0, payload={"choice": 1, "available_actions": [0, 1]}),
        Event(idx=2, type=EventType.OUTCOME, t=0, state=0, payload={"action": 1, "observed_outcome": 1.0}),
    ]
    log = EventLog(events=events, metadata={"m": 1})
    validate_event_log(log)

    d = log.to_json()
    log2 = EventLog.from_json(d)
    validate_event_log(log2)
    assert len(log2.events) == 3
    assert log2.events[1].type is EventType.CHOICE


def test_validate_event_log_failures():
    # Empty
    with pytest.raises(ValueError):
        validate_event_log(EventLog(events=[]))

    # First not BLOCK_START
    bad = EventLog(events=[Event(idx=0, type=EventType.CHOICE, t=0, state=None, payload={"choice": 0})])
    with pytest.raises(ValueError):
        validate_event_log(bad)

    # Bad indices
    bad2 = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={}),
            Event(idx=2, type=EventType.CHOICE, t=0, state=None, payload={"choice": 0}),
        ]
    )
    with pytest.raises(ValueError):
        validate_event_log(bad2)

    # Missing choice payload
    bad3 = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={}),
            Event(idx=1, type=EventType.CHOICE, t=0, state=None, payload={}),
        ]
    )
    with pytest.raises(ValueError):
        validate_event_log(bad3)

    # Invalid available_actions
    bad4 = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={}),
            Event(idx=1, type=EventType.CHOICE, t=0, state=None, payload={"choice": 0, "available_actions": []}),
        ]
    )
    with pytest.raises(ValueError):
        validate_event_log(bad4)

    # Missing outcome payload keys
    bad5 = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={}),
            Event(idx=1, type=EventType.OUTCOME, t=0, state=None, payload={"action": 0}),
        ]
    )
    with pytest.raises(ValueError):
        validate_event_log(bad5)


def test_get_event_log_from_block_metadata():
    # Valid log stored as object
    log = EventLog(
        events=[
            Event(idx=0, type=EventType.BLOCK_START, t=None, state=None, payload={"block_id": "b"}),
            Event(idx=1, type=EventType.CHOICE, t=0, state=None, payload={"choice": 0}),
            Event(idx=2, type=EventType.OUTCOME, t=0, state=None, payload={"action": 0, "observed_outcome": 1.0}),
        ]
    )
    block = Block(block_id="b", trials=[Trial(t=0, state=None, choice=None, observed_outcome=None, outcome=None)], event_log=log)
    out = get_event_log(block)
    assert isinstance(out, EventLog)
    assert out.events[0].type is EventType.BLOCK_START

    # Wrong type
    block4 = Block(block_id="b4", trials=[], event_log=123)
    with pytest.raises(TypeError):
        get_event_log(block4)

    # Invalid log should raise
    bad_log = EventLog(events=[Event(idx=0, type=EventType.CHOICE, t=0, state=None, payload={"choice": 0})])
    block5 = Block(block_id="b5", trials=[], event_log=bad_log)
    with pytest.raises(ValueError):
        get_event_log(block5)
