"""
Event-log schema for portable "experience traces".

An *event log* is a sequence of discrete events emitted by a generator (or read from
data) that defines **the exact timing/order** of observations, choices, outcomes,
and social observations. This makes replay-based inference robust to "flow mismatch"
bugs where a generator and estimator disagree on the update order.

The event log is intended to be:

- **Serializable** (JSON-friendly)
- **Stable** across languages (e.g., for Stan)
- **Minimal** (small set of event types plus per-event payload dictionaries)

See Also
--------
comp_model_core.events.types.EventLog
comp_model_core.events.accessors.get_event_log
"""

from .types import EventType, Event, EventLog, validate_event_log
from .accessors import get_event_log

__all__ = [
    "EventType",
    "Event",
    "EventLog",
    "validate_event_log",
    "get_event_log",
]
