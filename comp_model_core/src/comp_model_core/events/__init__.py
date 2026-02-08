"""
Event-log schema, accessors, and converters for portable "experience traces".

An *event log* is a sequence of discrete events emitted by a generator (or derived
from real data) that defines **the exact timing/order** of observations, choices,
outcomes, and social observations. This makes replay-based inference robust to
"flow mismatch" bugs where a generator and estimator disagree on update order.

This module provides:
- **Schema**: `EventType`, `Event`, `EventLog`, `validate_event_log`
- **Accessors**: `get_event_log`
- **Converters**: utilities to convert common dataset formats into `EventLog`s:
  - trial-wise tables (one row per trial)
  - combined self + social/demo rows
  - separate self and demo tables (merged/aggregated by keys)
  - convenience helper to attach missing logs to existing `StudyData`

Metadata policy
---------------
Dataset-specific columns are not modeled explicitly in the public API.
Unknown / extra fields are preserved as metadata (e.g., `Trial.info` and
`Trial.social_info`) so downstream code can audit provenance without coupling
the library to a particular dataset schema.

Design goals
------------
- **Serializable** (JSON-friendly)
- **Stable** across languages (e.g., for Stan)
- **Minimal** (small set of event types plus per-event payload dictionaries)
- **Deterministic flow** (event order is explicit and replayable)

See Also
--------
comp_model_core.events.types.EventLog
comp_model_core.events.accessors.get_event_log
comp_model_core.events.convert.event_log_from_any_rows
comp_model_core.events.convert.event_log_from_partner_self_rows
"""

from .types import EventType, Event, EventLog, validate_event_log
from .accessors import get_event_log
from .convert import (
    TrialTableColumns,
    PartnerSelfTrialTableColumns,
    attach_event_logs,
    event_log_from_any_rows,
    event_log_from_partner_self_rows,
    event_log_from_rows,
    event_log_from_separate_self_demo_rows,
    event_log_from_trials,
    merge_self_and_demo_rows,
    trials_from_partner_self_rows,
    trials_from_rows,
)


__all__ = [
    "EventType",
    "Event",
    "EventLog",
    "validate_event_log",
    "get_event_log",
    "TrialTableColumns",
    "PartnerSelfTrialTableColumns",
    "event_log_from_trials",
    "trials_from_rows",
    "event_log_from_rows",
    "trials_from_partner_self_rows",
    "event_log_from_partner_self_rows",
    "merge_self_and_demo_rows",
    "event_log_from_separate_self_demo_rows",
    "event_log_from_any_rows",
    "attach_event_logs",
]
