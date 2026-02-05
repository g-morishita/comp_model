"""Simulation generators (trial-by-trial and event-log based).

This subpackage exposes generator classes that simulate subjects and optionally
record event logs for later likelihood replay or Stan fitting.

Examples
--------
>>> from comp_model_impl.generators import EventLogAsocialGenerator
>>> gen = EventLogAsocialGenerator()
"""

from .event_log import (
    EventLogAsocialGenerator,
    EventLogSocialPostOutcomeGenerator,
    EventLogSocialPreChoiceGenerator,
)

__all__ = [
    "EventLogAsocialGenerator",
    "EventLogSocialPreChoiceGenerator",
    "EventLogSocialPostOutcomeGenerator",
]
