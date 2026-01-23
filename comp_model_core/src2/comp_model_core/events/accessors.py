"""
Convenience accessors for event logs stored in blocks.
"""

from __future__ import annotations

from typing import Mapping

from comp_model_core.data.types import Block
from comp_model_core.events.types import EVENT_LOG_KEY, EventLog, validate_event_log


def get_event_log(block: Block) -> EventLog:
    """
    Retrieve and validate the event log stored in a block's metadata.

    Parameters
    ----------
    block : Block
        Block whose ``metadata`` contains an event log.

    Returns
    -------
    EventLog
        Parsed and validated event log.

    Raises
    ------
    KeyError
        If the block does not contain ``metadata[EVENT_LOG_KEY]``.
    TypeError
        If the stored value is neither an :class:`~comp_model_core.events.types.EventLog`
        nor a mapping that can be parsed as one.
    ValueError
        If the parsed event log fails validation (see :func:`validate_event_log`).

    Notes
    -----
    Generators may store either:

    - A concrete :class:`~comp_model_core.events.types.EventLog` instance, or
    - A JSON-friendly mapping produced by :meth:`~comp_model_core.events.types.EventLog.to_json`.
    """
    md = block.metadata or {}
    if EVENT_LOG_KEY not in md:
        raise KeyError(f"Block {block.block_id!r} missing metadata[{EVENT_LOG_KEY!r}]")

    raw = md[EVENT_LOG_KEY]
    if isinstance(raw, EventLog):
        log = raw
    elif isinstance(raw, Mapping):
        # Accept JSON-friendly dicts without requiring callers to instantiate EventLog explicitly.
        log = EventLog.from_json(raw)  # type: ignore[arg-type]
    else:
        raise TypeError(f"metadata[{EVENT_LOG_KEY!r}] must be dict or EventLog, got {type(raw)}")

    validate_event_log(log)
    return log
