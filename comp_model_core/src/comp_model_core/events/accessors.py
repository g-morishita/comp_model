"""comp_model_core.events.accessors

Convenience accessors for event logs stored in blocks.

The canonical storage location is :attr:`comp_model_core.data.types.Block.event_log`.
"""

from __future__ import annotations

from typing import Mapping, TYPE_CHECKING

from comp_model_core.events.types import EventLog, validate_event_log
from comp_model_core.data.types import Block


def get_event_log(block: Block) -> EventLog:
    """
    Retrieve and validate the event log stored in :attr:`~comp_model_core.data.types.Block.event_log`.

    Parameters
    ----------
    block : Block
        Block containing an event log.

    Returns
    -------
    EventLog
        Parsed and validated event log.

    Raises
    ------
    KeyError
        If the block does not contain :attr:`~comp_model_core.data.types.Block.event_log`.
    TypeError
        If the stored value is neither an :class:`~comp_model_core.events.types.EventLog`
        nor a mapping that can be parsed as one.
    ValueError
        If the parsed event log fails validation (see :func:`validate_event_log`).
    """
    raw = block.event_log
    if raw is None:
        raise KeyError(f"Block {block.block_id!r} is missing Block.event_log")

    if isinstance(raw, EventLog):
        log = raw
    elif isinstance(raw, Mapping):
        log = EventLog.from_json(raw)  # type: ignore[arg-type]
    else:
        raise TypeError(f"Block.event_log must be dict or EventLog, got {type(raw)}")

    validate_event_log(log)
    return log
