from __future__ import annotations
from typing import Mapping

from comp_model_core.data.types import Block
from comp_model_core.events.types import EVENT_LOG_KEY, EventLog, validate_event_log

def get_event_log(block: Block) -> EventLog:
    md = block.metadata or {}
    if EVENT_LOG_KEY not in md:
        raise KeyError(f"Block {block.block_id!r} missing metadata[{EVENT_LOG_KEY!r}]")
    raw = md[EVENT_LOG_KEY]
    if isinstance(raw, EventLog):
        log = raw
    elif isinstance(raw, Mapping):
        log = EventLog.from_json(raw)  # type: ignore[arg-type]
    else:
        raise TypeError(f"metadata[{EVENT_LOG_KEY!r}] must be dict or EventLog, got {type(raw)}")
    validate_event_log(log)
    return log
