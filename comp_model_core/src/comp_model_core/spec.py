from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

class OutcomeType(Enum):
    BINARY = auto()
    CONTINUOUS = auto()

@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Defines the contract a task/bandit provides."""
    n_actions: int
    outcome_type: OutcomeType
    outcome_range: Optional[Tuple[float, float]] = None  # e.g. (0,1), (0,10), etc.
    outcome_is_bounded: bool = False
    is_social: bool = False
    has_state: bool = False