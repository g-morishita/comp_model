from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

class RewardType(Enum):
    BINARY = auto()
    CONTINUOUS = auto()

@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Defines the contract a task/bandit provides."""
    n_actions: int
    reward_type: RewardType
    reward_range: Optional[Tuple[float, float]] = None  # e.g. (0,1), (0,10), etc.
    reward_is_bounded: bool = False
    is_social: bool = False
    has_state: bool = False