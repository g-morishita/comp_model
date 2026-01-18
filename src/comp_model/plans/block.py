from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BlockPlan:
    block_id: str
    n_trials: int
    bandit_config: Mapping[str, Any]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id must be non-empty")
        if int(self.n_trials) <= 0:
            raise ValueError("n_trials must be > 0")
        if not self.bandit_type:
            raise ValueError("bandit_type must be non-empty")
        if self.bandit_config is None:
            raise ValueError("bandit_config must not be None")
