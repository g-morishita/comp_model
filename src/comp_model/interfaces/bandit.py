from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from ..spec import TaskSpec


@dataclass(frozen=True, slots=True)
class BanditStep:
    reward: float
    done: bool = False
    info: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SocialObservation:
    others_choices: Sequence[int] | None = None
    others_rewards: Sequence[float] | None = None
    info: dict[str, Any] | None = None


@runtime_checkable
class Bandit(Protocol):
    spec: TaskSpec

    def reset(self, rng: np.random.Generator) -> Any: ...
    def step(self, action: int, rng: np.random.Generator) -> BanditStep: ...
    def get_state(self) -> Any: ...


@runtime_checkable
class SocialBandit(Bandit, Protocol):
    def observe_others(self, rng: np.random.Generator) -> SocialObservation: ...
