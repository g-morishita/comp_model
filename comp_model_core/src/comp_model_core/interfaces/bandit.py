from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from abc import ABC, abstractmethod

import numpy as np

from ..spec import TaskSpec


@dataclass(frozen=True, slots=True)
class BanditStep:
    outcome: float
    done: bool = False
    info: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SocialObservation:
    others_choices: Sequence[int] | None = None
    others_outcomes: Sequence[float] | None = None
    info: dict[str, Any] | None = None


class Bandit(ABC):
    @property
    @abstractmethod
    def spec(self) -> TaskSpec: ...

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> Any: ...

    @abstractmethod
    def step(self, action: int, rng: np.random.Generator) -> BanditStep: ...

    @abstractmethod
    def get_state(self) -> Any: ...


class SocialBandit(Bandit, ABC):
    @abstractmethod
    def observe_others(self, rng: np.random.Generator) -> SocialObservation: ...
