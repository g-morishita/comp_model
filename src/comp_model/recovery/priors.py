from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math
import numpy as np


class Transform(Protocol):
    def forward(self, z: float) -> float: ...


@dataclass(frozen=True, slots=True)
class UnitInterval:
    """(0,1) via logistic."""
    def forward(self, z: float) -> float:
        # stable logistic
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)


@dataclass(frozen=True, slots=True)
class Positive:
    """(0,inf) via softplus."""
    def forward(self, z: float) -> float:
        if z > 20:
            return z
        return math.log1p(math.exp(z))


@dataclass(frozen=True, slots=True)
class Real:
    def forward(self, z: float) -> float:
        return z


@dataclass(frozen=True, slots=True)
class NormalPrior:
    mu: float
    sigma: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mu, self.sigma))
