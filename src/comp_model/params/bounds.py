from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Bound:
    lo: float
    hi: float

    def clip(self, x: float) -> float:
        if x < self.lo:
            return self.lo
        if x > self.hi:
            return self.hi
        return x


@dataclass(frozen=True, slots=True)
class ParameterBoundsSpace:
    """
    Direct parameterization (no transforms).

    - names define parameter order in the optimization vector x
    - bounds defines box constraints
    """
    names: Sequence[str]
    bounds: Mapping[str, Bound]

    def dim(self) -> int:
        return len(self.names)

    def clip_vec(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (self.dim(),):
            raise ValueError(f"Expected x.shape == ({self.dim()},)")
        y = x.copy()
        for i, name in enumerate(self.names):
            b = self.bounds[name]
            y[i] = b.clip(float(y[i]))
        return y

    def to_params(self, x: np.ndarray) -> dict[str, float]:
        x = self.clip_vec(x)
        return {name: float(x[i]) for i, name in enumerate(self.names)}

    def sample_init(self, rng: np.random.Generator) -> np.ndarray:
        x = np.empty((self.dim(),), dtype=float)
        for i, name in enumerate(self.names):
            b = self.bounds[name]
            x[i] = float(rng.uniform(b.lo, b.hi))
        return x
