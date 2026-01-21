from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.special import expit, logit, logsumexp

from ..errors import ParameterValidationError


class Transform(Protocol):
    def forward(self, z: float) -> float:
        """Map unconstrained z -> constrained x."""
        ...

    def inverse(self, x: float) -> float:
        """Map constrained x -> unconstrained z."""
        ...


@dataclass(frozen=True, slots=True)
class Identity:
    def forward(self, z: float) -> float:
        return float(z)

    def inverse(self, x: float) -> float:
        return float(x)


@dataclass(frozen=True, slots=True)
class Sigmoid:
    """(0,1) using scipy.special.expit/logit."""
    eps: float = 1e-12

    def forward(self, z: float) -> float:
        return float(expit(float(z)))

    def inverse(self, x: float) -> float:
        x = float(x)
        x = min(max(x, self.eps), 1.0 - self.eps)
        return float(logit(x))


@dataclass(frozen=True, slots=True)
class Softplus:
    """(0, +inf) with stable log(1+exp(z)) via logsumexp."""
    eps: float = 1e-12

    def forward(self, z: float) -> float:
        z = float(z)
        # softplus(z) = log(1 + exp(z)) = logsumexp([0, z])
        return float(logsumexp([0.0, z]))

    def inverse(self, x: float) -> float:
        """
        Inverse softplus: z = log(exp(x) - 1)
        Stable:
        - for small x: log(expm1(x))
        - for large x: ~ x
        """
        x = float(x)
        if x <= 0:
            raise ParameterValidationError(f"Softplus inverse requires x>0, got {x}.")
        if x > 30:
            return float(x)
        return float(np.log(np.expm1(max(x, self.eps))))


@dataclass(frozen=True, slots=True)
class BoundedTanh:
    """(lo, hi) via tanh; inverse via arctanh."""
    lo: float
    hi: float
    eps: float = 1e-12

    def forward(self, z: float) -> float:
        z = float(z)
        t = float(np.tanh(z))  # (-1,1)
        return float(self.lo + (self.hi - self.lo) * (t + 1.0) * 0.5)

    def inverse(self, x: float) -> float:
        x = float(x)
        # clip into (lo, hi) for invertibility
        x = min(max(x, self.lo + self.eps), self.hi - self.eps)
        y = (2.0 * (x - self.lo) / (self.hi - self.lo)) - 1.0
        y = min(max(y, -1.0 + self.eps), 1.0 - self.eps)
        return float(np.arctanh(y))
