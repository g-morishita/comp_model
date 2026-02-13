"""Mean-Variance-Skewness (MVS) lottery-choice model.

Utility for action ``a`` is computed as:

``U_a = E[X_a] + lambda_var * Var[X_a] + delta * Skew[X_a]``

where ``Skew`` is standardized skewness. Choices follow a softmax policy
with inverse temperature ``beta``.

This model is stateless (no learning update).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.params import ParameterSchema
from comp_model_core.requirements import RequireAsocialBlock, Requirement
from comp_model_core.spec import EnvironmentSpec
from comp_model_core.utility import _softmax

from .schema import mvs_schema


def _moments_from_lottery(lottery: Mapping[str, Any]) -> tuple[float, float, float]:
    if "outcomes" not in lottery or "probs" not in lottery:
        raise ValueError("Lottery mapping must contain 'outcomes' and 'probs'.")
    x = np.asarray(lottery["outcomes"], dtype=float)
    p = np.asarray(lottery["probs"], dtype=float)
    if x.ndim != 1 or p.ndim != 1 or x.shape[0] != p.shape[0] or x.shape[0] == 0:
        raise ValueError("Invalid lottery shapes: expected equal-length 1D outcomes/probs.")
    if np.any(p < 0.0):
        raise ValueError("Lottery probabilities must be non-negative.")
    s = float(np.sum(p))
    if s <= 0.0:
        raise ValueError("Lottery probabilities must sum to a positive value.")
    p = p / s
    mu = float(np.sum(p * x))
    dev = x - mu
    var = float(np.sum(p * (dev ** 2)))
    if var <= 1e-12:
        skew = 0.0
    else:
        skew = float(np.sum(p * (dev ** 3)) / (var ** 1.5))
    return mu, var, skew


def _normalize_action_moments(*, state: Any, n_actions: int) -> np.ndarray:
    """Extract action moments as an array with shape ``(n_actions, 3)``."""
    # Preferred format from LotteryChoiceBanditEnv state payload.
    if isinstance(state, Mapping) and "action_moments" in state:
        raw = state["action_moments"]
    # Fallback: compute moments from explicit lottery definitions.
    elif isinstance(state, Mapping) and "lotteries" in state:
        lots = state["lotteries"]
        if not isinstance(lots, Sequence) or isinstance(lots, (str, bytes)):
            raise ValueError("state['lotteries'] must be a sequence.")
        if len(lots) != int(n_actions):
            raise ValueError(
                f"Expected {n_actions} lotteries in state, got {len(lots)}."
            )
        raw = [_moments_from_lottery(l) for l in lots]
    else:
        raw = state

    arr = np.asarray(raw, dtype=float)
    if arr.shape != (int(n_actions), 3):
        raise ValueError(
            f"MVS expects state moments with shape ({n_actions}, 3), got {arr.shape}."
        )
    if np.any(~np.isfinite(arr)):
        raise ValueError("MVS received non-finite moment values in state.")
    return arr


@dataclass(slots=True)
class MVS(ComputationalModel):
    """Stateless MVS utility model for lottery-choice tasks."""

    lambda_var: float = -0.25
    delta: float = 0.2
    beta: float = 2.5

    lambda_abs_max: float = 10.0
    delta_abs_max: float = 10.0
    beta_max: float = 20.0

    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        return (RequireAsocialBlock(),)

    @property
    def param_schema(self) -> ParameterSchema:
        return mvs_schema(
            lambda_var_default=float(self.lambda_var),
            delta_default=float(self.delta),
            beta_default=float(self.beta),
            lambda_abs_max=float(self.lambda_abs_max),
            delta_abs_max=float(self.delta_abs_max),
            beta_max=float(self.beta_max),
        )

    def supports(self, spec: EnvironmentSpec) -> bool:
        return (not spec.is_social) and int(spec.n_actions) >= 2

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        return

    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        n_actions = int(spec.n_actions)
        moments = _normalize_action_moments(state=state, n_actions=n_actions)
        means = moments[:, 0]
        variances = moments[:, 1]
        skewness = moments[:, 2]
        utility = means + float(self.lambda_var) * variances + float(self.delta) * skewness
        return _softmax(utility, float(self.beta))

    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float | None,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        # Static preferences model: no trial-to-trial learning.
        return
