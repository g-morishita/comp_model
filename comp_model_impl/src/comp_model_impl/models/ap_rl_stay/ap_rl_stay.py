"""Action-policy RL model with stay/perseveration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import softmax

from comp_model_core.interfaces.block_runner import SocialObservation
from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.params import ParameterSchema
from comp_model_core.requirements import (
    RequireAllSelfOutcomesHidden,
    RequireSocialBlock,
    Requirement,
)
from comp_model_core.spec import EnvironmentSpec

from ..common import perseveration_bonus
from .schema import ap_rl_stay_schema


@dataclass(slots=True)
class AP_RL_Stay(SocialComputationalModel):
    """Action-policy-only social model with self-choice perseveration."""

    alpha_a: float = 0.2
    beta: float = 6.0
    kappa: float = 0.0

    # config (not estimated)
    beta_max: float = 20.0
    kappa_abs_max: float = 5.0

    _demo_pi: Sequence[float] = field(default_factory=list)
    _last_self_choice: int | None = None

    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        return (
            RequireSocialBlock(),
            RequireAllSelfOutcomesHidden(),
        )

    @property
    def param_schema(self) -> ParameterSchema:
        return ap_rl_stay_schema(
            alpha_a_default=float(self.alpha_a),
            beta_default=float(self.beta),
            kappa_default=float(self.kappa),
            beta_max=float(self.beta_max),
            kappa_abs_max=float(self.kappa_abs_max),
        )

    def supports(self, spec: EnvironmentSpec) -> bool:
        return spec.is_social and spec.n_actions >= 2 and spec.n_states == 1

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        nA = int(spec.n_actions)
        self._demo_pi = np.ones(nA, dtype=float) / nA
        self._last_self_choice = None

    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        nA = int(spec.n_actions)
        g = (self._demo_pi - 1 / nA) / (1 - 1 / nA)
        u = float(self.beta) * g + perseveration_bonus(self._last_self_choice, nA, self.kappa)
        return softmax(u)

    def social_update(
        self,
        *,
        state: Any,
        social: SocialObservation,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not social.others_choices:
            return

        nA = int(spec.n_actions)
        co = int(social.others_choices[0])
        if not (0 <= co < nA):
            raise ValueError("Observed action is out of range.")

        onehot = np.zeros(nA, dtype=float)
        onehot[co] = 1.0
        self._demo_pi = self._demo_pi + float(self.alpha_a) * (onehot - self._demo_pi)

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
        if action is None:
            return

        a = int(action)
        if 0 <= a < int(spec.n_actions):
            self._last_self_choice = a
        else:
            raise ValueError("Action is out of range.")
