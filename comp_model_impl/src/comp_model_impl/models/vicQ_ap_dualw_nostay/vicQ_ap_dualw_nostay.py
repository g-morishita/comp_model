"""Vicarious-Q + Action-Policy Learning (Social RL), no perseveration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import softmax

from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.interfaces.block_runner import SocialObservation
from comp_model_core.params import ParameterSchema
from comp_model_core.requirements import (
    RequireAllSelfOutcomesHidden,
    RequireAnyDemoOutcomeObservable,
    RequireSocialBlock,
    Requirement,
)
from comp_model_core.spec import EnvironmentSpec

from .schema_nostay import vicQ_ap_dualw_nostay_schema


@dataclass(slots=True)
class VicQ_AP_DualW_NoStay(SocialComputationalModel):
    """Vicarious-Q + Action-Policy Learning without stay/perseveration."""

    alpha_o: float = 0.2
    alpha_a: float = 0.2
    beta: float = 6.0
    w: float = 0.5

    # config (not estimated)
    beta_max: float = 20.0
    initial_q: float = 0.0

    _init_q_value: float = 0.0
    _q: Sequence[float] = field(default_factory=list)
    _demo_pi: Sequence[float] = field(default_factory=list)

    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        return (
            RequireSocialBlock(),
            RequireAnyDemoOutcomeObservable(),
            RequireAllSelfOutcomesHidden(),
        )

    @property
    def param_schema(self) -> ParameterSchema:
        return vicQ_ap_dualw_nostay_schema(
            alpha_o_default=float(self.alpha_o),
            alpha_a_default=float(self.alpha_a),
            beta_default=float(self.beta),
            w_default=float(self.w),
            beta_max=float(self.beta_max),
        )

    def supports(self, spec: EnvironmentSpec) -> bool:
        return spec.is_social and spec.n_actions >= 2 and spec.n_states == 1

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        nA = spec.n_actions
        self._q = np.tile(float(self.initial_q), nA)
        self._demo_pi = np.ones(nA, dtype=float) / nA

    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        nA = int(spec.n_actions)

        g = (self._demo_pi - 1 / nA) / (1 - 1 / nA)
        social_drive = float(self.w) * self._q + (1.0 - float(self.w)) * g
        u = float(self.beta) * social_drive

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
        self._demo_pi = self._demo_pi + self.alpha_a * (onehot - self._demo_pi)

        if not social.observed_others_outcomes:
            return

        oo = float(social.observed_others_outcomes[0])
        self._q[co] += float(self.alpha_o) * (float(oo) - self._q[co])

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
        return
