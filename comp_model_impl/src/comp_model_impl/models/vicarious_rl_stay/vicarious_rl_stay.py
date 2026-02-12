"""Vicarious reinforcement learning (VRL) model with stay/perseveration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from comp_model_core.interfaces.block_runner import SocialObservation
from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.params import ParameterSchema
from comp_model_core.requirements import (
    RequireAllSelfOutcomesHidden,
    RequireAnyDemoOutcomeObservable,
    RequireSocialBlock,
    Requirement,
)
from comp_model_core.spec import EnvironmentSpec
from comp_model_core.utility import _softmax

from ..common import perseveration_bonus
from .schema import vicarious_rl_stay_schema


@dataclass(slots=True)
class Vicarious_RL_Stay(SocialComputationalModel):
    """Vicarious RL with a self-action perseveration term."""

    alpha_o: float = 0.2
    beta: float = 3.0
    kappa: float = 0.0

    # config (not estimated)
    beta_max: float = 20.0
    kappa_abs_max: float = 5.0

    def __post_init__(self) -> None:
        self._q: list[np.ndarray] = []
        self._last_choice: list[int | None] = []

    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        return (
            RequireSocialBlock(),
            RequireAnyDemoOutcomeObservable(),
            RequireAllSelfOutcomesHidden(),
        )

    @property
    def param_schema(self) -> ParameterSchema:
        return vicarious_rl_stay_schema(
            alpha_o_default=float(self.alpha_o),
            beta_default=float(self.beta),
            kappa_default=float(self.kappa),
            beta_max=float(self.beta_max),
            kappa_abs_max=float(self.kappa_abs_max),
        )

    def supports(self, spec: EnvironmentSpec) -> bool:
        return spec.is_social and spec.n_actions >= 2

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        self._q = []
        self._last_choice = []

    def _ensure_state(self, s: int, n_actions: int) -> None:
        while len(self._q) <= s:
            self._q.append(np.zeros(n_actions, dtype=float))
            self._last_choice.append(None)

        if self._q[s].shape[0] != n_actions:
            self._q[s] = np.zeros(n_actions, dtype=float)
            self._last_choice[s] = None

    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        u = self._q[s] + perseveration_bonus(self._last_choice[s], nA, self.kappa)
        return _softmax(u, self.beta)

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
        if not social.observed_others_outcomes:
            return

        co = int(social.others_choices[0])
        oo = float(social.observed_others_outcomes[0])

        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        if 0 <= co < nA:
            self._q[s][co] += float(self.alpha_o) * (float(oo) - self._q[s][co])

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

        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        a = int(action)
        if 0 <= a < nA:
            self._last_choice[s] = a
        else:
            raise ValueError("Action is out of range.")
