from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.requirements import RequireAnyDemoOutcomeObservable, RequireSocialBlock, RequireAllSelfOutcomesHidden, Requirement
from comp_model_core.params import ParameterSchema
from comp_model_core.interfaces.bandit import SocialObservation
from comp_model_core.spec import EnvironmentSpec
from comp_model_core.utility import _softmax

from .schema import vicarious_vs_schema


@dataclass(slots=True)
class Vicarious_VS(SocialComputationalModel):
    """
    Vicarious + Value Shaping model generalized to K arms (chosen-only updates).

    Parameters
    ----------
    alpha_o : float
        Other's outcome learning rate (i.e., vicarious learning).
    alpha_a : float
        Social value-shaping learning rate (pseudo-reward toward demonstrated action).
    beta : float
        Softmax inverse temperature.
    pseudo_reward : float
        Target used on demonstrations (default 1.0).

    Notes
    -----
    - Works for any spec.n_actions >= 2.
    - No update for self chosen action
    - Social update: only demonstrated action is updated.
    """
    alpha_o: float = 0.2
    alpha_a: float = 0.2
    beta: float = 3.0
    pseudo_reward: float = 1.0  # not estimated by default
    
    # config (not estimated)
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        self._q: list[np.ndarray] = []

    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        return (
            RequireSocialBlock(),
            RequireAnyDemoOutcomeObservable(),
            RequireAllSelfOutcomesHidden(),
        )

    @property
    def param_schema(self) -> ParameterSchema:
        return vicarious_vs_schema(
            alpha_o_default=float(self.alpha_o),
            alpha_a_default=float(self.alpha_a),
            beta_default=float(self.beta),
            beta_max=float(self.beta_max),
        )

    def supports(self, spec: EnvironmentSpec) -> bool:
        return spec.is_social and spec.n_actions >= 2

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        self._q = []

    def _ensure_state(self, s: int, n_actions: int) -> None:
        while len(self._q) <= s:
            self._q.append(np.zeros(n_actions, dtype=float))

        # If action count differs across blocks (rare), reset that state's vector safely.
        if self._q[s].shape[0] != n_actions:
            self._q[s] = np.zeros(n_actions, dtype=float)
            self._last_choice[s] = None

    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        q = self._q[s]
        return _softmax(q, self.beta)

    def social_update(
        self,
        *,
        state: Any,
        social: SocialObservation,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        if not social.others_choices:
            return

        d = int(social.others_choices[0])
        oo = float(social.observed_others_outcomes[0])

        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        if 0 <= d < nA:
            # chosen-only social shaping toward pseudo_reward
            self._q[s][d] += float(self.alpha_a) * (float(self.pseudo_reward) - self._q[s][d])
            self._q[s][d] += float(self.alpha_o) * (oo - self._q[s][d])

    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float | None,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        return
