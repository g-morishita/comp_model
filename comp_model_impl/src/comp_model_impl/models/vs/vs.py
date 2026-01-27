"""Value Shaping (VS) model.

Implements a social reinforcement learning model that combines private outcome
learning with value shaping from demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.requirements import RequireAllDemoOutcomesHidden, RequireSocialBlock, RequireAnySelfOutcomeObservable, Requirement
from comp_model_core.params import ParameterSchema
from comp_model_core.interfaces.bandit import SocialObservation
from comp_model_core.spec import EnvironmentSpec
from comp_model_core.utility import _softmax

from .schema import vs_schema

def _perseveration_bonus(last_choice: int | None, n_actions: int, kappa: float) -> np.ndarray:
    """
    K-armed perseveration: add +kappa to the last chosen action, 0 elsewhere.
    """
    if last_choice is None or kappa == 0.0:
        return np.zeros(n_actions, dtype=float)
    b = np.zeros(n_actions, dtype=float)
    if 0 <= last_choice < n_actions:
        b[last_choice] = float(kappa)
    return b


@dataclass(slots=True)
class VS(SocialComputationalModel):
    """
    Value Shaping model generalized to K arms (chosen-only updates).

    Parameters
    ----------
    alpha_p : float
        Private outcome learning rate.
    alpha_i : float
        Social value-shaping learning rate (pseudo-reward toward demonstrated action).
    beta : float
        Softmax inverse temperature.
    kappa : float
        Perseveration strength: +kappa bonus to repeating last private choice.
    pseudo_reward : float
        Target used on demonstrations (default 1.0).

    Notes
    -----
    - Works for any spec.n_actions >= 2.
    - Private update: only chosen action is updated.
    - Social update: only demonstrated action is updated.
    - Latents reset per block via reset_block().
    """
    alpha_p: float = 0.2
    alpha_i: float = 0.2
    beta: float = 3.0
    kappa: float = 0.0
    pseudo_reward: float = 1.0  # not estimated by default
    
    # config (not estimated)
    beta_max: float = 20.0
    kappa_abs_max: float = 1.0

    def __post_init__(self) -> None:
        self._q: list[np.ndarray] = []
        self._last_choice: list[int | None] = []

    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        return (
            RequireSocialBlock(),
            RequireAllDemoOutcomesHidden(),
            RequireAnySelfOutcomeObservable(),
        )

    @property
    def param_schema(self) -> ParameterSchema:
        return vs_schema(
            alpha_p_default=float(self.alpha_p),
            alpha_i_default=float(self.alpha_i),
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
        while len(self._last_choice) <= s:
            self._last_choice.append(None)

        # If action count differs across blocks (rare), reset that state's vector safely.
        if self._q[s].shape[0] != n_actions:
            self._q[s] = np.zeros(n_actions, dtype=float)
            self._last_choice[s] = None

    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        q = self._q[s]
        u = q + _perseveration_bonus(self._last_choice[s], nA, self.kappa)
        return _softmax(u, self.beta)

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

        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        if 0 <= d < nA:
            # chosen-only social shaping toward pseudo_reward
            self._q[s][d] += float(self.alpha_i) * (float(self.pseudo_reward) - self._q[s][d])

    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float | None,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        if outcome is None:
            return
        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        a = int(action)
        if 0 <= a < nA:
            # chosen-only private learning toward realized outcome
            self._q[s][a] += float(self.alpha_p) * (float(outcome) - self._q[s][a])
            self._last_choice[s] = a
