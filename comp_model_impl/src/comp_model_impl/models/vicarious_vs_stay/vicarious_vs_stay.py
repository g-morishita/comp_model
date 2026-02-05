"""Vicarious + Value Shaping (Vicarious VS) + stay(i.e., perseveration) model.

This model combines vicarious outcome learning with value shaping from
demonstrations. It learns solely from demonstrator information and does not
use private outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.requirements import RequireAnyDemoOutcomeObservable, RequireSocialBlock, RequireAllSelfOutcomesHidden, Requirement
from comp_model_core.params import ParameterSchema
from comp_model_core.interfaces.bandit import SocialObservation
from comp_model_core.spec import EnvironmentSpec
from comp_model_core.utility import _softmax

from .schema import vicarious_vs_stay_schema
from ..common import perseveration_bonus


@dataclass(slots=True)
class Vicarious_VS_Stay(SocialComputationalModel):
    """Vicarious + Value Shaping model generalized to K arms.

    Parameters
    ----------
    alpha_o : float
        Other's outcome learning rate (vicarious learning).
    alpha_a : float
        Social value-shaping learning rate (pseudo-reward toward demonstrated
        action).
    beta : float
        Softmax inverse temperature.
    kappa : float
        Perseveration strength: +kappa bonus to repeating last private choice.
    pseudo_reward : float
        Target used on demonstrations (default 1.0).
    beta_max : float
        Upper bound used by estimators (not estimated directly).
    kappa_abs_max : float
        Absolute bound for ``kappa`` used by estimators.

    Notes
    -----
    - Works for any ``spec.n_actions >= 2`` and only ``spec.n_states == 1`` with social observations.
    - No update for self chosen action (self outcomes are ignored).
    - Social update: only demonstrated action is updated.

    Update flow
    -----------
    Equations (chosen-only):
    - ``pi(a|s) = softmax(Q_s + kappa * I[a = last_choice], beta)``
    - Social (two-step):
      ``Q_s[d] <- Q_s[d] + alpha_a * (pseudo_reward - Q_s[d])``
      ``Q_s[d] <- Q_s[d] + alpha_o * (o_other - Q_s[d])``
    """
    alpha_o: float = 0.2
    alpha_a: float = 0.2
    beta: float = 3.0
    kappa: float = 0.3
    pseudo_reward: float = 1.0  # not estimated by default
    
    # config (not estimated)
    beta_max: float = 20.0
    kappa_max: float = 1.0
    _init_q_val: float = 0.0

    _prev_self_action: int | None = field(init=False, default=None, repr=False)
    _q: np.ndarray = field(init=False, repr=False)
    
    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        """Return plan/data requirements for this social model.

        Returns
        -------
        tuple[Requirement, ...]
            Requirements enforcing social blocks and observable demo outcomes
            with hidden self outcomes.
        """
        return (
            RequireSocialBlock(),
            RequireAnyDemoOutcomeObservable(),
            RequireAllSelfOutcomesHidden(),
        )

    @property
    def param_schema(self) -> ParameterSchema:
        """Return the parameter schema for Vicarious VS.

        Returns
        -------
        ParameterSchema
            Schema with ``alpha_o``, ``alpha_a``, ``beta``, ``kappa`` parameters.
        """
        return vicarious_vs_stay_schema(
            alpha_o_default=float(self.alpha_o),
            alpha_a_default=float(self.alpha_a),
            beta_default=float(self.beta),
            beta_max=float(self.beta_max),
        )

    def supports(self, spec: EnvironmentSpec) -> bool:
        """Return True if the model supports the environment spec.

        Parameters
        ----------
        spec : EnvironmentSpec
            Environment contract to check.

        Returns
        -------
        bool
            True if the task is social, has at least two actions and has only one state.
        """
        return spec.is_social and spec.n_actions >= 2 and spec.n_states == 1

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        """Reset latent state at the start of a block.

        Parameters
        ----------
        spec : EnvironmentSpec
            Environment contract for the upcoming block.
        """
        self._prev_self_action = None
        self._q = np.full(spec.n_actions, self._init_q_val, dtype=float)
            
    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        """Compute action probabilities under the softmax policy.

        Parameters
        ----------
        state : Any
            Current state identifier.
        spec : EnvironmentSpec
            Environment contract.

        Returns
        -------
        numpy.ndarray
            Action probability vector of length ``spec.n_actions``.
        """
        nA = int(spec.n_actions)
        u = self._q.copy() + perseveration_bonus(self._prev_self_action, nA, self.kappa)
        return _softmax(u, self.beta)

    def social_update(
        self,
        *,
        state: Any,
        social: SocialObservation,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """Update values from a demonstrator observation.

        Parameters
        ----------
        state : Any
            Current state identifier.
        social : SocialObservation
            Demonstrator observation (choices and outcomes).
        spec : EnvironmentSpec
            Environment contract.
        info : Mapping[str, Any] or None, optional
            Optional metadata.
        """
        if not social.others_choices:
            return

        d = int(social.others_choices[0])
        oo = float(social.observed_others_outcomes[0])

        nA = int(spec.n_actions)

        if 0 <= d < nA:
            # chosen-only social shaping toward pseudo_reward
            self._q[d] += float(self.alpha_a) * (float(self.pseudo_reward) - self._q[d])
            self._q[d] += float(self.alpha_o) * (oo - self._q[d])
            self._prev_self_action = d

    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float | None,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """No-op private update (self outcomes are ignored).

        Parameters
        ----------
        state : Any
            Current state identifier.
        action : int
            Action index taken by the agent.
        outcome : float or None
            Observed private outcome (ignored).
        spec : EnvironmentSpec
            Environment contract.
        info : Mapping[str, Any] or None, optional
            Optional metadata.
        """
        return
