"""Vicarious + Value Shaping (Vicarious VS) model.

This model combines vicarious outcome learning with value shaping from
demonstrations. It learns solely from demonstrator information and does not
use private outcomes.

Examples
--------
>>> from comp_model_impl.models.vicarious_vs.vicarious_vs import Vicarious_VS
>>> model = Vicarious_VS(alpha_o=0.2, alpha_a=0.3, beta=4.0)
>>> model.get_params()["alpha_o"]
0.2
"""

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
    pseudo_reward : float
        Target used on demonstrations (default 1.0).
    beta_max : float
        Upper bound used by estimators (not estimated directly).

    Notes
    -----
    - Works for any ``spec.n_actions >= 2`` with social observations.
    - No update for self chosen action (self outcomes are ignored).
    - Social update: only demonstrated action is updated.

    Update flow
    -----------
    Equations (chosen-only):

    - ``pi(a|s) = softmax(Q_s, beta)``
    - Social (two-step):
      ``Q_s[d] <- Q_s[d] + alpha_a * (pseudo_reward - Q_s[d])``
      ``Q_s[d] <- Q_s[d] + alpha_o * (o_other - Q_s[d])``

    Code example
    ------------
    >>> from comp_model_core.interfaces.block_runner import SocialObservation
    >>> from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
    >>> spec = EnvironmentSpec(
    ...     n_actions=2,
    ...     outcome_type=OutcomeType.BINARY,
    ...     outcome_range=(0.0, 1.0),
    ...     outcome_is_bounded=True,
    ...     is_social=True,
    ...     state_kind=StateKind.DISCRETE,
    ...     n_states=1,
    ... )
    >>> model = Vicarious_VS(alpha_o=0.2, alpha_a=0.3, beta=3.0)
    >>> model.reset_block(spec=spec)
    >>> _ = model.action_probs(state=0, spec=spec)
    >>> model.social_update(
    ...     state=0,
    ...     social=SocialObservation(others_choices=[1], observed_others_outcomes=[1.0]),
    ...     spec=spec,
    ... )

    Examples
    --------
    >>> from comp_model_impl.models.vicarious_vs.vicarious_vs import Vicarious_VS
    >>> model = Vicarious_VS(alpha_o=0.2, alpha_a=0.3, beta=3.0)
    >>> model.get_params()["alpha_a"]
    0.3
    """
    alpha_o: float = 0.2
    alpha_a: float = 0.2
    beta: float = 3.0
    pseudo_reward: float = 1.0  # not estimated by default
    
    # config (not estimated)
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        """Initialize latent state containers."""
        self._q: list[np.ndarray] = []

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
            Schema with ``alpha_o``, ``alpha_a``, and ``beta`` parameters.
        """
        return vicarious_vs_schema(
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
            True if the task is social and has at least two actions.
        """
        return spec.is_social and spec.n_actions >= 2

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        """Reset latent state at the start of a block.

        Parameters
        ----------
        spec : EnvironmentSpec
            Environment contract for the upcoming block.
        """
        self._q = []

    def _ensure_state(self, s: int, n_actions: int) -> None:
        """Ensure internal arrays are initialized for state ``s``.

        Parameters
        ----------
        s : int
            State index.
        n_actions : int
            Number of actions in the environment.
        """
        while len(self._q) <= s:
            self._q.append(np.zeros(n_actions, dtype=float))

        # If action count differs across blocks (rare), reset that state's vector safely.
        if self._q[s].shape[0] != n_actions:
            self._q[s] = np.zeros(n_actions, dtype=float)
            
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
