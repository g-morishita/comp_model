"""Vicarious reinforcement learning (VRL) model implementation.

This module implements a social RL agent that learns exclusively from
**demonstrator outcomes**, ignoring the agent's own outcomes.

References
----------
Burke CJ, Tobler PN, Baddeley M, Schultz W (2010) Neural mechanisms of observational
learning. Proc Natl Acad Sci U S A 107(32): 14431-14436.
https://doi.org/10.1073/pnas.1003111107

Examples
--------
>>> from comp_model_impl.models.vicarious_rl.vicarious_rl import Vicarious_RL
>>> model = Vicarious_RL(alpha_o=0.3, beta=4.0)
>>> model.get_params()["alpha_o"]
0.3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.requirements import RequireAnyDemoOutcomeObservable, RequireSocialBlock, Requirement, RequireAllSelfOutcomesHidden
from comp_model_core.params import ParameterSchema
from comp_model_core.interfaces.block_runner import SocialObservation
from comp_model_core.spec import EnvironmentSpec
from comp_model_core.utility import _softmax

from .schema import vicarious_rl_schema


@dataclass(slots=True)
class Vicarious_RL(SocialComputationalModel):
    """Vicarious reinforcement learning model (chosen-only updates).

    Parameters
    ----------
    alpha_o : float
        Vicarious outcome learning rate (demonstrator outcomes).
    beta : float
        Softmax inverse temperature.

    Notes
    -----
    - Works for any ``spec.n_actions >= 2`` with social observations.
    - The model ignores private outcomes (self outcomes must be hidden).

    Update flow
    -----------
    Equations (chosen-only):

    - ``pi(a|s) = softmax(Q_s, beta)``
    - Social: ``Q_s[d] <- Q_s[d] + alpha_o * (o_other - Q_s[d])``

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
    >>> model = Vicarious_RL(alpha_o=0.2, beta=3.0)
    >>> model.reset_block(spec=spec)
    >>> _ = model.action_probs(state=0, spec=spec)
    >>> model.social_update(
    ...     state=0,
    ...     social=SocialObservation(others_choices=[1], observed_others_outcomes=[1.0]),
    ...     spec=spec,
    ... )

    References
    ----------
    Burke CJ, Tobler PN, Baddeley M, Schultz W (2010) Neural mechanisms of observational
    learning. Proc Natl Acad Sci U S A 107(32): 14431-14436.
    https://doi.org/10.1073/pnas.1003111107

    Examples
    --------
    >>> from comp_model_impl.models.vicarious_rl.vicarious_rl import Vicarious_RL
    >>> model = Vicarious_RL(alpha_o=0.2, beta=3.0)
    >>> model.get_params()["alpha_o"]
    0.2
    """
    alpha_o: float = 0.2
    beta: float = 3.0
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
        """Return the parameter schema for VRL.

        Returns
        -------
        ParameterSchema
            Schema with ``alpha_o`` and ``beta`` parameters.
        """
        return vicarious_rl_schema(
            alpha_o_default=float(self.alpha_o),
            beta_default=float(self.beta),
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
        rng: np.random.Generator | None = None,
    ) -> None:
        """Update values from a demonstrator outcome.

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
        rng : numpy.random.Generator or None, optional
            Optional RNG (unused by this deterministic update).
        """
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
            # chosen-only social shaping toward pseudo_reward
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
        rng : numpy.random.Generator or None, optional
            Optional RNG (unused).
        """
        return
