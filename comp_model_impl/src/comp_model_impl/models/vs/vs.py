"""Value Shaping (VS) model for social reinforcement learning.

The VS model combines private outcome learning with **value shaping** from
demonstrations, treating others' actions as a pseudo-reward signal.
It also has perseveration bonus, which adds bonus to its own previous choice.

References
----------
Najar A, Bonnet E, Bahrami B, Palminteri S (2020) The actions of others act as a
pseudo-reward to drive imitation in the context of social reinforcement learning.
PLoS Biol 18(12): e3001028. https://doi.org/10.1371/journal.pbio.3001028
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, List

import numpy as np

from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.requirements import RequireAllDemoOutcomesHidden, RequireSocialBlock, RequireAnySelfOutcomeObservable, Requirement
from comp_model_core.params import ParameterSchema
from comp_model_core.interfaces.bandit import SocialObservation
from comp_model_core.spec import EnvironmentSpec
from comp_model_core.utility import _softmax

from .schema import vs_schema
from ..common import perseveration_bonus


@dataclass(slots=True)
class VS(SocialComputationalModel):
    """Value Shaping model generalized to K arms (chosen-only updates).

    Parameters
    ----------
    alpha_p : float
        Private outcome learning rate.
    alpha_i : float
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
    - Private update: only chosen action is updated.
    - Social update: only demonstrated action is updated.
    - Latents reset per block via :meth:`reset_block`.

    Update flow
    -----------
    Equations (chosen-only):

    - ``pi(a|s) = softmax(Q_s + kappa * I[a = last_choice], beta)``
    - Social: ``Q_s[d] <- Q_s[d] + alpha_i * (pseudo_reward - Q_s[d])``
    - Private: ``Q_s[a] <- Q_s[a] + alpha_p * (r - Q_s[a])``

    References
    ----------
    Najar A, Bonnet E, Bahrami B, Palminteri S (2020) The actions of others act as a
    pseudo-reward to drive imitation in the context of social reinforcement learning.
    PLoS Biol 18(12): e3001028. https://doi.org/10.1371/journal.pbio.3001028
    """
    alpha_p: float = 0.2
    alpha_i: float = 0.2
    beta: float = 3.0
    kappa: float = 0.0
    pseudo_reward: float = 1.0  # not estimated by default
    
    # config (not estimated)
    beta_max: float = 20.0
    kappa_abs_max: float = 1.0
    _init_q_value: float = 0.0
    _q: List[float] = field(default_factory=list)
    _last_choice: list[int | None] = field(default_factory=list)

    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        """Return plan/data requirements for this social model.

        Returns
        -------
        tuple[Requirement, ...]
            Requirements enforcing social blocks and observable private outcomes.
        """
        return (
            RequireSocialBlock(),
            RequireAllDemoOutcomesHidden(),
            RequireAnySelfOutcomeObservable(),
        )

    @property
    def param_schema(self) -> ParameterSchema:
        """Return the parameter schema for VS.

        Returns
        -------
        ParameterSchema
            Schema with learning rates, inverse temperature, and perseveration.
        """
        return vs_schema(
            alpha_p_default=float(self.alpha_p),
            alpha_i_default=float(self.alpha_i),
            beta_default=float(self.beta),
            kappa_default=float(self.kappa),
            beta_max=float(self.beta_max),
            kappa_abs_max=float(self.kappa_abs_max),
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
        self._last_choice = []

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
            self._q.append(np.tile(float(self._init_q_value), n_actions))
        while len(self._last_choice) <= s:
            self._last_choice.append(None)

        # If action count differs across blocks (rare), reset that state's vector safely.
        if self._q[s].shape[0] != n_actions:
            self._q[s] = np.zeros(n_actions, dtype=float)
            self._last_choice[s] = None

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
        u = q + perseveration_bonus(self._last_choice[s], nA, self.kappa)
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
        """Update values from a private outcome.

        Parameters
        ----------
        state : Any
            Current state identifier.
        action : int
            Action index taken by the agent.
        outcome : float or None
            Observed private outcome (None if unobserved).
        spec : EnvironmentSpec
            Environment contract.
        info : Mapping[str, Any] or None, optional
            Optional metadata.
        """
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
