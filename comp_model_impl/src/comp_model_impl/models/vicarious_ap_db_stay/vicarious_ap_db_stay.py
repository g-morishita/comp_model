"""Vicarious Action-Policy-gated (AP) Decision-Bias (DB) model with perseveration (stay).

The Vicarious AP-DB-Stay model combines the demonstrator's outcome learning with **decision bias** from
demonstrations, biasing toward the demonstrator's action.
The decision bias parameter is modulated by the demonstrator's action policy.
In particular, it tracks the demonstrator's action policy and computes the reliability.
It also has perseveration bonus, which adds bonus to its own previous choice.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import entropy
from scipy.special import softmax

from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.requirements import RequireAnyDemoOutcomeObservable, RequireSocialBlock, Requirement
from comp_model_core.params import ParameterSchema
from comp_model_core.interfaces.bandit import SocialObservation
from comp_model_core.spec import EnvironmentSpec

from .schema import vicarious_db_vs_stay_schema
from ..common import perseveration_bonus


def determinism_index_reliability(demo_pi: np.ndarray, n_actions: int) -> float:
    """
    Compute demonstrator reliability using a determinism (peakedness) index.

    Reliability reflects how strongly the demonstrator favors a single action.

    Parameters
    ----------
    demo_pi : np.ndarray
        Estimated demonstrator action policy (probability vector).
    n_actions : int
        Number of available actions.

    Returns
    -------
    float
        Reliability value in [0, 1].
        0 = uniform/random policy
        1 = fully deterministic policy
    """

    uniform_prob = 1.0 / n_actions
    rel = (np.max(demo_pi) - uniform_prob) / (1.0 - uniform_prob)

    return float(np.clip(rel, 0.0, 1.0))


@dataclass(slots=True)
class Vicarious_AP_DB_STAY(SocialComputationalModel):
    """Vicarious Action-Policy-gated (AP) Decision-Bias (DB) model with perseveration (stay).

   Parameters
    ----------
    alpha_o : float, optional
        Learning rate of the demonstrator's outcome
    alpha_a : float, optional
        Learning rate of the demonstrator's action to compute reliability
    demo_bias_rel : float, optional
        Reliability modulation of the demonstrator-choice decision bias; scales how strongly reliability increases (or decreases) copying.
    beta : float, optional
        Inverse temperature.
    kappa : float, optional
        Perseveration parameter.
    beta_max : float, optional
        Maximum allowed beta.
    kappa_abs_max : float, optional
        Maximum absolute kappa.

    Model contract
    --------------

        Overview
        --------
        The agent learns action values exclusively from observing a demonstrator.
        The agent additionally infers the demonstrator’s action policy to estimate
        partner reliability. Reliability modulates copying of the demonstrator’s
        action during decision making.

        Update
        ------
        No self-outcome learning.
        The agent does not update action values using its own reward outcomes.

        Social update
        -------------

        1) Demonstrator action-policy learning
        The agent maintains an estimated demonstrator policy over K actions:

            demo_pi_t(i) = P(demonstrator chooses action i)

        After observing demonstrator action demo_a at trial t:

            For all actions i:
                demo_pi_{t+1}(i) =
                    demo_pi_t(i) + alpha_a * (y_t(i) - demo_pi_t(i))

        where:
            y_t(i) = 1 if i == demo_a, otherwise 0

        2) Partner reliability computation
        Reliability reflects how deterministic the demonstrator's policy is.
        Reliability is computed using a normalized maximum-probability index:

            Rel_t =
                (max_i demo_pi_t(i) - 1/K) / (1 - 1/K)

        where K is the number of available actions.
        Rel_t is bounded to [0, 1].

        3) Vicarious outcome learning
        If the demonstrator's reward outcome r_t is observed, the agent updates
        the value of the demonstrator's chosen action:

            Q_{t+1}(demo_a) =
                Q_t(demo_a) + alpha_o * (r_t - Q_t(demo_a))

        Decision
        --------
        The agent selects action a_t according to a softmax choice rule including:

        • value-based choice
        • perseveration (stay bias)
        • reliability-modulated demonstrator copying bias

        P(a_t = i) ∝ exp(
            beta * Q_t(i)
            + kappa * I[i == a_{t-1}]
            + (demo_bias_rel * Rel_t) * I[i == demo_a]
        )

    Notes
    -----
    • Action values are updated only from demonstrator outcomes.
    • Reliability is inferred solely from demonstrator action statistics.
    • Reliability influences behavior through decision bias, not value learning.
    """
    alpha_o: float = 0.2
    alpha_a: float = 0.2
    demo_bias_rel: float = 1.0
    beta: float = 3.0
    kappa: float = 0.0

    # config (not estimated)
    beta_max: float = 20.0
    kappa_abs_max: float = 5.0
    demo_bias_abs_max: float = 5.0
    
    _init_q_value: float = 0.0
    _q: Sequence[float] = field(default_factory=list)
    _demo_pi: Sequence[float] = field(default_factory=list)
    _last_self_choice: int | None = None
    _recent_demo_choice: int | None = None

    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        """Return plan/data requirements for this social model.
        The demonstrator's outcome is observable at least once.

        Returns
        -------
        tuple[Requirement, ...]
            Requirements enforcing social blocks and observable private outcomes.
        """
        return (
            RequireSocialBlock(),
            RequireAnyDemoOutcomeObservable(),
        )

    @property
    def param_schema(self) -> ParameterSchema:
        """Return the parameter schema for Vicarious AP-VS model.

        Returns
        -------
        ParameterSchema
            Schema with learning rates, inverse temperature, and perseveration.
        """
        return vicarious_db_vs_stay_schema(
            alpha_o_default=float(self.alpha_o),
            alpha_a_default=float(self.alpha_a),
            demo_bias_rel_default=float(self.demo_bias_rel),
            beta_default=float(self.beta),
            kappa_default=float(self.kappa),
            demo_bias_rel_abs_max=float(self.demo_bias_abs_max),
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
            True if the task
                1) is social 
                2) has at least 2 actions
                3) the number state is 1.
        """
        return spec.is_social and spec.n_actions >= 2 and spec.n_states == 1
    
    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        """Reset latent state at the start of a block.

        Parameters
        ----------
        spec : EnvironmentSpec
            Environment contract for the upcoming block.
        """
        self._q = np.tile(float(self._init_q_value), spec.n_actions)
        self._demo_pi = np.ones(spec.n_actions, dtype=float) / spec.n_actions
        self._last_self_choice = None

    def action_probs(self, *, state: Any, spec: EnvironmentSpec) -> np.ndarray:
        """Compute action probabilities under the softmax policy.

        Utility includes:
        - value-based term: beta * Q
        - self-choice perseveration: kappa * 1[a == last_self_choice]
        - demonstrator-choice bias: (demo_bias_rel * Rel_t) * 1[a == recent_demo_choice]
        where Rel_t is computed from the inferred demonstrator policy demo_pi.

        Parameters
        ----------
        state : Any
            Current state identifier (unused if task is stateless).
        spec : EnvironmentSpec
            Environment contract.

        Returns
        -------
        np.ndarray
            Action probability vector of length ``spec.n_actions``.
        """
        nA = int(spec.n_actions)

        demo_pi = np.asarray(self._demo_pi, dtype=float)
        s = demo_pi.sum()
        if s > 0:
            demo_pi = demo_pi / s  # robust normalization

        rel = determinism_index_reliability(demo_pi=demo_pi, n_actions=nA)
        demo_bias = self.demo_bias_rel * rel

        u = (
            self._q * self.beta
            + perseveration_bonus(self._last_self_choice, nA, self.kappa)
            + perseveration_bonus(self._recent_demo_choice, nA, demo_bias)
        )
        return softmax(u)

    def social_update(
        self,
        *,
        state: Any,
        social: SocialObservation,
        spec: EnvironmentSpec,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """Update values from a demonstrator observation.

        1) Demonstrator action-policy learning
        The agent maintains an estimated demonstrator policy over K actions:

            demo_pi_t(i) = P(demonstrator chooses action i)

        After observing demonstrator action demo_a at trial t:

            For all actions i:
                demo_pi_{t+1}(i) =
                    demo_pi_t(i) + alpha_a * (y_t(i) - demo_pi_t(i))

        where:
            y_t(i) = 1 if i == demo_a, otherwise 0

        2) Vicarious outcome learning
        If the demonstrator's reward outcome r_t is observed, the agent updates
        the value of the demonstrator's chosen action:

            Q_{t+1}(demo_a) =
                Q_t(demo_a) + alpha_o * (r_t - Q_t(demo_a))

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
        self._recent_demo_choice = d
        nA = int(spec.n_actions)

        if 0 <= d < nA:
            # the infered action policy of the demonstrator is updated.
            onehot = np.zeros(nA, dtype=float)
            onehot[d] = 1.0
            self._demo_pi = self._demo_pi + self.alpha_a * (onehot - self._demo_pi)
            self._demo_pi = np.clip(self._demo_pi, 1e-8, 1.0)
            self._demo_pi = self._demo_pi / self._demo_pi.sum()

            # If outcome is observed: apply vicarious learning rule
            if social.observed_others_outcomes:
                oo = float(social.observed_others_outcomes[0])
                self._q[d] = self._q[d] + self.alpha_o * (oo - self._q[d])
        else:
            raise ValueError("Observed action is out of range.")

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
            Observed private outcome (None if unobserved).
        spec : EnvironmentSpec
            Environment contract.
        info : Mapping[str, Any] or None, optional
            Optional metadata.
        """
        if action is not None:
            self._last_self_choice = action
        else:
            raise ValueError("Action must be not None.")
