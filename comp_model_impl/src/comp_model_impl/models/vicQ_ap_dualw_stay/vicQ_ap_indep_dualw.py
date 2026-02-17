"""
Vicarious-Q + Action-Policy Learning (Social RL)

This module implements a social reinforcement learning agent with two latent
representations learned from a demonstrator:

1) Vicarious action values (Qv): updated from the demonstrator's outcomes
   (vicarious RPE / outcome prediction error).

2) Demonstrator action policy (pi): learned from the demonstrator's observed
   actions (policy prediction error / action surprise).

At choice time, the agent integrates these two sources via separate decision
weights, combining value-based evidence (Qv) and policy-based evidence (pi) in
a softmax decision rule. This parameterization separates learning dynamics
(learning rates) from decision reliance (decision weights), enabling clean
comparisons of imitation- vs emulation-like control.

Typical decision form:
    P(a) ∝ exp( β_Q * Qv(a) + β_pi * g(pi(a)) )

where g(.) is a scale-fixed policy signal (e.g., log-probability or normalized
above-chance probability).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import softmax

from comp_model_core.interfaces.model import SocialComputationalModel
from comp_model_core.requirements import RequireAnyDemoOutcomeObservable, RequireSocialBlock, Requirement, RequireAllSelfOutcomesHidden
from comp_model_core.params import ParameterSchema
from comp_model_core.interfaces.block_runner import SocialObservation
from comp_model_core.spec import EnvironmentSpec

from .schema_indep_dualw import vicQ_ap_indep_dualw_schema
from ..common import perseveration_bonus

@dataclass(slots=True)
class VicQ_AP_IndepDualW(SocialComputationalModel):
    """Vicarious-Q + Action-Policy Learning (Social RL)

    Parameters
    ----------
    alpha_o : float, optional
        Vicarious outcome learning rate (demonstrator outcomes).
    alpha_a : float, optional
        Action policy learning rate (demonstrator action).
    beta_q : float, optional
        Weight of learning from the demonstrator's outcome.
    beta_a : float, optional
        Weight of learning from the dmeonstartor's actions
    kappa : float, optional
        Perseveration parameter.
    kappa_abs_max : float, optional
        Maximum absolute kappa.
        
    Model contract
    --------------

        Overview
        --------
        The agentlearning agent with two latent representations learned from a demonstrator.

        1) Vicarious action values (Qv): updated from the demonstrator's outcomes
        (vicarious RPE / outcome prediction error).

        2) Demonstrator action policy (pi): learned from the demonstrator's observed
        actions (policy prediction error / action surprise).

        At choice time, the agent integrates these two sources via separate decision
        weights, combining value-based evidence (Qv) and policy-based evidence (pi) in
        a softmax decision rule. This parameterization separates learning dynamics
        (learning rates) from decision reliance (decision weights), enabling clean
        comparisons of imitation- vs emulation-like control.

        Update
        ------
        No self-outcome learning.
        The agent does not update action values using its own reward outcomes.

        Social update
        -------------

        1) Demonstrator action-policy learning
        For possible action i

            demo_pi(i) = demo_pi(i) + alpha_a * (1[i == demo_a] - demo_pi(i))

        2) Vicarious outcome learning
        If the demonstrator's reward outcome r_t is observed, the agent updates
        the value of the demonstrator's chosen action:

            Q_{t+1}(demo_a) = Q_t(demo_a) + alpha_o * (r_t - Q_t(demo_a))

        Decision
        --------
        The agent selects action a_t according to a softmax choice rule including:

        - vicarious Q
        - tendency from action policy
        - perseveration (stay bias)

        P(a_t = i) ∝ exp(
            beta_q * Q_t(i)
            + beta_a * g(demo_pi)
            + kappa * I[i == prev_self_action]
        )

    Notes
    -----
    - Works for any ``spec.n_actions >= 2`` with social observations.
    """
    alpha_o: float = 0.2
    alpha_a: float = 0.2
    beta_q: float = 3.0
    beta_a: float = 3.0
    kappa: float = 3.0
    
    # config (not estimated)
    kappa_abs_max: float = 5.0
    initial_q: float = 0.0

    _init_q_value: float = 0.0
    _q: Sequence[float] = field(default_factory=list)
    _demo_pi: Sequence[float] = field(default_factory=list)
    _last_self_choice: int | None = None

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
        """
        return vicQ_ap_indep_dualw_schema(
            alpha_o_default=float(self.alpha_o),
            alpha_a_default=float(self.alpha_a),
            beta_q_default=float(self.beta_q),
            beta_a_default=float(self.beta_a),
            kappa_default=float(self.kappa),
            kappa_abs_max=float(self.kappa_abs_max)
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
        return spec.is_social and spec.n_actions >= 2 and spec.n_states == 1

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        """Reset latent state at the start of a block.

        Parameters
        ----------
        spec : EnvironmentSpec
            Environment contract for the upcoming block.
        """
        nA = spec.n_actions
        self._q = np.tile(float(self.initial_q), nA)
        self._demo_pi = np.ones(nA, dtype=float) / nA
        self._last_self_choice = None

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

        g = (self._demo_pi - 1 / nA) / (1 - 1 / nA)
        u = self.beta_q * self._q + self.beta_a * g + perseveration_bonus(self._last_self_choice, nA, self.kappa)

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

        # chosen-only social shaping toward pseudo_reward
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
        if action is None:
            return

        a = int(action)
        if 0 <= a < int(spec.n_actions):
            self._last_self_choice = a
        else:
            raise ValueError("Action is out of range.")
