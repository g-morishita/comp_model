"""Vicarious Action-Policy-gated Value Shaping (AP-VS) model.

The Vicarious AP-VS model combines the demonstrator's outcome learning with **value shaping** from
demonstrations, treating others' actions as a pseudo-reward signal.
The scial value-shaping learning rate is modulated by the demonstrator's action policy.
In particular, it tracks the demonstrator's action policy and computes the reliability 
(i.e., 1 - standarized entropy of the policy).
The inferred action-policy reliability modified the VS learning rate.
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

from .schema import vicarious_ap_vs_schema
from ..common import perseveration_bonus


def _entropy_based_reliability(demo_pi, n_actions):
    """As a legacy, keep it here but do not use.
    The reliability is too small, the parameter recovery failed.
    """
    rel = 1 - entropy(demo_pi) / np.log(n_actions)
    rel = float(np.clip(rel, 0.0, 1.0))
    return rel


def _determinism_index_reliability(demo_pi, n_actions):
    rel = np.max(demo_pi) - 1 / n_actions
    rel = rel / (1 - 1 / n_actions)
    rel = float(np.clip(rel, 0.0, 1.0))
    return rel

@dataclass(slots=True)
class Vicarious_AP_VS(SocialComputationalModel):
    """Vicarious Action-Policy-gated Value Shaping (AP-VS) model.

   Parameters
    ----------
    alpha_o : float, optional
        Learning rate of the demonstrator's outcome
    alpha_vs_base : float, optional
        Social value-shape (VS) base learning rate.
    alpha_a : float, optional
        Learning rate of the demonstrator's action
    beta : float, optional
        Inverse temperature.
    kappa : float, optional
        Perseveration parameter.
    pseudo_reward : float
        Target used on demonstrations (default 1.0).
    beta_max : float, optional
        Maximum allowed beta.
    kappa_abs_max : float, optional
        Maximum absolute kappa.

    Model contract
    --------------
    Update:
        No updates.

    Social update:
        1) When the demonstrator's action is observed,
            the infered action policy of the demonstrator is updated.
            demo_pi(demo_a) <- demo_pi(demo_a) + alpha_a * (1 - demo_pi(demo_a))
        2) Also, the VS update happens:
            alpha_vs = alpha_vs_base * reliability(demo_pi)
            Q(demo_a) <- Q(demo_a) + alpha_vs * (pseudo_reward - Q(dmeo_a))
        3) If outcome is observed: apply vicarious learning rule:
            Q(demo_a) <- Q(mode_a) + alpha_o * (r - Q(demo_a))

    Decision:
        pi_s ~ exp(beta * Q + I[previous_choice] * kappa)

    Notes
    -----
    - This model does not use its own outcome to update the action values
    unlike the normal VS model.
    """
    alpha_o: float = 0.2
    alpha_vs_base: float = 0.2
    alpha_a: float = 0.2
    beta: float = 3.0
    kappa: float = 0.0
    pseudo_reward: float = 1.0  # not estimated by default

    # config (not estimated)
    beta_max: float = 20.0
    kappa_abs_max: float = 5.0
    _init_q_value: float = 0.0
    _q: Sequence[float] = field(default_factory=list)
    _demo_pi: Sequence[float] = field(default_factory=list)
    _last_choice: int | None = None

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
        return vicarious_ap_vs_schema(
            alpha_o_default=float(self.alpha_o),
            alpha_vs_base_default=float(self.alpha_vs_base),
            alpha_a_default=float(self.alpha_a),
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
        self._last_choice = None

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

        u = self._q * self.beta + perseveration_bonus(self._last_choice, nA, self.kappa)
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
        1) When the demonstrator's action is observed,
            the infered action policy of the demonstrator is updated.
            demo_pi(demo_a) <- demo_pi(demo_a) + alpha_a * (1 - demo_pi(demo_a))
        2) Also, the VS update happens:
            alpha_vs = alpha_vs_base * reliability(demo_pi)
            Q(demo_a) <- Q(demo_a) + alpha_vs * (pseudo_reward - Q(dmeo_a))
        3) If outcome is observed: apply vicarious learning rule:
            Q(demo_a) <- Q(mode_a) + alpha_o * (r - Q(demo_a))


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
        nA = int(spec.n_actions)

        if 0 <= d < nA:
            # the infered action policy of the demonstrator is updated.
            onehot = np.zeros(nA, dtype=float)
            onehot[d] = 1.0
            self._demo_pi = self._demo_pi + self.alpha_a * (onehot - self._demo_pi)
            self._demo_pi = np.clip(self._demo_pi, 1e-8, 1.0)
            self._demo_pi = self._demo_pi / self._demo_pi.sum()

            # the VS update happens
            rel = _determinism_index_reliability(self._demo_pi, spec.n_actions)
            alpha_vs = self.alpha_vs_base * rel
            self._q[d] = self._q[d] + alpha_vs * (self.pseudo_reward - self._q[d])

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
            self._last_choice = action
        else:
            raise ValueError("Action must be not None.")
