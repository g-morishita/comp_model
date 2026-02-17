"""Q-learning with a stay (perseveration) bias.

This module defines an asocial Rescorla-Wagner/Q-learning agent that combines:

- chosen-only value updates from private outcomes, and
- an additive perseveration bonus for repeating the last chosen action.

Action probabilities are computed with a softmax over the augmented action
utilities. The implementation is intended for asocial tasks with at least two
actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import softmax

from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.requirements import RequireAnySelfOutcomeObservable, RequireAsocialBlock, Requirement
from comp_model_core.spec import EnvironmentSpec
from comp_model_core.params import ParameterSchema

from .schema import qrl_stay_schema
from ..common import perseveration_bonus


@dataclass(slots=True)
class QRL_Stay(ComputationalModel):
    """Asocial Q-learning model with softmax choice and stay bias.

    Parameters
    ----------
    alpha : float, optional
        Learning rate applied to chosen-action prediction errors.
    beta : float, optional
        Inverse temperature for softmax action selection.
    kappa : float, optional
        Perseveration strength. Positive values increase the utility of the
        previously selected action (stay bias), while negative values induce a
        switch tendency.
    kappa_abs_max : float, optional
        Absolute bound used by the parameter schema for ``kappa``.

    Notes
    -----
    This implementation currently assumes a single latent state
    (``spec.n_states == 1``), enforced by :meth:`supports`.

    Model contract
    --------------

        Overview
        --------
        The agent maintains one value per action and chooses with a softmax
        policy. A stay (perseveration) term adds utility to the previously
        chosen action.

        Update
        ------
        Let ``Q_t(i)`` be the value of action ``i`` at trial ``t``. For chosen
        action ``a_t`` and observed private outcome ``r_t``:

            Q_{t+1}(a_t) = Q_t(a_t) + alpha * (r_t - Q_t(a_t))

        Only the chosen action is updated.

        Decision
        --------
        Action probabilities are computed from utilities:

            u_t(i) = beta * Q_t(i) + kappa * I[i == a_{t-1}]
            P(a_t = i) = softmax(u_t)(i)
    """

    alpha: float = 0.2
    beta: float = 5.0
    kappa: float = 1.0
    kappa_abs_max: float = float("inf")
    _q: Sequence[float] = field(default_factory=list)
    _init_q_val: float = 0.0
    _last_choice: int | None = None

    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        """Return plan/data requirements for this model.

        Returns
        -------
        tuple[Requirement, ...]
            Requirements that enforce asocial blocks and observable outcomes.
        """
        # QRL_Stay requires an asocial task and at least one trial with self outcome
        # observable (possibly noisy).
        return (
            RequireAsocialBlock(),
            RequireAnySelfOutcomeObservable(),
        )

    @property
    def param_schema(self) -> ParameterSchema:
        """Return the parameter schema for QRL+Stay.

        Returns
        -------
        ParameterSchema
            Schema with ``alpha``, ``beta``, and ``kappa`` parameters.
        """
        return qrl_stay_schema(
            alpha_default=float(self.alpha),
            beta_default=float(self.beta),
            kappa_default=float(self.kappa),
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
            True if the task is asocial and has at least two actions.
        """
        return (not spec.is_social) and int(spec.n_actions) >= 2 and int(spec.n_states) == 1

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        """Reset latent state at the start of a block.

        Parameters
        ----------
        spec : EnvironmentSpec
            Environment contract for the upcoming block.
        """
        self._q = np.tile(float(self._init_q_val), spec.n_actions)
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
        """Update the chosen action value from an observed outcome.

        Parameters
        ----------
        state : Any
            Current state identifier.
        action : int
            Action index taken by the agent.
        outcome : float or None
            Observed outcome (None if unobserved).
        spec : EnvironmentSpec
            Environment contract.
        info : Mapping[str, Any] or None, optional
            Optional metadata from the environment/runner.
        rng : numpy.random.Generator or None, optional
            Optional RNG (unused by this deterministic update).
        """
        nA = int(spec.n_actions)

        if action is None:
            return

        a = int(action)
        if not (0 <= a < nA):
            return

        self._last_choice = a
        if outcome is not None:
            self._q[a] += float(self.alpha) * (float(outcome) - self._q[a])
