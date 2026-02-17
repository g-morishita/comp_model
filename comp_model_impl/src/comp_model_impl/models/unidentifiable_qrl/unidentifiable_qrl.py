"""Q-learning (QRL) model implementation.

This module implements a standard, asocial Q-learning agent with a softmax
choice rule. It is intended for tasks with observable private outcomes and
at least two actions.

Examples
--------
Instantiate the model and query action probabilities:

>>> from comp_model_impl.models.qrl.qrl import QRL
>>> from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
>>> spec = EnvironmentSpec(
...     n_actions=2,
...     outcome_type=OutcomeType.BINARY,
...     outcome_range=(0.0, 1.0),
...     outcome_is_bounded=True,
...     is_social=False,
...     state_kind=StateKind.DISCRETE,
...     n_states=1,
... )
>>> model = QRL(alpha=0.3, beta=4.0)
>>> model.reset_block(spec=spec)
>>> probs = model.action_probs(state=0, spec=spec)
>>> probs.shape
(2,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.requirements import RequireAnySelfOutcomeObservable, RequireAsocialBlock, Requirement
from comp_model_core.spec import EnvironmentSpec
from comp_model_core.utility import _softmax
from comp_model_core.params import ParameterSchema

from .schema import unidentifiable_qrl_schema


@dataclass(slots=True)
class UnidentifiableQRL(ComputationalModel):
    """Standard asocial Q-learning model with softmax choice.

    Parameters
    ----------
    alpha_1 : float
        Learning rate for private outcomes.
    alpha_2 : float
        Learning rate for private outcomes.
    beta : float
        Softmax inverse temperature.

    Notes
    -----
    - Updates are **chosen-only** (the selected action's value is updated).
    - The model is asocial and requires at least one trial with observable
      self outcome (see :meth:`requirements`).

    Update flow
    -----------
    Equations (chosen-only):

    - ``pi(a|s) = softmax(Q_s, beta)``
    - ``Q_s[a] <- Q_s[a] + (alpha_1 + alpha_2) * (r - Q_s[a])``

    Code example
    ------------
    >>> from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind
    >>> spec = EnvironmentSpec(
    ...     n_actions=2,
    ...     outcome_type=OutcomeType.BINARY,
    ...     outcome_range=(0.0, 1.0),
    ...     outcome_is_bounded=True,
    ...     is_social=False,
    ...     state_kind=StateKind.DISCRETE,
    ...     n_states=1,
    ... )
    >>> model = QRL(alpha=0.2, beta=3.0)
    >>> model.reset_block(spec=spec)
    >>> _ = model.action_probs(state=0, spec=spec)
    >>> model.update(state=0, action=1, outcome=1.0, spec=spec)

    Examples
    --------
    >>> from comp_model_impl.models.qrl.qrl import QRL
    >>> model = QRL(alpha=0.2, beta=3.0)
    >>> model.get_params()["alpha"]
    0.2
    """

    alpha_1: float = 0.2
    alpha_2: float = 0.2
    beta: float = 5.0
    def __post_init__(self) -> None:
        """Initialize latent state containers."""
        self._q: list[np.ndarray] = []

    @classmethod
    def requirements(cls) -> tuple[Requirement, ...]:
        """Return plan/data requirements for this model.

        Returns
        -------
        tuple[Requirement, ...]
            Requirements that enforce asocial blocks and observable outcomes.
        """
        # QRL requires an asocial task and at least one trial with self outcome
        # observable (possibly noisy).
        return (
            RequireAsocialBlock(),
            RequireAnySelfOutcomeObservable(),
        )

    @property
    def param_schema(self) -> ParameterSchema:
        """Return the parameter schema for QRL.

        Returns
        -------
        ParameterSchema
            Schema with ``alpha`` and ``beta`` parameters.
        """
        return unidentifiable_qrl_schema(
            alpha_1_default=float(self.alpha_1),
            alpha_2_default=float(self.alpha_2),
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
            True if the task is asocial and has at least two actions.
        """
        return (not spec.is_social) and int(spec.n_actions) >= 2

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        """Reset latent state at the start of a block.

        Parameters
        ----------
        spec : EnvironmentSpec
            Environment contract for the upcoming block.
        """
        self._q = []

    def _ensure_state(self, s: int, n_actions: int) -> None:
        """Ensure that the internal Q-table contains state ``s``.

        Parameters
        ----------
        s : int
            State index.
        n_actions : int
            Number of actions in the environment.
        """
        while len(self._q) <= s:
            self._q.append(np.zeros(n_actions, dtype=float))
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
        return _softmax(self._q[s], self.beta)

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
        if outcome is None:
            return
        s = int(state)
        nA = int(spec.n_actions)
        self._ensure_state(s, nA)

        a = int(action)
        if 0 <= a < nA:
            self._q[s][a] += (float(self.alpha_1) + float(self.alpha_2)) * (float(outcome) - self._q[s][a])
