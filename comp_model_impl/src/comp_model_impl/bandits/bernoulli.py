"""comp_model_impl.bandits.bernoulli

Bernoulli bandit environment implementation.

This environment generates binary outcomes (0/1) for each action (arm) according
to per-arm Bernoulli probabilities.

Important
---------
This environment produces **true outcomes** only. Trial-level outcome visibility and
observation noise (hidden / noisy feedback) are handled by block runners, not here.

See Also
--------
comp_model_core.interfaces.bandit.BanditEnv
comp_model_core.spec.EnvironmentSpec
comp_model_core.spec.OutcomeType
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from comp_model_core.interfaces.bandit import BanditEnv, EnvStep
from comp_model_core.spec import EnvironmentSpec, OutcomeType, StateKind


@dataclass(slots=True)
class BernoulliBanditEnv(BanditEnv):
    """Bernoulli bandit with fixed per-arm reward probabilities.

    Parameters
    ----------
    p : Sequence[float]
        Bernoulli success probabilities for each arm. Must have length ``n_actions``.
    n_states : int, optional
        Number of discrete states/contexts. Default is 1 (context-free bandit).
        This environment currently treats state as a discrete label and does not
        condition reward probabilities on state.
    outcome_range : tuple[float, float], optional
        Range for outcomes. Defaults to (0, 1).

    Notes
    -----
    If you want contextual Bernoulli bandits (state-dependent probabilities),
    you can extend this class to accept a (n_states, n_actions) probability table.
    """

    p: Sequence[float]
    n_states: int = 1
    outcome_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        p = np.asarray(self.p, dtype=float)
        if p.ndim != 1:
            raise ValueError("BernoulliBanditEnv.p must be a 1D sequence.")
        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("BernoulliBanditEnv.p must be within [0, 1].")
        self._p = p
        self._state: int = 0

    @property
    def spec(self) -> EnvironmentSpec:
        """Return environment specification.

        Returns
        -------
        EnvironmentSpec
            Contract describing a binary-outcome, discrete-state bandit.
        """
        return EnvironmentSpec(
            n_actions=int(self._p.shape[0]),
            outcome_type=OutcomeType.BINARY,
            outcome_range=self.outcome_range,
            outcome_is_bounded=True,
            is_social=False,
            state_kind=StateKind.DISCRETE,
            n_states=int(self.n_states),
            state_shape=None,
        )

    def reset(self, *, rng: np.random.Generator) -> Any:
        """Reset environment state.

        Parameters
        ----------
        rng : numpy.random.Generator
            RNG (unused here, but included for interface consistency).

        Returns
        -------
        int
            Initial state id (0).
        """
        self._state = 0
        return self._state

    def get_state(self) -> Any:
        """Return the current state id."""
        return self._state

    def step(self, *, action: int, rng: np.random.Generator) -> EnvStep:
        """Sample a Bernoulli outcome for the chosen action.

        Parameters
        ----------
        action : int
            Action (arm) index.
        rng : numpy.random.Generator
            RNG used for sampling the Bernoulli outcome.

        Returns
        -------
        EnvStep
            True outcome in {0.0, 1.0}.
        """
        a = int(action)
        p = float(self._p[a])
        y = 1.0 if rng.random() < p else 0.0
        return EnvStep(outcome=y, done=False, info={"p": p})
