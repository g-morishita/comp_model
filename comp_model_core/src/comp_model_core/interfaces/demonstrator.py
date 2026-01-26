"""
comp_model_core.interfaces.demonstrator

Demonstrator/policy interface for social tasks.

A :class:`~comp_model_core.interfaces.demonstrator.Demonstrator` is a small,
policy-like object used to generate "other agent" behavior in social tasks.
It is intentionally simpler than a full computational model.

Notes
-----
- A demonstrator is typically used by a social block runner to produce
  :class:`~comp_model_core.interfaces.block_runner.SocialObservation` objects.
- Unlike a full :class:`~comp_model_core.interfaces.model.ComputationalModel`,
  a demonstrator does not expose parameter schemas or action probabilities; it
  simply produces actions and can update from outcomes.
- Demonstrators operate against an :class:`~comp_model_core.spec.EnvironmentSpec`
  contract rather than a concrete environment class.

See Also
--------
comp_model_core.interfaces.model.ComputationalModel
    Full computational model interface (policy + learning + parameters).
comp_model_core.interfaces.block_runner.SocialObservation
    Container for demonstrator actions/outcomes provided to the subject.
comp_model_core.spec.EnvironmentSpec
    Environment contract used by both models and demonstrators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..spec import EnvironmentSpec


class Demonstrator(ABC):
    """
    Abstract interface for demonstrator policies.

    A demonstrator is used to generate other-agent actions (and potentially
    learn/update) during social tasks. It provides a minimal API:

    - :meth:`~Demonstrator.reset` for block initialization,
    - :meth:`~Demonstrator.act` to choose an action given the current state,
    - :meth:`~Demonstrator.update` to update internal state from the resulting
      true outcome.

    Notes
    -----
    - Demonstrators are typically executed within a block runner rather than by
      the generator directly.
    - Demonstrators consume the *true* outcome from the environment during
      :meth:`~Demonstrator.update`. Any transformation into what the subject
      observes is handled by the runner.

    See Also
    --------
    comp_model_core.interfaces.block_runner.SocialBlockRunner
        Runner interface that produces social observations.
    comp_model_core.interfaces.model.SocialComputationalModel
        Model interface that can update from social observations.
    """

    @abstractmethod
    def reset(self, *, spec: EnvironmentSpec, rng: np.random.Generator) -> None:
        """
        Reset demonstrator state at the start of a block.

        Parameters
        ----------
        spec
            Environment contract for the upcoming block.
        rng
            RNG used to initialize any stochastic demonstrator state.
        """
        ...

    @abstractmethod
    def act(self, *, state: Any, spec: EnvironmentSpec, rng: np.random.Generator) -> int:
        """
        Choose an action for the demonstrator given the current state.

        Parameters
        ----------
        state
            Current environment state/context identifier (opaque to the API).
        spec
            Environment contract (e.g., number of actions).
        rng
            RNG for stochastic action selection.

        Returns
        -------
        int
            Action index selected by the demonstrator.

        Raises
        ------
        Exception
            Implementations may raise if the action cannot be produced (e.g.,
            invalid spec).
        """
        ...

    @abstractmethod
    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float,
        spec: EnvironmentSpec,
        rng: np.random.Generator,
    ) -> None:
        """
        Update demonstrator state from the executed action and true outcome.

        Parameters
        ----------
        state
            Current environment state/context identifier.
        action
            Action index that was taken.
        outcome
            True outcome emitted by the environment for the demonstrator action.
        spec
            Environment contract.
        rng
            RNG for stochastic learning updates (if applicable).

        Notes
        -----
        This method receives the *true* outcome. Observation/noise transformations
        for what the subject sees are handled elsewhere (typically by the runner).
        """
        ...
