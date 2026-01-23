"""
Demonstrator/policy interface for social tasks.

A :class:`Demonstrator` can be used by a generator to produce "other agent" choices
and outcomes for social-learning paradigms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..spec import TaskSpec


class Demonstrator(ABC):
    """
    Abstract base class for demonstrators (other agents).

    Notes
    -----
    A demonstrator is conceptually similar to a model acting inside the environment,
    but this interface is intentionally smaller than :class:`~comp_model_core.interfaces.model.ComputationalModel`.
    """

    @abstractmethod
    def reset(self, *, spec: TaskSpec, rng: np.random.Generator) -> None:
        """
        Reset demonstrator state for a new block.

        Parameters
        ----------
        spec : TaskSpec
            Task specification for the block.
        rng : numpy.random.Generator
            RNG used for stochastic resets.
        """
        ...

    @abstractmethod
    def act(self, *, state: Any, spec: TaskSpec, rng: np.random.Generator) -> int:
        """
        Choose an action.

        Parameters
        ----------
        state : Any
            Current task state/context.
        spec : TaskSpec
            Task specification.
        rng : numpy.random.Generator
            RNG for stochastic policies.

        Returns
        -------
        int
            Action index.
        """
        ...

    @abstractmethod
    def update(
        self,
        *,
        state: Any,
        action: int,
        outcome: float,
        spec: TaskSpec,
        rng: np.random.Generator,
    ) -> None:
        """
        Update internal demonstrator state after observing the outcome.

        Parameters
        ----------
        state : Any
            State/context on the trial.
        action : int
            Action that was taken.
        outcome : float
            True outcome resulting from the action.
        spec : TaskSpec
            Task specification.
        rng : numpy.random.Generator
            RNG for stochastic updates (if any).
        """
        ...
