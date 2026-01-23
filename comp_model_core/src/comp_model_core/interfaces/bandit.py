"""
Bandit/task interfaces.

A *bandit* (task environment) encapsulates task dynamics and returns outcomes in
response to actions. Models interact with a bandit during simulation, while
estimators typically consume recorded data.

This module defines:

- :class:`BanditStep` as a small container returned by :meth:`Bandit.step`.
- :class:`SocialObservation` as a container for demonstrator observations.
- :class:`Bandit` as the core task interface.
- :class:`SocialBandit` as an extension that supports social observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from abc import ABC, abstractmethod

import numpy as np

from ..spec import TaskSpec


@dataclass(frozen=True, slots=True)
class BanditStep:
    """
    Result of a task step.

    Parameters
    ----------
    outcome : float
        True environment outcome.
    observed_outcome : float or None
        Outcome observed by the agent/subject. ``None`` indicates the outcome was
        hidden from the agent.
    done : bool, optional
        Whether the episode/block should terminate.
    info : dict[str, Any] or None, optional
        Additional task-specific information.

    Attributes
    ----------
    outcome : float
    observed_outcome : float or None
    done : bool
    info : dict[str, Any] or None
    """

    outcome: float
    observed_outcome: float | None
    done: bool = False
    info: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SocialObservation:
    """
    Social observation about another agent/demonstrator.

    Parameters
    ----------
    others_choices : Sequence[int] or None, optional
        Demonstrator action(s), typically length 1 in simple paradigms.
    others_outcomes : Sequence[float] or None, optional
        True demonstrator outcome(s) from the environment.
    observed_others_outcomes : Sequence[float] or None, optional
        Demonstrator outcome(s) as observed by the subject. ``None`` indicates the
        subject did not observe demonstrator outcomes.
    info : dict[str, Any] or None, optional
        Arbitrary metadata (e.g., whether the outcome was revealed).

    Attributes
    ----------
    others_choices : Sequence[int] or None
    others_outcomes : Sequence[float] or None
    observed_others_outcomes : Sequence[float] or None
    info : dict[str, Any] or None
    """

    others_choices: Sequence[int] | None = None
    others_outcomes: Sequence[float] | None = None
    observed_others_outcomes: Sequence[float] | None = None
    info: dict[str, Any] | None = None


class Bandit(ABC):
    """
    Abstract base class for a task environment.

    A bandit exposes a :class:`~comp_model_core.spec.TaskSpec`, provides an initial
    state via :meth:`reset`, and advances dynamics with :meth:`step`.

    Notes
    -----
    The interface is intentionally minimal and does not prescribe episode semantics.
    Higher-level code (generators, wrappers) can decide how to structure blocks.
    """

    @property
    @abstractmethod
    def spec(self) -> TaskSpec:
        """
        Return this task's specification.

        Returns
        -------
        TaskSpec
            Contract defining action space and outcome semantics.
        """
        ...

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> Any:
        """
        Reset the environment to the start of a block/episode.

        Parameters
        ----------
        rng : numpy.random.Generator
            RNG to use for stochastic resets.

        Returns
        -------
        Any
            Initial state identifier. Implementations may also store state internally.
        """
        ...

    @abstractmethod
    def step(self, action: int, rng: np.random.Generator) -> BanditStep:
        """
        Advance the environment by taking an action.

        Parameters
        ----------
        action : int
            Discrete action index in ``[0, spec.n_actions)``.
        rng : numpy.random.Generator
            RNG to use for stochastic dynamics.

        Returns
        -------
        BanditStep
            Result of the environment transition.
        """
        ...

    @abstractmethod
    def get_state(self) -> Any:
        """
        Return the current observable state/context.

        Returns
        -------
        Any
            Current state identifier.
        """
        ...


class SocialBandit(Bandit, ABC):
    """
    Extension of :class:`Bandit` that supports social observations.

    Social bandits provide an :meth:`observe_others` method that returns information
    about a demonstrator/other agent. The timing of when this observation occurs is
    defined by the generator (e.g., pre-choice vs post-outcome).
    """

    @abstractmethod
    def observe_others(self, rng: np.random.Generator) -> SocialObservation:
        """
        Observe another agent/demonstrator.

        Parameters
        ----------
        rng : numpy.random.Generator
            RNG to use for stochastic demonstrator behavior.

        Returns
        -------
        SocialObservation
            The social observation payload.
        """
        ...
