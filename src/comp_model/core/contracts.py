"""Protocol contracts for problems and agent models.

This module defines the generic contracts for interactive decision-making.
The abstractions are intentionally domain-agnostic so the same runtime can
execute bandits, Markov decision tasks, social learning tasks, and other
problem classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np

ObsProblemT_co = TypeVar("ObsProblemT_co", covariant=True)
ObsModelT_contra = TypeVar("ObsModelT_contra", contravariant=True)
ActionT = TypeVar("ActionT")
OutcomeProblemT_co = TypeVar("OutcomeProblemT_co", covariant=True)
OutcomeModelT_contra = TypeVar("OutcomeModelT_contra", contravariant=True)


@dataclass(frozen=True, slots=True)
class DecisionContext(Generic[ActionT]):
    """Per-trial metadata provided to both problem and model.

    Parameters
    ----------
    trial_index : int
        Zero-based trial index in the current episode.
    available_actions : tuple[ActionT, ...]
        Actions that are legal for the current trial.
    actor_id : str, optional
        Actor performing the decision/update for this step.
    decision_index : int, optional
        Zero-based decision step index within the trial.
    decision_label : str | None, optional
        Optional semantic label for the decision step.

    Raises
    ------
    ValueError
        If no action is available for the trial.

    Notes
    -----
    The context object is immutable so every trial has an explicit, auditable
    action set seen by both the environment and the model.
    """

    trial_index: int
    available_actions: tuple[ActionT, ...]
    actor_id: str = "subject"
    decision_index: int = 0
    decision_label: str | None = None

    def __post_init__(self) -> None:
        if len(self.available_actions) == 0:
            raise ValueError("available_actions must contain at least one action")


@runtime_checkable
class DecisionProblem(Protocol[ObsProblemT_co, ActionT, OutcomeProblemT_co]):
    """Interface for decision problems/environments.

    Methods in this protocol define the environment side of one decision step.

    Notes
    -----
    The runtime applies the step order ``observe -> decide -> transition``.
    The problem implementation owns state transitions and outcome generation.
    """

    def reset(self, *, rng: np.random.Generator) -> None:
        """Reset internal state before a new episode.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random generator supplied by the runtime for deterministic seeding.
        """

    def available_actions(self, *, trial_index: int) -> Sequence[ActionT]:
        """Return legal actions for a trial.

        Parameters
        ----------
        trial_index : int
            Zero-based trial index.

        Returns
        -------
        Sequence[ActionT]
            Ordered action collection valid at the given trial.
        """

    def observe(self, *, context: DecisionContext[ActionT]) -> ObsProblemT_co:
        """Generate an observation for the current trial.

        Parameters
        ----------
        context : DecisionContext[ActionT]
            Immutable per-trial metadata.

        Returns
        -------
        ObsProblemT_co
            Observation consumed by the model policy.
        """

    def transition(
        self,
        action: ActionT,
        *,
        context: DecisionContext[ActionT],
        rng: np.random.Generator,
    ) -> OutcomeProblemT_co:
        """Apply an action and return trial outcome.

        Parameters
        ----------
        action : ActionT
            Action selected by the model.
        context : DecisionContext[ActionT]
            Immutable per-trial metadata.
        rng : numpy.random.Generator
            Random generator supplied by the runtime.

        Returns
        -------
        OutcomeProblemT_co
            Outcome emitted by the problem after action execution.
        """


@runtime_checkable
class AgentModel(Protocol[ObsModelT_contra, ActionT, OutcomeModelT_contra]):
    """Interface for models/agents that interact with decision problems.

    Notes
    -----
    Implementations may maintain arbitrary internal state. The runtime calls
    ``start_episode`` once, then repeatedly calls ``action_distribution`` and
    ``update`` for each trial.
    """

    def start_episode(self) -> None:
        """Reset model internal state for a new episode."""

    def action_distribution(
        self,
        observation: ObsModelT_contra,
        *,
        context: DecisionContext[ActionT],
    ) -> Mapping[ActionT, float]:
        """Compute a probability distribution over available actions.

        Parameters
        ----------
        observation : ObsModelT_contra
            Observation emitted by the problem.
        context : DecisionContext[ActionT]
            Immutable per-trial metadata.

        Returns
        -------
        Mapping[ActionT, float]
            Action-probability mapping. Probabilities do not need to be
            pre-normalized; the runtime normalizes robustly.
        """

    def update(
        self,
        observation: ObsModelT_contra,
        action: ActionT,
        outcome: OutcomeModelT_contra,
        *,
        context: DecisionContext[ActionT],
    ) -> None:
        """Update model state after receiving outcome.

        Parameters
        ----------
        observation : ObsModelT_contra
            Trial observation seen before choosing the action.
        action : ActionT
            Action sampled by the runtime.
        outcome : OutcomeModelT_contra
            Trial outcome returned by the problem.
        context : DecisionContext[ActionT]
            Immutable per-trial metadata.

        Notes
        -----
        Stateless policies may implement this as a no-op.
        """
