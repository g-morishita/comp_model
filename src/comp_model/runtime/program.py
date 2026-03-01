"""Trial-program abstractions for multi-phase simulation.

A trial program defines one or more decision nodes per trial. This allows
social or hierarchical tasks to include multiple decision phases while keeping
runtime semantics explicit and deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from comp_model.core.contracts import DecisionContext, DecisionProblem
from comp_model.core.events import SimulationEvent


@dataclass(frozen=True, slots=True)
class DecisionNode:
    """A single decision step in a trial program.

    Parameters
    ----------
    node_id : str
        Stable identifier for this decision node inside a trial.
    actor_id : str, optional
        Actor responsible for selecting the action in this node.
    learner_id : str | None, optional
        Actor model that receives the update callback. If ``None``, the runtime
        updates ``actor_id``.
    """

    node_id: str
    actor_id: str = "subject"
    learner_id: str | None = None


@runtime_checkable
class TrialProgram(Protocol):
    """Protocol for multi-phase trial programs.

    Notes
    -----
    Runtime order for each decision node is fixed:

    ``observe -> decide -> transition -> update``

    The program controls how many decision nodes exist per trial and which
    actor is assigned to each node.
    """

    def reset(self, *, rng: np.random.Generator) -> None:
        """Reset program state before an episode."""

    def decision_nodes(
        self,
        *,
        trial_index: int,
        trial_events: Sequence[SimulationEvent],
    ) -> Sequence[DecisionNode]:
        """Return ordered decision nodes for one trial.

        Parameters
        ----------
        trial_index : int
            Zero-based trial index.
        trial_events : Sequence[SimulationEvent]
            Events already emitted for this trial. Programs may use this as
            context for dynamic node construction.

        Returns
        -------
        Sequence[DecisionNode]
            Ordered decision nodes for the trial.
        """

    def available_actions(
        self,
        *,
        trial_index: int,
        node: DecisionNode,
        trial_events: Sequence[SimulationEvent],
    ) -> Sequence[Any]:
        """Return legal actions for a node in the current trial."""

    def observe(
        self,
        *,
        trial_index: int,
        node: DecisionNode,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
    ) -> Any:
        """Return node-specific observation."""

    def transition(
        self,
        action: Any,
        *,
        trial_index: int,
        node: DecisionNode,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
        rng: np.random.Generator,
    ) -> Any:
        """Apply action and return node-specific outcome."""


class SingleStepProgramAdapter:
    """Adapter that turns a single-step ``DecisionProblem`` into a trial program.

    Parameters
    ----------
    problem : DecisionProblem
        Existing single-step problem implementation.

    Notes
    -----
    This adapter maps each trial to one decision node owned by ``"subject"``,
    allowing single-problem tasks to run on the same generic runtime.
    """

    def __init__(self, problem: DecisionProblem) -> None:
        self._problem = problem

    def reset(self, *, rng: np.random.Generator) -> None:
        """Reset wrapped problem state."""

        self._problem.reset(rng=rng)

    def decision_nodes(
        self,
        *,
        trial_index: int,
        trial_events: Sequence[SimulationEvent],
    ) -> tuple[DecisionNode, ...]:
        """Return one default subject decision node per trial."""

        del trial_index, trial_events
        return (DecisionNode(node_id="decision_0", actor_id="subject"),)

    def available_actions(
        self,
        *,
        trial_index: int,
        node: DecisionNode,
        trial_events: Sequence[SimulationEvent],
    ) -> Sequence[Any]:
        """Delegate action availability to wrapped problem."""

        del node, trial_events
        return self._problem.available_actions(trial_index=trial_index)

    def observe(
        self,
        *,
        trial_index: int,
        node: DecisionNode,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
    ) -> Any:
        """Delegate observation generation to wrapped problem."""

        del trial_index, node, trial_events
        return self._problem.observe(context=context)

    def transition(
        self,
        action: Any,
        *,
        trial_index: int,
        node: DecisionNode,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
        rng: np.random.Generator,
    ) -> Any:
        """Delegate transition to wrapped problem."""

        del trial_index, node, trial_events
        return self._problem.transition(action, context=context, rng=rng)
