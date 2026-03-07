"""Trial-program abstractions for ordered multi-phase simulation.

A trial program defines an ordered sequence of steps for each trial. This keeps
timing semantics explicit while allowing programs to interleave observations,
decisions, outcomes, and updates in program-defined order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from comp_model.core.contracts import DecisionContext, DecisionProblem
from comp_model.core.events import EventPhase, SimulationEvent


@dataclass(frozen=True, slots=True)
class ProgramStep:
    """One ordered runtime step inside a trial program.

    Parameters
    ----------
    phase : EventPhase
        Runtime phase executed for this step.
    decision_node_id : str
        Stable identifier tying related observation/decision/outcome/update
        steps to the same decision node. This names one decision instance
        within a trial, not a whole environment or task family.
    actor_id : str, optional
        Actor responsible for the node associated with this step.
    learner_id : str | None, optional
        Learner updated by this step when ``phase == EventPhase.UPDATE``.
        When omitted, the runtime updates ``actor_id``.
    """

    phase: EventPhase
    decision_node_id: str
    actor_id: str = "subject"
    learner_id: str | None = None

    def __post_init__(self) -> None:
        if self.decision_node_id.strip() == "":
            raise ValueError("decision_node_id must be a non-empty string")
        if self.actor_id.strip() == "":
            raise ValueError("actor_id must be a non-empty string")
        if self.learner_id is not None and self.learner_id.strip() == "":
            raise ValueError("learner_id must be a non-empty string when provided")


@runtime_checkable
class TrialProgram(Protocol):
    """Protocol for ordered multi-phase trial programs.

    Notes
    -----
    The program controls the exact per-trial step order. The runtime executes
    the returned sequence as-is, subject to phase-specific dependency checks:

    - ``OBSERVATION`` initializes a node with available actions and an
      observation payload,
    - ``DECISION`` samples an action for a previously observed node,
    - ``OUTCOME`` computes an outcome for a previously decided node,
    - ``UPDATE`` applies one learner update for a previously decided node.
    """

    def reset(self, *, rng: np.random.Generator) -> None:
        """Reset program state before an episode."""

    def trial_steps(
        self,
        *,
        trial_index: int,
        trial_events: Sequence[SimulationEvent],
    ) -> Sequence[ProgramStep]:
        """Return ordered runtime steps for one trial."""

    def available_actions(
        self,
        *,
        trial_index: int,
        step: ProgramStep,
        trial_events: Sequence[SimulationEvent],
    ) -> Sequence[Any]:
        """Return legal actions for an observation step."""

    def observe(
        self,
        *,
        trial_index: int,
        step: ProgramStep,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
    ) -> Any:
        """Return observation payload for an observation step."""

    def transition(
        self,
        action: Any,
        *,
        trial_index: int,
        step: ProgramStep,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
        rng: np.random.Generator,
    ) -> Any:
        """Apply action and return outcome payload for an outcome step."""


class SingleStepProgramAdapter:
    """Adapter that turns a single-step ``DecisionProblem`` into a trial program.

    Parameters
    ----------
    problem : DecisionProblem
        Existing single-step problem implementation.

    Notes
    -----
    ``DecisionProblem`` and ``TrialProgram`` are different interfaces.
    ``run_trial_program(...)`` is the canonical runtime, so this adapter emits
    the minimal ordered step sequence for a single-step problem:
    observation, decision, outcome, then update for one ``"subject"`` node.
    That lets simple problems reuse the same engine and trace format as more
    complex multi-phase programs.
    """

    def __init__(self, problem: DecisionProblem) -> None:
        self._problem = problem

    def reset(self, *, rng: np.random.Generator) -> None:
        """Reset wrapped problem state."""

        self._problem.reset(rng=rng)

    def trial_steps(
        self,
        *,
        trial_index: int,
        trial_events: Sequence[SimulationEvent],
    ) -> tuple[ProgramStep, ...]:
        """Return the canonical four-step single-node sequence for a trial."""

        # A wrapped DecisionProblem has no per-trial dynamic step planning.
        # `del` only removes these local names and does not affect caller state.
        del trial_index, trial_events
        return (
            ProgramStep(phase=EventPhase.OBSERVATION, decision_node_id="decision_0", actor_id="subject"),
            ProgramStep(phase=EventPhase.DECISION, decision_node_id="decision_0", actor_id="subject"),
            ProgramStep(phase=EventPhase.OUTCOME, decision_node_id="decision_0", actor_id="subject"),
            ProgramStep(phase=EventPhase.UPDATE, decision_node_id="decision_0", actor_id="subject"),
        )

    def available_actions(
        self,
        *,
        trial_index: int,
        step: ProgramStep,
        trial_events: Sequence[SimulationEvent],
    ) -> Sequence[Any]:
        """Delegate action availability to wrapped problem."""

        # TrialProgram carries step-local metadata, but DecisionProblem only
        # needs the trial index here. `del` only removes local bindings.
        del step, trial_events
        return self._problem.available_actions(trial_index=trial_index)

    def observe(
        self,
        *,
        trial_index: int,
        step: ProgramStep,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
    ) -> Any:
        """Delegate observation generation to wrapped problem."""

        # Wrapped DecisionProblem implementations consume step metadata through
        # DecisionContext rather than separate program-step arguments.
        del trial_index, step, trial_events
        return self._problem.observe(context=context)

    def transition(
        self,
        action: Any,
        *,
        trial_index: int,
        step: ProgramStep,
        context: DecisionContext[Any],
        trial_events: Sequence[SimulationEvent],
        rng: np.random.Generator,
    ) -> Any:
        """Delegate transition to wrapped problem."""

        # Wrapped DecisionProblem implementations consume step metadata through
        # DecisionContext rather than separate program-step arguments.
        del trial_index, step, trial_events
        return self._problem.transition(action, context=context, rng=rng)


__all__ = ["ProgramStep", "SingleStepProgramAdapter", "TrialProgram"]
