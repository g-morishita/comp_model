"""Simulation engines for generic decision problems and trial programs.

This module provides three runtime entry points:

- :func:`run_trial_program`: ordered multi-phase, multi-actor episode runner.
- :func:`run_episode`: single-problem convenience wrapper.
- :func:`run_social_episode`: social two-actor convenience wrapper.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from comp_model.core.contracts import AgentModel, DecisionContext, DecisionProblem
from comp_model.core.events import EpisodeTrace, EventPhase, SimulationEvent
from comp_model.runtime.probabilities import normalize_distribution, sample_action
from comp_model.runtime.program import ProgramStep, SingleStepProgramAdapter, TrialProgram


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Runtime configuration for one simulation episode.

    Parameters
    ----------
    n_trials : int
        Number of trials to simulate.
    seed : int | None, optional
        Seed used to initialize the random generator. ``None`` uses
        NumPy's entropy source.

    Raises
    ------
    ValueError
        If ``n_trials`` is negative.
    """

    n_trials: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_trials < 0:
            raise ValueError("n_trials must be >= 0")


@dataclass(slots=True)
class _NodeExecutionState:
    """Mutable per-node execution state tracked within one trial."""

    context: DecisionContext[Any]
    observation: Any
    action: Any | None = None
    outcome: Any = None
    decision_done: bool = False
    outcome_done: bool = False


def run_trial_program(
    *,
    program: TrialProgram,
    models: Mapping[str, AgentModel],
    config: SimulationConfig,
) -> EpisodeTrace:
    """Run a full episode from an ordered trial program.

    Parameters
    ----------
    program : TrialProgram
        Trial program that defines ordered steps and transition semantics.
    models : Mapping[str, AgentModel]
        Actor-ID to model mapping. Every actor and learner referenced by the
        program must exist in this mapping.
    config : SimulationConfig
        Episode runtime options.

    Returns
    -------
    EpisodeTrace
        Canonical event trace in the exact order emitted by the program.

    Raises
    ------
    ValueError
        If model mapping is empty, step list is empty, or a step violates
        runtime dependency constraints.
    """

    if not models:
        raise ValueError("models mapping must include at least one actor model")

    rng = np.random.default_rng(config.seed)
    program.reset(rng=rng)

    for model in _unique_models(models):
        model.start_episode()

    # Episode-wide event log accumulated across all trials and returned as the
    # final trace.
    events: list[SimulationEvent] = []

    for trial_index in range(config.n_trials):
        # Per-trial event log reset on each trial. Programs can inspect this to
        # condition later steps on earlier events from the same trial only.
        trial_events: list[SimulationEvent] = []
        steps = tuple(program.trial_steps(trial_index=trial_index, trial_events=tuple(trial_events)))
        if len(steps) == 0:
            raise ValueError(f"program returned no steps for trial {trial_index}")

        # decision_node_id names one decision instance inside the current trial.
        # All later phases for that decision look up the same state entry.
        node_states: dict[str, _NodeExecutionState] = {}
        next_decision_index = 0

        for step in steps:
            if step.phase is EventPhase.OBSERVATION:
                if step.decision_node_id in node_states:
                    raise ValueError(
                        "trial "
                        f"{trial_index}: duplicate observation step for decision node "
                        f"{step.decision_node_id!r}"
                    )

                available_actions = tuple(
                    program.available_actions(
                        trial_index=trial_index,
                        step=step,
                        trial_events=tuple(trial_events),
                    )
                )
                context = DecisionContext(
                    trial_index=trial_index,
                    available_actions=available_actions,
                    actor_id=step.actor_id,
                    decision_index=next_decision_index,
                    decision_label=step.decision_node_id,
                )
                next_decision_index += 1

                observation = program.observe(
                    trial_index=trial_index,
                    step=step,
                    context=context,
                    trial_events=tuple(trial_events),
                )
                node_states[step.decision_node_id] = _NodeExecutionState(
                    context=context,
                    observation=observation,
                )
                _append_trial_event(
                    all_events=events,
                    trial_events=trial_events,
                    event=SimulationEvent(
                        trial_index=trial_index,
                        phase=EventPhase.OBSERVATION,
                        payload={
                            "observation": observation,
                            "available_actions": available_actions,
                            "actor_id": step.actor_id,
                            "decision_node_id": step.decision_node_id,
                        },
                    ),
                )
                continue

            state = _get_node_state(node_states=node_states, step=step, trial_index=trial_index)

            if step.phase is EventPhase.DECISION:
                if state.decision_done:
                    raise ValueError(
                        "trial "
                        f"{trial_index}: duplicate decision step for decision node "
                        f"{step.decision_node_id!r}"
                    )
                if step.actor_id != state.context.actor_id:
                    raise ValueError(
                        f"trial {trial_index}: decision actor {step.actor_id!r} does not match "
                        f"observation actor {state.context.actor_id!r} for decision node "
                        f"{step.decision_node_id!r}"
                    )

                actor_model = _get_actor_model(models=models, actor_id=step.actor_id)
                raw_distribution = actor_model.action_distribution(
                    state.observation,
                    context=state.context,
                )
                distribution = normalize_distribution(raw_distribution, state.context.available_actions)
                action = sample_action(distribution, rng)
                state.action = action
                state.decision_done = True
                _append_trial_event(
                    all_events=events,
                    trial_events=trial_events,
                    event=SimulationEvent(
                        trial_index=trial_index,
                        phase=EventPhase.DECISION,
                        payload={
                            "distribution": distribution,
                            "action": action,
                            "actor_id": step.actor_id,
                            "decision_index": state.context.decision_index,
                            "decision_node_id": step.decision_node_id,
                        },
                    ),
                )
                continue

            if step.phase is EventPhase.OUTCOME:
                if not state.decision_done:
                    raise ValueError(
                        f"trial {trial_index}: outcome step for decision node "
                        f"{step.decision_node_id!r} "
                        "requires a prior decision"
                    )
                if state.outcome_done:
                    raise ValueError(
                        f"trial {trial_index}: duplicate outcome step for decision node "
                        f"{step.decision_node_id!r}"
                    )
                if step.actor_id != state.context.actor_id:
                    raise ValueError(
                        f"trial {trial_index}: outcome actor {step.actor_id!r} does not match "
                        f"decision actor {state.context.actor_id!r} for decision node "
                        f"{step.decision_node_id!r}"
                    )
                assert state.action is not None
                outcome = program.transition(
                    state.action,
                    trial_index=trial_index,
                    step=step,
                    context=state.context,
                    trial_events=tuple(trial_events),
                    rng=rng,
                )
                state.outcome = outcome
                state.outcome_done = True
                _append_trial_event(
                    all_events=events,
                    trial_events=trial_events,
                    event=SimulationEvent(
                        trial_index=trial_index,
                        phase=EventPhase.OUTCOME,
                        payload={
                            "outcome": outcome,
                            "actor_id": step.actor_id,
                            "decision_index": state.context.decision_index,
                            "decision_node_id": step.decision_node_id,
                        },
                    ),
                )
                continue

            if step.phase is EventPhase.UPDATE:
                if not state.decision_done:
                    raise ValueError(
                        f"trial {trial_index}: update step for decision node "
                        f"{step.decision_node_id!r} "
                        "requires a prior decision"
                    )

                learner_id = step.learner_id if step.learner_id is not None else state.context.actor_id
                learner_model = _get_actor_model(models=models, actor_id=learner_id)
                learner_context = (
                    state.context
                    if learner_id == state.context.actor_id
                    else replace(state.context, actor_id=learner_id)
                )
                assert state.action is not None
                learner_model.update(
                    state.observation,
                    state.action,
                    state.outcome,
                    context=learner_context,
                )
                _append_trial_event(
                    all_events=events,
                    trial_events=trial_events,
                    event=SimulationEvent(
                        trial_index=trial_index,
                        phase=EventPhase.UPDATE,
                        payload={
                            "update_called": True,
                            "action": state.action,
                            "actor_id": state.context.actor_id,
                            "learner_id": learner_id,
                            "decision_index": state.context.decision_index,
                            "decision_node_id": step.decision_node_id,
                        },
                    ),
                )
                continue

            raise ValueError(f"trial {trial_index}: unsupported event phase {step.phase!r}")

    return EpisodeTrace(events=events)


def run_episode(problem: DecisionProblem, model: AgentModel, config: SimulationConfig) -> EpisodeTrace:
    """Run a single-problem episode via the generic trial-program engine.

    Parameters
    ----------
    problem : DecisionProblem
        Single-step problem implementation.
    model : AgentModel
        Subject model implementation.
    config : SimulationConfig
        Episode runtime options.

    Returns
    -------
    EpisodeTrace
        Canonical trace emitted by the ordered trial-program runtime.
    """

    program = SingleStepProgramAdapter(problem)
    return run_trial_program(program=program, models={"subject": model}, config=config)


def run_social_episode(
    *,
    program: TrialProgram,
    subject_model: AgentModel,
    demonstrator_model: AgentModel,
    config: SimulationConfig,
    subject_actor_id: str = "subject",
    demonstrator_actor_id: str = "demonstrator",
) -> EpisodeTrace:
    """Run a social program with explicit subject/demonstrator model arguments.

    Parameters
    ----------
    program : TrialProgram
        Ordered social trial program.
    subject_model : AgentModel
        Subject actor model.
    demonstrator_model : AgentModel
        Demonstrator actor model.
    config : SimulationConfig
        Episode runtime options.
    subject_actor_id : str, optional
        Actor ID used by the program for the subject model.
    demonstrator_actor_id : str, optional
        Actor ID used by the program for the demonstrator model.

    Returns
    -------
    EpisodeTrace
        Canonical event trace for the configured episode.

    Raises
    ------
    ValueError
        If actor IDs are empty or collide.
    """

    subject_id = str(subject_actor_id).strip()
    demonstrator_id = str(demonstrator_actor_id).strip()
    if subject_id == "":
        raise ValueError("subject_actor_id must be a non-empty string")
    if demonstrator_id == "":
        raise ValueError("demonstrator_actor_id must be a non-empty string")
    if subject_id == demonstrator_id:
        raise ValueError("subject_actor_id and demonstrator_actor_id must differ")

    return run_trial_program(
        program=program,
        models={
            subject_id: subject_model,
            demonstrator_id: demonstrator_model,
        },
        config=config,
    )


def _get_node_state(
    *,
    node_states: Mapping[str, _NodeExecutionState],
    step: ProgramStep,
    trial_index: int,
) -> _NodeExecutionState:
    """Resolve node execution state or raise a clear dependency error."""

    state = node_states.get(step.decision_node_id)
    if state is None:
        raise ValueError(
            "trial "
            f"{trial_index}: step {step.phase.value!r} for decision node "
            f"{step.decision_node_id!r} "
            "requires a prior observation step"
        )
    return state


def _get_actor_model(models: Mapping[str, AgentModel], actor_id: str) -> AgentModel:
    """Resolve actor model or raise a clear error."""

    model = models.get(actor_id)
    if model is None:
        available = ", ".join(sorted(models))
        raise ValueError(f"unknown actor_id {actor_id!r}; available actors: {available}")
    return model


def _append_trial_event(
    *,
    all_events: list[SimulationEvent],
    trial_events: list[SimulationEvent],
    event: SimulationEvent,
) -> None:
    """Append one event to global and trial-local event buffers."""

    all_events.append(event)
    trial_events.append(event)


def _unique_models(models: Mapping[str, AgentModel]) -> tuple[AgentModel, ...]:
    """Return unique model instances preserving insertion order."""

    unique: list[AgentModel] = []
    seen: set[int] = set()
    for model in models.values():
        key = id(model)
        if key in seen:
            continue
        seen.add(key)
        unique.append(model)
    return tuple(unique)
