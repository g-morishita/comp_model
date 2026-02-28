"""Simulation engines for generic decision problems and trial programs.

This module provides two runtime entry points:

- :func:`run_trial_program`: multi-phase, multi-actor episode runner.
- :func:`run_episode`: backward-compatible single-step wrapper.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from comp_model_v2.core.contracts import AgentModel, DecisionContext, DecisionProblem
from comp_model_v2.core.events import EpisodeTrace, EventPhase, SimulationEvent
from comp_model_v2.runtime.probabilities import normalize_distribution, sample_action
from comp_model_v2.runtime.program import TrialProgram, SingleStepProgramAdapter


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


def run_trial_program(
    *,
    program: TrialProgram,
    models: Mapping[str, AgentModel],
    config: SimulationConfig,
) -> EpisodeTrace:
    """Run a full episode from a multi-phase trial program.

    Parameters
    ----------
    program : TrialProgram
        Trial program that defines decision nodes and transition semantics.
    models : Mapping[str, AgentModel]
        Actor-ID to model mapping. Every node actor and learner must exist in
        this mapping.
    config : SimulationConfig
        Episode runtime options.

    Returns
    -------
    EpisodeTrace
        Canonical event trace with one phase quartet per decision node:
        ``OBSERVATION -> DECISION -> OUTCOME -> UPDATE``.

    Raises
    ------
    ValueError
        If model mapping is empty, a node list is empty, or a node references
        an unknown actor/learner.
    """

    if not models:
        raise ValueError("models mapping must include at least one actor model")

    rng = np.random.default_rng(config.seed)
    program.reset(rng=rng)

    for model in _unique_models(models):
        model.start_episode()

    events: list[SimulationEvent] = []

    for trial_index in range(config.n_trials):
        trial_events: list[SimulationEvent] = []
        nodes = tuple(program.decision_nodes(trial_index=trial_index, trial_events=tuple(trial_events)))
        if len(nodes) == 0:
            raise ValueError(f"program returned no decision nodes for trial {trial_index}")

        for decision_index, node in enumerate(nodes):
            actor_model = _get_actor_model(models=models, actor_id=node.actor_id)
            available_actions = tuple(
                program.available_actions(
                    trial_index=trial_index,
                    node=node,
                    trial_events=tuple(trial_events),
                )
            )
            context = DecisionContext(
                trial_index=trial_index,
                available_actions=available_actions,
                actor_id=node.actor_id,
                decision_index=decision_index,
                decision_label=node.node_id,
            )

            observation = program.observe(
                trial_index=trial_index,
                node=node,
                context=context,
                trial_events=tuple(trial_events),
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
                        "actor_id": node.actor_id,
                        "decision_index": decision_index,
                        "node_id": node.node_id,
                    },
                ),
            )

            raw_distribution = actor_model.action_distribution(observation, context=context)
            distribution = normalize_distribution(raw_distribution, available_actions)
            action = sample_action(distribution, rng)
            _append_trial_event(
                all_events=events,
                trial_events=trial_events,
                event=SimulationEvent(
                    trial_index=trial_index,
                    phase=EventPhase.DECISION,
                    payload={
                        "distribution": distribution,
                        "action": action,
                        "actor_id": node.actor_id,
                        "decision_index": decision_index,
                        "node_id": node.node_id,
                    },
                ),
            )

            outcome = program.transition(
                action,
                trial_index=trial_index,
                node=node,
                context=context,
                trial_events=tuple(trial_events),
                rng=rng,
            )
            _append_trial_event(
                all_events=events,
                trial_events=trial_events,
                event=SimulationEvent(
                    trial_index=trial_index,
                    phase=EventPhase.OUTCOME,
                    payload={
                        "outcome": outcome,
                        "actor_id": node.actor_id,
                        "decision_index": decision_index,
                        "node_id": node.node_id,
                    },
                ),
            )

            learner_id = node.learner_id if node.learner_id is not None else node.actor_id
            learner_model = _get_actor_model(models=models, actor_id=learner_id)
            learner_context = context if learner_id == context.actor_id else replace(context, actor_id=learner_id)
            learner_model.update(observation, action, outcome, context=learner_context)
            _append_trial_event(
                all_events=events,
                trial_events=trial_events,
                event=SimulationEvent(
                    trial_index=trial_index,
                    phase=EventPhase.UPDATE,
                    payload={
                        "update_called": True,
                        "action": action,
                        "actor_id": node.actor_id,
                        "learner_id": learner_id,
                        "decision_index": decision_index,
                        "node_id": node.node_id,
                    },
                ),
            )

    return EpisodeTrace(events=events)


def run_episode(problem: DecisionProblem, model: AgentModel, config: SimulationConfig) -> EpisodeTrace:
    """Run a backward-compatible single-step episode.

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
        Canonical trace with one decision node per trial.

    Notes
    -----
    This function wraps ``problem`` using
    :class:`comp_model_v2.runtime.program.SingleStepProgramAdapter` and runs the
    new trial-program engine.
    """

    program = SingleStepProgramAdapter(problem)
    return run_trial_program(program=program, models={"subject": model}, config=config)


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
