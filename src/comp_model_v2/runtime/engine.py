"""Simulation engine for generic decision problems.

This runtime is the canonical implementation of the trial loop. It enforces a
single phase order, validates action distributions, and emits a structured trace
for downstream analysis and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from comp_model_v2.core.contracts import AgentModel, DecisionContext, DecisionProblem
from comp_model_v2.core.events import EpisodeTrace, EventPhase, SimulationEvent


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


def run_episode(problem: DecisionProblem, model: AgentModel, config: SimulationConfig) -> EpisodeTrace:
    """Run a full simulation episode and return the event trace.

    Parameters
    ----------
    problem : DecisionProblem
        Decision problem/environment implementation.
    model : AgentModel
        Agent/model implementation.
    config : SimulationConfig
        Episode runtime options.

    Returns
    -------
    EpisodeTrace
        Ordered event sequence with one ``OBSERVATION -> DECISION -> OUTCOME ->
        UPDATE`` chain per trial.

    Notes
    -----
    Distribution normalization is centralized here so model implementations can
    return unnormalized positive weights.
    """

    rng = np.random.default_rng(config.seed)
    problem.reset(rng=rng)
    model.start_episode()

    events: list[SimulationEvent] = []

    for trial_index in range(config.n_trials):
        available_actions = tuple(problem.available_actions(trial_index=trial_index))
        context = DecisionContext(trial_index=trial_index, available_actions=available_actions)

        observation = problem.observe(context=context)
        events.append(
            SimulationEvent(
                trial_index=trial_index,
                phase=EventPhase.OBSERVATION,
                payload={
                    "observation": observation,
                    "available_actions": available_actions,
                },
            )
        )

        raw_distribution = model.action_distribution(observation, context=context)
        distribution = _normalize_distribution(raw_distribution, available_actions)
        action = _sample_action(distribution, rng)
        events.append(
            SimulationEvent(
                trial_index=trial_index,
                phase=EventPhase.DECISION,
                payload={
                    "distribution": distribution,
                    "action": action,
                },
            )
        )

        outcome = problem.transition(action, context=context, rng=rng)
        events.append(
            SimulationEvent(
                trial_index=trial_index,
                phase=EventPhase.OUTCOME,
                payload={"outcome": outcome},
            )
        )

        # Update is always called, even for stateless/no-op models.
        model.update(observation, action, outcome, context=context)
        events.append(
            SimulationEvent(
                trial_index=trial_index,
                phase=EventPhase.UPDATE,
                payload={"update_called": True, "action": action},
            )
        )

    return EpisodeTrace(events=events)


def _normalize_distribution(
    raw_distribution: Mapping[object, float],
    available_actions: tuple[object, ...],
) -> dict[object, float]:
    """Validate and normalize action probabilities.

    Parameters
    ----------
    raw_distribution : Mapping[object, float]
        Model-emitted action weights.
    available_actions : tuple[object, ...]
        Legal actions for the trial.

    Returns
    -------
    dict[object, float]
        Normalized probabilities over ``available_actions``.

    Raises
    ------
    ValueError
        If the model emits invalid keys/values or total probability is zero.
    """

    unknown_actions = set(raw_distribution.keys()) - set(available_actions)
    if unknown_actions:
        raise ValueError(f"distribution contains unknown actions: {sorted(unknown_actions)!r}")

    weights: dict[object, float] = {}
    for action in available_actions:
        value = float(raw_distribution.get(action, 0.0))
        if value < 0:
            raise ValueError(f"distribution contains negative weight for action {action!r}")
        weights[action] = value

    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("distribution sum must be > 0 for available actions")

    return {action: value / total for action, value in weights.items()}


def _sample_action(distribution: Mapping[object, float], rng: np.random.Generator) -> object:
    """Sample one action from a validated distribution."""

    actions = tuple(distribution.keys())
    probs = np.asarray(tuple(distribution.values()), dtype=float)
    return actions[int(rng.choice(len(actions), p=probs))]
