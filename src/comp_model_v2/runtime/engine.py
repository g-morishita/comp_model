"""Simulation engine for generic decision problems.

This runtime is the canonical implementation of the trial loop. It enforces a
single phase order, validates action distributions, and emits a structured trace
for downstream analysis and inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from comp_model_v2.core.contracts import AgentModel, DecisionContext, DecisionProblem
from comp_model_v2.core.events import EpisodeTrace, EventPhase, SimulationEvent
from comp_model_v2.runtime.probabilities import normalize_distribution, sample_action


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
    Distribution normalization is centralized in
    :mod:`comp_model_v2.runtime.probabilities` so model implementations can
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
        distribution = normalize_distribution(raw_distribution, available_actions)
        action = sample_action(distribution, rng)
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
