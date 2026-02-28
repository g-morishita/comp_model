"""Tests for replay likelihood engine."""

from __future__ import annotations

import pytest

from comp_model_v2.core.events import EpisodeTrace, EventPhase, SimulationEvent
from comp_model_v2.models import QLearningAgent, RandomAgent
from comp_model_v2.models.q_learning import QLearningConfig
from comp_model_v2.problems import StationaryBanditProblem
from comp_model_v2.runtime import SimulationConfig, replay_episode, run_episode


def test_replay_returns_trialwise_likelihood_records() -> None:
    """Replay should return one likelihood step per trial in order."""

    problem = StationaryBanditProblem([0.2, 0.8])
    generating_model = QLearningAgent(config=QLearningConfig(alpha=0.3, beta=2.0, initial_value=0.0))
    trace = run_episode(problem=problem, model=generating_model, config=SimulationConfig(n_trials=12, seed=7))

    replay_model = QLearningAgent(config=QLearningConfig(alpha=0.3, beta=2.0, initial_value=0.0))
    replay_result = replay_episode(trace=trace, model=replay_model)

    assert len(replay_result.steps) == 12
    assert replay_result.total_log_likelihood <= 0.0
    assert all(0.0 < step.probability <= 1.0 for step in replay_result.steps)
    assert [step.trial_index for step in replay_result.steps] == list(range(12))


def test_replay_rejects_trace_with_illegal_logged_action() -> None:
    """Replay should fail fast when a trace action is outside available actions."""

    trace = EpisodeTrace(
        events=[
            SimulationEvent(
                trial_index=0,
                phase=EventPhase.OBSERVATION,
                payload={"observation": {"trial_index": 0}, "available_actions": (0, 1)},
            ),
            SimulationEvent(
                trial_index=0,
                phase=EventPhase.DECISION,
                payload={"distribution": {0: 0.5, 1: 0.5}, "action": 2},
            ),
            SimulationEvent(
                trial_index=0,
                phase=EventPhase.OUTCOME,
                payload={"outcome": {"reward": 1.0}},
            ),
            SimulationEvent(
                trial_index=0,
                phase=EventPhase.UPDATE,
                payload={"update_called": True},
            ),
        ]
    )

    with pytest.raises(ValueError, match="not available"):
        replay_episode(trace=trace, model=RandomAgent())
