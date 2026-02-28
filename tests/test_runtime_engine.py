"""Tests for runtime event semantics."""

from __future__ import annotations

from comp_model_v2.core.events import EventPhase
from comp_model_v2.models import RandomAgent
from comp_model_v2.problems import StationaryBanditProblem
from comp_model_v2.runtime import SimulationConfig, run_episode


def test_run_episode_emits_fixed_phase_order_per_trial() -> None:
    """Runtime should emit one complete phase chain per trial.

    The expected order is ``OBSERVATION -> DECISION -> OUTCOME -> UPDATE``.
    """

    problem = StationaryBanditProblem([1.0, 0.0])
    model = RandomAgent()

    trace = run_episode(problem=problem, model=model, config=SimulationConfig(n_trials=4, seed=9))

    assert len(trace.events) == 4 * 4
    expected = [EventPhase.OBSERVATION, EventPhase.DECISION, EventPhase.OUTCOME, EventPhase.UPDATE]

    for trial_index in range(4):
        phases = [event.phase for event in trace.by_trial(trial_index)]
        assert phases == expected


def test_runtime_decision_keys_match_available_actions() -> None:
    """Decision distributions should exactly match per-trial legal actions."""

    problem = StationaryBanditProblem(
        reward_probabilities=[0.1, 0.2, 0.3],
        action_schedule=[(0, 1), (1, 2), (2,)],
    )
    model = RandomAgent()

    trace = run_episode(problem=problem, model=model, config=SimulationConfig(n_trials=3, seed=1))

    for trial_index in range(3):
        trial_events = trace.by_trial(trial_index)
        observation_event = trial_events[0]
        decision_event = trial_events[1]

        available = set(observation_event.payload["available_actions"])
        distribution = decision_event.payload["distribution"]
        chosen_action = decision_event.payload["action"]

        assert set(distribution.keys()) == available
        assert chosen_action in available


def test_runtime_calls_update_for_noop_models() -> None:
    """No-op models should still receive update callbacks every trial."""

    problem = StationaryBanditProblem([0.4, 0.6])
    model = RandomAgent()

    trace = run_episode(problem=problem, model=model, config=SimulationConfig(n_trials=5, seed=3))

    update_events = [event for event in trace.events if event.phase is EventPhase.UPDATE]
    assert len(update_events) == 5
    assert all(event.payload["update_called"] for event in update_events)
