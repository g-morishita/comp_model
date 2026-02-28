"""Tests for Q-learning model behavior."""

from __future__ import annotations

import pytest

from comp_model.models import AsocialQValueSoftmaxModel
from comp_model.models.q_learning import AsocialQValueSoftmaxConfig
from comp_model.problems import StationaryBanditProblem
from comp_model.runtime import SimulationConfig, run_episode


def test_q_learning_single_action_updates_toward_reward_one() -> None:
    """Q value should follow the expected exponential moving-average path."""

    model = AsocialQValueSoftmaxModel(config=AsocialQValueSoftmaxConfig(alpha=0.5, beta=1.0, initial_value=0.0))
    problem = StationaryBanditProblem([1.0])

    run_episode(problem=problem, model=model, config=SimulationConfig(n_trials=3, seed=0))

    q_values = model.q_values_snapshot()
    assert q_values[0] == pytest.approx(0.875)


def test_q_learning_handles_zero_beta_uniform_policy() -> None:
    """When beta is zero, policy should be uniform regardless of learned values."""

    model = AsocialQValueSoftmaxModel(config=AsocialQValueSoftmaxConfig(alpha=0.3, beta=0.0, initial_value=1.0))
    problem = StationaryBanditProblem([0.0, 1.0, 1.0], action_schedule=[(0, 1, 2)])

    trace = run_episode(problem=problem, model=model, config=SimulationConfig(n_trials=1, seed=5))
    decision_payload = trace.by_trial(0)[1].payload

    assert decision_payload["distribution"] == pytest.approx({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})


def test_q_learning_config_validation_rejects_invalid_hyperparameters() -> None:
    """Configuration should fail fast for invalid learning hyperparameters."""

    with pytest.raises(ValueError, match="alpha"):
        AsocialQValueSoftmaxConfig(alpha=1.5)

    with pytest.raises(ValueError, match="beta"):
        AsocialQValueSoftmaxConfig(beta=-0.1)
