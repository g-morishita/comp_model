"""Tests for transformed-parameter MLE and transform primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.core.events import EventPhase
from comp_model.inference import (
    ActionReplayLikelihood,
    TransformedScipyMinimizeMLEEstimator,
    positive_log_transform,
    unit_interval_logit_transform,
)
from comp_model.problems import StationaryBanditProblem
from comp_model.runtime import SimulationConfig, run_episode


@dataclass
class FixedChoiceModel:
    """Toy model with one free right-choice probability parameter."""

    p_right: float

    def start_episode(self) -> None:
        """No-op episode reset."""

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[int],
    ) -> dict[int, float]:
        """Return fixed Bernoulli action probabilities."""

        assert context.available_actions == (0, 1)
        return {0: 1.0 - self.p_right, 1: self.p_right}

    def update(
        self,
        observation: Any,
        action: int,
        outcome: Any,
        *,
        context: DecisionContext[int],
    ) -> None:
        """No-op update."""



def test_parameter_transforms_roundtrip_behavior() -> None:
    """Built-in transforms should roundtrip representative values."""

    unit = unit_interval_logit_transform()
    for value in (-3.0, -1.0, 0.0, 1.0, 3.0):
        theta = unit.forward(value)
        z_back = unit.inverse(theta)
        assert z_back == pytest.approx(value, abs=1e-8)

    positive = positive_log_transform()
    for value in (-2.0, -0.5, 0.0, 1.0, 2.0):
        theta = positive.forward(value)
        z_back = positive.inverse(theta)
        assert z_back == pytest.approx(value, abs=1e-8)


def test_transformed_scipy_mle_recovers_unit_interval_parameter() -> None:
    """Transformed optimizer should recover Bernoulli parameter via logit transform."""

    pytest.importorskip("scipy")

    generating_model = FixedChoiceModel(p_right=0.78)
    problem = StationaryBanditProblem([0.5, 0.5])
    trace = run_episode(problem=problem, model=generating_model, config=SimulationConfig(n_trials=180, seed=31))

    estimator = TransformedScipyMinimizeMLEEstimator(
        likelihood_program=ActionReplayLikelihood(),
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        transforms={"p_right": unit_interval_logit_transform()},
    )
    fit = estimator.fit(
        trace,
        initial_params={"p_right": 0.4},
    )

    decision_actions = [
        int(event.payload["action"]) for event in trace.events if event.phase is EventPhase.DECISION
    ]
    empirical_p_right = sum(action == 1 for action in decision_actions) / float(len(decision_actions))

    assert fit.best.params["p_right"] == pytest.approx(empirical_p_right, abs=1e-4)
    assert 0.0 < fit.best.params["p_right"] < 1.0
