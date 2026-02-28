"""Tests for parameter-recovery workflow utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.inference import ActionReplayLikelihood, GridSearchMLEEstimator
from comp_model.problems import StationaryBanditProblem
from comp_model.recovery import run_parameter_recovery


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



def test_run_parameter_recovery_with_grid_search_fit() -> None:
    """Recovery run should estimate parameters close to truth on grid."""

    def fit_function(trace: Any):
        estimator = GridSearchMLEEstimator(
            likelihood_program=ActionReplayLikelihood(),
            model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        )
        return estimator.fit(
            trace=trace,
            parameter_grid={"p_right": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        )

    result = run_parameter_recovery(
        problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        fit_function=fit_function,
        true_parameter_sets=(
            {"p_right": 0.2},
            {"p_right": 0.8},
        ),
        n_trials=100,
        seed=11,
    )

    assert len(result.cases) == 2
    for case in result.cases:
        assert abs(case.estimated_params["p_right"] - case.true_params["p_right"]) <= 0.1

    assert result.mean_absolute_error["p_right"] <= 0.1


def test_run_parameter_recovery_validates_inputs() -> None:
    """Recovery API should reject invalid configuration inputs."""

    with pytest.raises(ValueError, match="true_parameter_sets must not be empty"):
        run_parameter_recovery(
            problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
            model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
            fit_function=lambda trace: GridSearchMLEEstimator(
                likelihood_program=ActionReplayLikelihood(),
                model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
            ).fit(trace=trace, parameter_grid={"p_right": [0.5]}),
            true_parameter_sets=(),
            n_trials=10,
        )

    with pytest.raises(ValueError, match="n_trials must be > 0"):
        run_parameter_recovery(
            problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
            model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
            fit_function=lambda trace: GridSearchMLEEstimator(
                likelihood_program=ActionReplayLikelihood(),
                model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
            ).fit(trace=trace, parameter_grid={"p_right": [0.5]}),
            true_parameter_sets=({"p_right": 0.5},),
            n_trials=0,
        )
