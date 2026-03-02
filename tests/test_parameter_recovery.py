"""Tests for parameter-recovery workflow utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.demonstrators import FixedSequenceDemonstrator
from comp_model.inference import (
    ActionReplayLikelihood,
    ActorSubsetReplayLikelihood,
    FitSpec,
    GridSearchMLEEstimator,
    fit_model,
)
from comp_model.problems import StationaryBanditProblem, TwoStageSocialBanditProgram
from comp_model.recovery import run_parameter_recovery
from comp_model.runtime import SimulationConfig, run_trial_program


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


def test_run_parameter_recovery_accepts_map_fit_functions() -> None:
    """Recovery should accept MAP fit outputs in addition to MLE outputs."""

    class _MapCandidate:
        def __init__(self, p_right: float) -> None:
            self.params = {"p_right": float(p_right)}
            self.log_likelihood = -10.0
            self.log_posterior = -10.5

    class _MapLikeResult:
        def __init__(self, p_right: float) -> None:
            self.map_candidate = _MapCandidate(p_right)

    def fit_function(trace: Any):
        del trace
        return _MapLikeResult(0.7)

    result = run_parameter_recovery(
        problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        fit_function=fit_function,
        true_parameter_sets=(
            {"p_right": 0.7},
        ),
        n_trials=40,
        seed=17,
    )

    assert len(result.cases) == 1
    assert 0.0 <= result.cases[0].estimated_params["p_right"] <= 1.0
    assert result.cases[0].best_log_posterior is not None
    assert "p_right" in result.mean_absolute_error


def test_run_parameter_recovery_supports_custom_social_trace_factory() -> None:
    """Recovery should support multi-actor traces via trace_factory hook."""

    def fit_function(trace: Any):
        return fit_model(
            trace,
            model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
            fit_spec=FitSpec(
                solver="grid_search",
                parameter_grid={"p_right": [0.2, 0.5, 0.8]},
            ),
            likelihood_program=ActorSubsetReplayLikelihood(
                fitted_actor_id="subject",
                scored_actor_ids=("subject",),
                auto_fill_unmodeled_actors=True,
            ),
        )

    def social_trace_factory(model: Any, simulation_seed: int):
        return run_trial_program(
            program=TwoStageSocialBanditProgram([0.5, 0.5]),
            models={
                "subject": model,
                "demonstrator": FixedSequenceDemonstrator(sequence=[1] * 80),
            },
            config=SimulationConfig(n_trials=80, seed=simulation_seed),
        )

    result = run_parameter_recovery(
        problem_factory=None,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        fit_function=fit_function,
        true_parameter_sets=({"p_right": 0.8},),
        n_trials=80,
        seed=23,
        trace_factory=social_trace_factory,
    )

    assert len(result.cases) == 1
    assert result.cases[0].estimated_params["p_right"] == pytest.approx(0.8)


def test_run_parameter_recovery_requires_problem_or_trace_factory() -> None:
    """Recovery should reject missing simulation source definitions."""

    with pytest.raises(ValueError, match="either problem_factory or trace_factory must be provided"):
        run_parameter_recovery(
            problem_factory=None,
            model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
            fit_function=lambda trace: GridSearchMLEEstimator(
                likelihood_program=ActionReplayLikelihood(),
                model_factory=lambda fit_params: FixedChoiceModel(
                    p_right=fit_params["p_right"],
                ),
            ).fit(trace=trace, parameter_grid={"p_right": [0.5]}),
            true_parameter_sets=({"p_right": 0.5},),
            n_trials=10,
        )
