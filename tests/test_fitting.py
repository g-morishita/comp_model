"""Tests for reusable fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.core.data import BlockData, TrialDecision
from comp_model.inference import FitSpec, fit_model
from comp_model.problems import StationaryBanditProblem
from comp_model.runtime import SimulationConfig, run_episode


@dataclass
class FixedChoiceModel:
    """Toy model with one free right-choice probability parameter."""

    p_right: float

    def start_episode(self) -> None:
        """No-op reset."""

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



def test_fit_model_on_episode_trace_with_grid_search() -> None:
    """fit_model should maximize likelihood for trace inputs."""

    generating_model = FixedChoiceModel(p_right=0.8)
    problem = StationaryBanditProblem([0.5, 0.5])
    trace = run_episode(problem=problem, model=generating_model, config=SimulationConfig(n_trials=100, seed=10))

    fit = fit_model(
        trace,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        fit_spec=FitSpec(
            solver="grid_search",
            parameter_grid={"p_right": [0.2, 0.5, 0.8]},
        ),
    )

    assert fit.best.params["p_right"] == pytest.approx(0.8)


def test_fit_model_accepts_block_data_with_trial_rows() -> None:
    """fit_model should accept BlockData datasets by coercing to episode trace."""

    decisions = (
        TrialDecision(
            trial_index=0,
            decision_index=0,
            actor_id="subject",
            available_actions=(0, 1),
            action=1,
            observation={"state": 0},
            outcome={"reward": 1.0},
        ),
        TrialDecision(
            trial_index=1,
            decision_index=0,
            actor_id="subject",
            available_actions=(0, 1),
            action=1,
            observation={"state": 0},
            outcome={"reward": 1.0},
        ),
    )
    block = BlockData(block_id="b0", trials=decisions)

    fit = fit_model(
        block,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        fit_spec=FitSpec(
            solver="grid_search",
            parameter_grid={"p_right": [0.1, 0.9]},
        ),
    )

    assert fit.best.params["p_right"] == pytest.approx(0.9)


def test_fit_model_rejects_missing_estimator_inputs() -> None:
    """fit_model should enforce estimator-specific FitSpec requirements."""

    decisions = (
        TrialDecision(
            trial_index=0,
            decision_index=0,
            actor_id="subject",
            available_actions=(0, 1),
            action=0,
            observation={"state": 0},
            outcome={"reward": 0.0},
        ),
    )

    with pytest.raises(ValueError, match="parameter_grid is required"):
        fit_model(
            decisions,
            model_factory=lambda params: FixedChoiceModel(p_right=params.get("p_right", 0.5)),
            fit_spec=FitSpec(solver="grid_search"),
        )


def test_fit_model_supports_high_level_mle_inference_defaults() -> None:
    """FitSpec should allow inference='mle' without explicitly naming solver."""

    generating_model = FixedChoiceModel(p_right=0.8)
    problem = StationaryBanditProblem([0.5, 0.5])
    trace = run_episode(problem=problem, model=generating_model, config=SimulationConfig(n_trials=80, seed=7))

    fit = fit_model(
        trace,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        fit_spec=FitSpec(
            inference="mle",
            initial_params={"p_right": 0.4},
            bounds={"p_right": (0.0, 1.0)},
        ),
    )

    assert 0.0 <= fit.best.params["p_right"] <= 1.0


def test_fit_model_rejects_bayesian_inference_flag() -> None:
    """fit_model should fail fast when FitSpec requests Bayesian inference."""

    decisions = (
        TrialDecision(
            trial_index=0,
            decision_index=0,
            actor_id="subject",
            available_actions=(0, 1),
            action=0,
            observation={"state": 0},
            outcome={"reward": 0.0},
        ),
    )

    with pytest.raises(ValueError, match="supports only inference='mle'"):
        fit_model(
            decisions,
            model_factory=lambda params: FixedChoiceModel(p_right=params.get("p_right", 0.5)),
            fit_spec=FitSpec(inference="bayesian"),
        )


def test_fit_model_rejects_single_start_for_scipy_solver() -> None:
    """SciPy-based MLE should require multi-start optimization."""

    decisions = (
        TrialDecision(
            trial_index=0,
            decision_index=0,
            actor_id="subject",
            available_actions=(0, 1),
            action=0,
            observation={"state": 0},
            outcome={"reward": 0.0},
        ),
    )

    with pytest.raises(ValueError, match="n_starts must be >= 2"):
        fit_model(
            decisions,
            model_factory=lambda params: FixedChoiceModel(p_right=params.get("p_right", 0.5)),
            fit_spec=FitSpec(
                inference="mle",
                solver="scipy_minimize",
                initial_params={"p_right": 0.5},
                bounds={"p_right": (0.0, 1.0)},
                n_starts=1,
            ),
        )


def test_fit_model_multi_start_is_reproducible_with_seed() -> None:
    """Seeded multi-start SciPy fitting should be reproducible."""

    generating_model = FixedChoiceModel(p_right=0.8)
    problem = StationaryBanditProblem([0.5, 0.5])
    trace = run_episode(problem=problem, model=generating_model, config=SimulationConfig(n_trials=60, seed=21))

    fit1 = fit_model(
        trace,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        fit_spec=FitSpec(
            inference="mle",
            solver="scipy_minimize",
            initial_params={"p_right": 0.5},
            bounds={"p_right": (0.0, 1.0)},
            n_starts=5,
            random_seed=123,
        ),
    )
    fit2 = fit_model(
        trace,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        fit_spec=FitSpec(
            inference="mle",
            solver="scipy_minimize",
            initial_params={"p_right": 0.5},
            bounds={"p_right": (0.0, 1.0)},
            n_starts=5,
            random_seed=123,
        ),
    )

    assert fit1.best.params == pytest.approx(fit2.best.params)
    assert fit1.best.log_likelihood == pytest.approx(fit2.best.log_likelihood)


def test_fit_model_from_registry_component_id() -> None:
    """fit_model_from_registry should fit built-in model components directly."""

    from comp_model.inference import fit_model_from_registry

    generating_model = FixedChoiceModel(p_right=0.8)
    problem = StationaryBanditProblem([0.5, 0.5])
    trace = run_episode(problem=problem, model=generating_model, config=SimulationConfig(n_trials=100, seed=10))

    fit = fit_model_from_registry(
        trace,
        model_component_id="asocial_state_q_value_softmax",
        fit_spec=FitSpec(
            solver="grid_search",
            parameter_grid={
                "alpha": [0.2],
                "beta": [0.0, 5.0],
                "initial_value": [0.0],
            },
        ),
    )

    assert fit.best.params["beta"] in {0.0, 5.0}
    assert "alpha" in fit.best.params
