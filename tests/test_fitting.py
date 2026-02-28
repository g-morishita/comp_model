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
            estimator_type="grid_search",
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
            estimator_type="grid_search",
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
            fit_spec=FitSpec(estimator_type="grid_search"),
        )
