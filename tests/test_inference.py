"""Tests for inference compatibility and MLE skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from comp_model_v2.core.contracts import DecisionContext
from comp_model_v2.core.events import EpisodeTrace, EventPhase, SimulationEvent
from comp_model_v2.core.requirements import ComponentRequirements
from comp_model_v2.inference import ActionReplayLikelihood, GridSearchMLEEstimator, check_trace_compatibility
from comp_model_v2.models import RandomAgent
from comp_model_v2.plugins import build_default_registry
from comp_model_v2.problems import StationaryBanditProblem
from comp_model_v2.runtime import SimulationConfig, run_episode


@dataclass
class FixedChoiceModel:
    """Toy model with one free choice-probability parameter."""

    p_right: float

    def start_episode(self) -> None:
        """No-op reset for stateless model."""

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[int],
    ) -> dict[int, float]:
        """Return fixed probabilities for actions 0 and 1."""

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


def test_trace_compatibility_uses_model_requirements() -> None:
    """Compatibility checks should enforce required outcome fields."""

    registry = build_default_registry()
    q_learning_manifest = registry.get("model", "q_learning")
    assert q_learning_manifest.requirements is not None

    problem = StationaryBanditProblem([1.0])
    trace = run_episode(problem=problem, model=RandomAgent(), config=SimulationConfig(n_trials=1, seed=2))

    report = check_trace_compatibility(trace, q_learning_manifest.requirements)
    assert report.is_compatible


def test_trace_compatibility_reports_missing_reward_field() -> None:
    """Compatibility report should list missing required fields."""

    trace = EpisodeTrace(
        events=[
            SimulationEvent(
                trial_index=0,
                phase=EventPhase.OBSERVATION,
                payload={"observation": {"trial_index": 0}, "available_actions": (0,)},
            ),
            SimulationEvent(
                trial_index=0,
                phase=EventPhase.DECISION,
                payload={"distribution": {0: 1.0}, "action": 0},
            ),
            SimulationEvent(
                trial_index=0,
                phase=EventPhase.OUTCOME,
                payload={"outcome": {"value": 1.0}},
            ),
            SimulationEvent(
                trial_index=0,
                phase=EventPhase.UPDATE,
                payload={"update_called": True},
            ),
        ]
    )

    requirements = ComponentRequirements(required_outcome_fields=("reward",))
    report = check_trace_compatibility(trace, requirements)

    assert not report.is_compatible
    assert any("missing outcome field 'reward'" in issue for issue in report.issues)


def test_grid_search_mle_finds_best_parameter_on_replay_likelihood() -> None:
    """Grid-search MLE should maximize replay log-likelihood across candidates."""

    generating_model = FixedChoiceModel(p_right=0.85)
    problem = StationaryBanditProblem([0.5, 0.5])
    trace = run_episode(problem=problem, model=generating_model, config=SimulationConfig(n_trials=80, seed=42))

    estimator = GridSearchMLEEstimator(
        likelihood_program=ActionReplayLikelihood(),
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
    )
    fit = estimator.fit(trace=trace, parameter_grid={"p_right": [0.2, 0.5, 0.85]})

    assert fit.best.params["p_right"] == pytest.approx(0.85)
    assert len(fit.candidates) == 3


def test_grid_search_mle_fails_when_compatibility_fails() -> None:
    """Estimator should raise if declared requirements are not met."""

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
                payload={"distribution": {0: 0.5, 1: 0.5}, "action": 1},
            ),
            SimulationEvent(
                trial_index=0,
                phase=EventPhase.OUTCOME,
                payload={"outcome": {"value": 1.0}},
            ),
            SimulationEvent(
                trial_index=0,
                phase=EventPhase.UPDATE,
                payload={"update_called": True},
            ),
        ]
    )

    estimator = GridSearchMLEEstimator(
        likelihood_program=ActionReplayLikelihood(),
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        requirements=ComponentRequirements(required_outcome_fields=("reward",)),
    )

    with pytest.raises(ValueError, match="not compatible"):
        estimator.fit(trace=trace, parameter_grid={"p_right": [0.5]})
