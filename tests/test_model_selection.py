"""Tests for model-comparison fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.core.data import TrialDecision
from comp_model.inference import (
    BayesFitResult,
    CandidateFitSpec,
    FitSpec,
    MLECandidate,
    MLEFitResult,
    PosteriorCandidate,
    RegistryCandidateFitSpec,
    compare_candidate_models,
    compare_registry_candidate_models,
)
from comp_model.inference.likelihood import ActionReplayLikelihood
from comp_model.inference.mle import GridSearchMLEEstimator
from comp_model.models import UniformRandomPolicyModel
from comp_model.plugins import build_default_registry
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



def _fit_fixed_choice(trace: Any) -> MLEFitResult:
    """Fit fixed-choice model with grid-search MLE."""

    estimator = GridSearchMLEEstimator(
        likelihood_program=ActionReplayLikelihood(),
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
    )
    return estimator.fit(
        trace=trace,
        parameter_grid={"p_right": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]},
    )



def _fit_uniform_policy(trace: Any) -> MLEFitResult:
    """Compute likelihood under fixed uniform-random policy (no fitting)."""

    replay = ActionReplayLikelihood().evaluate(trace, UniformRandomPolicyModel())
    candidate = MLECandidate(params={}, log_likelihood=float(replay.total_log_likelihood))
    return MLEFitResult(best=candidate, candidates=(candidate,))



def _constant_fit(log_likelihood: float, params: dict[str, float]) -> MLEFitResult:
    """Build a constant fit result helper for deterministic criterion tests."""

    candidate = MLECandidate(params=dict(params), log_likelihood=float(log_likelihood))
    return MLEFitResult(best=candidate, candidates=(candidate,))


def _constant_map_fit(
    *,
    log_likelihood: float,
    log_prior: float,
    params: dict[str, float],
) -> BayesFitResult:
    """Build a constant MAP fit result helper for compatibility tests."""

    candidate = PosteriorCandidate(
        params=dict(params),
        log_likelihood=float(log_likelihood),
        log_prior=float(log_prior),
        log_posterior=float(log_likelihood + log_prior),
    )
    return BayesFitResult(map_candidate=candidate, candidates=(candidate,))


def test_compare_candidate_models_prefers_higher_log_likelihood() -> None:
    """Comparison helper should select candidate with higher log-likelihood."""

    generating_model = FixedChoiceModel(p_right=0.8)
    problem = StationaryBanditProblem([0.5, 0.5])
    trace = run_episode(problem=problem, model=generating_model, config=SimulationConfig(n_trials=120, seed=10))

    result = compare_candidate_models(
        trace,
        candidate_specs=(
            CandidateFitSpec(name="fixed_choice", fit_function=_fit_fixed_choice, n_parameters=1),
            CandidateFitSpec(name="uniform_random", fit_function=_fit_uniform_policy, n_parameters=0),
        ),
        criterion="log_likelihood",
    )

    assert result.criterion == "log_likelihood"
    assert result.n_observations == 120
    assert result.selected_candidate_name == "fixed_choice"



def test_compare_candidate_models_infers_bic_observations_from_decisions() -> None:
    """BIC should use decision-event count, including multi-decision trials."""

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
            trial_index=0,
            decision_index=1,
            actor_id="subject",
            available_actions=(0, 1),
            action=0,
            observation={"state": 1},
            outcome={"reward": 0.0},
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

    result = compare_candidate_models(
        decisions,
        candidate_specs=(
            CandidateFitSpec(
                name="simple_model",
                fit_function=lambda trace: _constant_fit(-1.0, {}),
                n_parameters=0,
            ),
            CandidateFitSpec(
                name="complex_model",
                fit_function=lambda trace: _constant_fit(0.0, {"a": 0.1, "b": 0.2}),
                n_parameters=2,
            ),
        ),
        criterion="bic",
    )

    assert result.n_observations == 3
    assert result.selected_candidate_name == "simple_model"



def test_compare_registry_candidate_models_runs_end_to_end() -> None:
    """Registry-based comparison should fit all candidates and select one."""

    registry = build_default_registry()
    generating_model = registry.create_model(
        "asocial_state_q_value_softmax",
        alpha=0.3,
        beta=2.0,
        initial_value=0.0,
    )
    trace = run_episode(
        problem=StationaryBanditProblem([0.2, 0.8]),
        model=generating_model,
        config=SimulationConfig(n_trials=60, seed=7),
    )

    result = compare_registry_candidate_models(
        trace,
        candidate_specs=(
            RegistryCandidateFitSpec(
                name="good",
                model_component_id="asocial_state_q_value_softmax",
                fit_spec=FitSpec(
                    solver="grid_search",
                    parameter_grid={
                        "alpha": [0.3],
                        "beta": [2.0],
                        "initial_value": [0.0],
                    },
                ),
                n_parameters=3,
            ),
            RegistryCandidateFitSpec(
                name="bad",
                model_component_id="asocial_state_q_value_softmax",
                fit_spec=FitSpec(
                    solver="grid_search",
                    parameter_grid={
                        "alpha": [0.95],
                        "beta": [0.1],
                        "initial_value": [1.0],
                    },
                ),
                n_parameters=3,
            ),
        ),
        criterion="log_likelihood",
        registry=registry,
    )

    assert result.selected_candidate_name == "good"
    assert len(result.comparisons) == 2



def test_compare_candidate_models_validates_candidate_specs() -> None:
    """Comparison helper should reject missing candidate specifications."""

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
    )

    with pytest.raises(ValueError, match="candidate_specs must not be empty"):
        compare_candidate_models(decisions, candidate_specs=(), criterion="log_likelihood")


def test_compare_candidate_models_accepts_map_fit_results() -> None:
    """Model comparison should accept MAP-style fit results as candidates."""

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

    result = compare_candidate_models(
        decisions,
        candidate_specs=(
            CandidateFitSpec(
                name="map_good",
                fit_function=lambda trace: _constant_map_fit(
                    log_likelihood=-1.0,
                    log_prior=-0.2,
                    params={"p_right": 0.8},
                ),
                n_parameters=1,
            ),
            CandidateFitSpec(
                name="map_bad",
                fit_function=lambda trace: _constant_map_fit(
                    log_likelihood=-3.0,
                    log_prior=-0.1,
                    params={"p_right": 0.6},
                ),
                n_parameters=1,
            ),
        ),
        criterion="log_likelihood",
    )

    assert result.selected_candidate_name == "map_good"
    assert len(result.comparisons) == 2


def test_compare_candidate_models_waic_rejects_non_posterior_fit_results() -> None:
    """WAIC criterion should fail when candidate fit lacks pointwise draws."""

    trace = run_episode(
        problem=StationaryBanditProblem([0.5, 0.5]),
        model=FixedChoiceModel(p_right=0.8),
        config=SimulationConfig(n_trials=20, seed=12),
    )

    with pytest.raises(ValueError, match="does not support criterion"):
        compare_candidate_models(
            trace,
            candidate_specs=(
                CandidateFitSpec(
                    name="mle_only",
                    fit_function=_fit_fixed_choice,
                    n_parameters=1,
                ),
            ),
            criterion="waic",
        )
