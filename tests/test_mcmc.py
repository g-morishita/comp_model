"""Tests for MCMC posterior sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.inference import (
    ActionReplayLikelihood,
    IndependentPriorProgram,
    RandomWalkMetropolisEstimator,
    posterior_samples_from_draws,
    sample_posterior_model,
    sample_posterior_model_from_registry,
    uniform_log_prior,
)
from comp_model.plugins import build_default_registry
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


def test_sample_posterior_model_recovers_reasonable_mean() -> None:
    """MCMC helper should return posterior draws with sensible center."""

    trace = run_episode(
        problem=StationaryBanditProblem([0.5, 0.5]),
        model=FixedChoiceModel(p_right=0.75),
        config=SimulationConfig(n_trials=80, seed=4),
    )

    result = sample_posterior_model(
        trace,
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        prior_program=IndependentPriorProgram(
            {"p_right": uniform_log_prior(lower=0.0, upper=1.0)}
        ),
        initial_params={"p_right": 0.5},
        n_samples=200,
        n_warmup=200,
        thin=2,
        proposal_scales={"p_right": 0.08},
        bounds={"p_right": (0.0, 1.0)},
        random_seed=123,
    )

    assert result.posterior_samples.n_draws == 200
    assert result.diagnostics.n_kept_draws == 200
    assert 0.0 <= result.diagnostics.acceptance_rate <= 1.0
    assert 0.55 <= result.posterior_samples.mean("p_right") <= 0.9
    assert result.map_candidate.log_posterior == pytest.approx(
        result.map_candidate.log_likelihood + result.map_candidate.log_prior
    )


def test_sample_posterior_model_from_registry_runs_end_to_end() -> None:
    """Registry-based posterior helper should sample built-in model params."""

    registry = build_default_registry()
    generating_params = {"alpha": 0.3, "beta": 2.5, "initial_value": 0.0}
    trace = run_episode(
        problem=StationaryBanditProblem([0.2, 0.8]),
        model=registry.create_model("asocial_state_q_value_softmax", **generating_params),
        config=SimulationConfig(n_trials=40, seed=20),
    )

    result = sample_posterior_model_from_registry(
        trace,
        model_component_id="asocial_state_q_value_softmax",
        prior_program=IndependentPriorProgram(
            {
                "alpha": uniform_log_prior(lower=0.0, upper=1.0),
                "beta": uniform_log_prior(lower=0.0, upper=20.0),
                "initial_value": uniform_log_prior(lower=-5.0, upper=5.0),
            }
        ),
        initial_params={"alpha": 0.5, "beta": 2.0, "initial_value": 0.0},
        n_samples=25,
        n_warmup=25,
        thin=1,
        proposal_scales={"alpha": 0.05, "beta": 0.2, "initial_value": 0.1},
        bounds={
            "alpha": (0.0, 1.0),
            "beta": (0.0, 20.0),
            "initial_value": (-5.0, 5.0),
        },
        registry=registry,
        random_seed=21,
    )

    assert result.posterior_samples.n_draws == 25
    assert set(result.posterior_samples.parameter_names) == {
        "alpha",
        "beta",
        "initial_value",
    }
    assert result.compatibility is not None


def test_random_walk_estimator_validates_arguments() -> None:
    """MCMC estimator should reject invalid configuration and proposals."""

    trace = run_episode(
        problem=StationaryBanditProblem([0.5, 0.5]),
        model=FixedChoiceModel(p_right=0.6),
        config=SimulationConfig(n_trials=20, seed=5),
    )
    estimator = RandomWalkMetropolisEstimator(
        likelihood_program=ActionReplayLikelihood(),
        model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
        prior_program=IndependentPriorProgram(
            {"p_right": uniform_log_prior(lower=0.0, upper=1.0)}
        ),
    )

    with pytest.raises(ValueError, match="n_samples must be > 0"):
        estimator.fit(trace, initial_params={"p_right": 0.5}, n_samples=0)

    with pytest.raises(ValueError, match="thin must be > 0"):
        estimator.fit(trace, initial_params={"p_right": 0.5}, n_samples=10, thin=0)

    with pytest.raises(ValueError, match="unknown parameters"):
        estimator.fit(
            trace,
            initial_params={"p_right": 0.5},
            n_samples=10,
            proposal_scales={"unknown": 0.1},
        )


def test_posterior_samples_from_draws_rejects_empty() -> None:
    """Posterior draw conversion should fail on empty draw sequence."""

    with pytest.raises(ValueError, match="must not be empty"):
        posterior_samples_from_draws([])
