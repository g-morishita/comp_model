"""Tests for model-recovery workflow utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from comp_model.core.contracts import DecisionContext
from comp_model.inference import (
    ActionReplayLikelihood,
    GridSearchMLEEstimator,
    MLECandidate,
    MLEFitResult,
)
from comp_model.models import UniformRandomPolicyModel
from comp_model.problems import StationaryBanditProblem
from comp_model.recovery import CandidateModelSpec, GeneratingModelSpec, run_model_recovery


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
        parameter_grid={"p_right": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    )


def _fit_uniform_policy(trace: Any) -> MLEFitResult:
    """Compute likelihood under fixed uniform-random policy (no fitting)."""

    replay = ActionReplayLikelihood().evaluate(trace, UniformRandomPolicyModel())
    candidate = MLECandidate(params={}, log_likelihood=float(replay.total_log_likelihood))
    return MLEFitResult(best=candidate, candidates=(candidate,))


def test_run_model_recovery_prefers_matching_candidate() -> None:
    """Model recovery should usually select generating-equivalent candidate."""

    result = run_model_recovery(
        problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
        generating_specs=(
            GeneratingModelSpec(
                name="fixed_choice",
                model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                true_params={"p_right": 0.85},
            ),
        ),
        candidate_specs=(
            CandidateModelSpec(name="fixed_choice_mle", fit_function=_fit_fixed_choice, n_parameters=1),
            CandidateModelSpec(name="uniform_random", fit_function=_fit_uniform_policy, n_parameters=0),
        ),
        n_trials=120,
        n_replications_per_generator=6,
        criterion="log_likelihood",
        seed=5,
    )

    assert len(result.cases) == 6
    fixed_hits = result.confusion_matrix["fixed_choice"].get("fixed_choice_mle", 0)
    assert fixed_hits >= 5


def test_run_model_recovery_supports_aic_selection() -> None:
    """Model recovery should support information-criterion based selection."""

    result = run_model_recovery(
        problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
        generating_specs=(
            GeneratingModelSpec(
                name="fixed_choice",
                model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                true_params={"p_right": 0.8},
            ),
        ),
        candidate_specs=(
            CandidateModelSpec(name="fixed_choice_mle", fit_function=_fit_fixed_choice, n_parameters=1),
            CandidateModelSpec(name="uniform_random", fit_function=_fit_uniform_policy, n_parameters=0),
        ),
        n_trials=120,
        n_replications_per_generator=4,
        criterion="aic",
        seed=8,
    )

    assert result.criterion == "aic"
    assert result.confusion_matrix["fixed_choice"].get("fixed_choice_mle", 0) >= 3


def test_run_model_recovery_validates_inputs() -> None:
    """Model recovery API should reject invalid configuration values."""

    with pytest.raises(ValueError, match="generating_specs must not be empty"):
        run_model_recovery(
            problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
            generating_specs=(),
            candidate_specs=(CandidateModelSpec(name="fixed", fit_function=_fit_fixed_choice),),
            n_trials=20,
            n_replications_per_generator=1,
        )

    with pytest.raises(ValueError, match="candidate_specs must not be empty"):
        run_model_recovery(
            problem_factory=lambda: StationaryBanditProblem([0.5, 0.5]),
            generating_specs=(
                GeneratingModelSpec(
                    name="fixed_choice",
                    model_factory=lambda params: FixedChoiceModel(p_right=params["p_right"]),
                    true_params={"p_right": 0.8},
                ),
            ),
            candidate_specs=(),
            n_trials=20,
            n_replications_per_generator=1,
        )
